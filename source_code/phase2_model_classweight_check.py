"""Compare SMOTE-calibrated vs class-weight-calibrated XGBoost on Phase 2 data."""

from pathlib import Path
import ctypes
import importlib
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_xgb_classifier() -> type:
    """Import XGBClassifier with a macOS fallback for missing libomp runtime."""
    try:
        return importlib.import_module("xgboost").XGBClassifier
    except Exception as first_err:
        import sklearn

        sklearn_libomp = Path(sklearn.__file__).resolve().parent / ".dylibs" / "libomp.dylib"
        if sklearn_libomp.exists():
            lib_dir = str(sklearn_libomp.parent)
            existing = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
            try:
                ctypes.CDLL(str(sklearn_libomp))
            except Exception:
                pass
            return importlib.import_module("xgboost").XGBClassifier
        raise RuntimeError(
            "Unable to import xgboost. Install macOS OpenMP runtime (libomp) or ensure sklearn .dylibs is available."
        ) from first_err


def metric_row(name: str, y_true: pd.Series, y_prob: np.ndarray) -> dict:
    return {
        "model": name,
        "brier_score": brier_score_loss(y_true, y_prob),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
    }


def reliability_table(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"[{bins[i]:.1f},{bins[i + 1]:.1f}]" for i in range(10)]

    df_rel = pd.DataFrame({"y_true": y_true.to_numpy(), "y_prob": y_prob})
    df_rel["bin"] = pd.cut(df_rel["y_prob"], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        df_rel.groupby("bin", observed=False)
        .agg(mean_predicted=("y_prob", "mean"), mean_actual=("y_true", "mean"), count=("y_true", "size"))
        .reset_index()
    )
    grouped["bin_centre"] = [0.05 + 0.1 * i for i in range(10)]
    grouped["gap_pred_minus_actual"] = grouped["mean_predicted"] - grouped["mean_actual"]
    grouped["overprediction_excess"] = grouped["gap_pred_minus_actual"].clip(lower=0)
    return grouped[[
        "bin",
        "bin_centre",
        "mean_predicted",
        "mean_actual",
        "count",
        "gap_pred_minus_actual",
        "overprediction_excess",
    ]]


def weighted_overprediction_above_point_two(rel_df: pd.DataFrame) -> float:
    subset = rel_df[(rel_df["bin_centre"] > 0.2) & (rel_df["count"] > 0)].copy()
    if subset.empty:
        return 0.0
    return float(np.average(subset["overprediction_excess"], weights=subset["count"]))


def main() -> None:
    root = Path(__file__).parent.parent
    XGBClassifier = load_xgb_classifier()

    data_path = root / "data" / "processed" / "phase2_features_final.csv"
    feature_list_path = root / "data" / "processed" / "model_feature_list.txt"
    existing_model_path = root / "data" / "models" / "xgb_calibrated.pkl"
    existing_preds_path = root / "data" / "models" / "test_predictions.csv"

    new_model_path = root / "data" / "models" / "xgb_classweight_calibrated.pkl"
    new_preds_path = root / "data" / "models" / "classweight_test_predictions.csv"

    print_header("Step 0 - Load and validate")
    df = pd.read_csv(data_path)
    df["is_strategic_stop"] = df["is_strategic_stop"].astype(pd.BooleanDtype())
    df["sc_active"] = df["sc_active"].astype(bool)
    df["vsc_active"] = df["vsc_active"].astype(bool)
    df["caution_active"] = df["caution_active"].astype(bool)

    print(f"Loaded shape: {df.shape[0]} x {df.shape[1]}")
    if df.shape != (70549, 69):
        raise ValueError(f"Expected shape (70549, 69), found {df.shape}")

    MODEL_FEATURES = [
        line.strip()
        for line in feature_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded MODEL_FEATURES count: {len(MODEL_FEATURES)}")

    print_header("Step 1 - Two-way temporal split")
    train_df = df[df["Year"] <= 2023].copy()
    test_df = df[df["Year"] == 2024].copy()

    print(f"Train rows (Year <= 2023): {len(train_df)}")
    print(f"Test rows (Year == 2024): {len(test_df)}")
    print(f"Train positive rate (sc_vsc_next3): {train_df['sc_vsc_next3'].mean():.4%}")
    print(f"Test positive rate (sc_vsc_next3): {test_df['sc_vsc_next3'].mean():.4%}")
    if len(train_df) + len(test_df) != len(df):
        raise ValueError("Two-way split lost rows")
    print("Confirmed: no rows lost in split")

    print_header("Step 2 - Extract feature matrices")
    missing_features = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing_features:
        raise KeyError(f"MODEL_FEATURES missing from dataset: {missing_features}")

    X_train = train_df[MODEL_FEATURES].copy()
    X_test = test_df[MODEL_FEATURES].copy()
    y_train = train_df["sc_vsc_next3"].astype(int).copy()
    y_test = test_df["sc_vsc_next3"].astype(int).copy()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    null_counts = {
        "X_train": int(X_train.isna().sum().sum()),
        "X_test": int(X_test.isna().sum().sum()),
        "y_train": int(y_train.isna().sum()),
        "y_test": int(y_test.isna().sum()),
    }
    for name, count in null_counts.items():
        print(f"{name} null count: {count}")
    if any(count != 0 for count in null_counts.values()):
        raise ValueError("Nulls found in feature matrices or targets")

    print_header("Step 3 - Train calibrated class-weighted XGBoost (no SMOTE)")
    class_counts = y_train.value_counts().sort_index()
    print("Training class counts:")
    print(class_counts.to_string())
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    if pos_count == 0:
        raise ValueError("No positive class in y_train")
    scale_pos_weight = neg_count / pos_count
    print(f"scale_pos_weight (neg/pos): {scale_pos_weight:.6f}")

    xgb_classweight_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
    )
    xgb_classweight_cal = CalibratedClassifierCV(xgb_classweight_base, cv=5, method="sigmoid")
    xgb_classweight_cal.fit(X_train, y_train)
    y_pred_classweight = xgb_classweight_cal.predict_proba(X_test)[:, 1]

    cw_metrics = metric_row("Class-weight calibrated", y_test, y_pred_classweight)
    print("Class-weight calibrated XGBoost metrics:")
    print(f"  Brier Score: {cw_metrics['brier_score']:.6f}")
    print(f"  AUC-ROC: {cw_metrics['auc_roc']:.6f}")
    print(f"  Average Precision: {cw_metrics['avg_precision']:.6f}")

    print_header("Step 4 - Load existing SMOTE model and compute comparison metrics")
    if not existing_model_path.exists() or not existing_preds_path.exists():
        raise FileNotFoundError(
            "Expected existing SMOTE artifacts were not found: "
            f"{existing_model_path} and {existing_preds_path}"
        )

    smote_preds_df = pd.read_csv(existing_preds_path)
    smote_model = joblib.load(existing_model_path)

    can_use_preds_file = (
        {"y_true", "y_pred_xgb_cal"}.issubset(smote_preds_df.columns)
        and len(smote_preds_df) == len(y_test)
    )
    if can_use_preds_file:
        y_true_smote = smote_preds_df["y_true"].astype(int)
        y_pred_smote = smote_preds_df["y_pred_xgb_cal"].astype(float).to_numpy()
        print("Using stored SMOTE predictions from test_predictions.csv")
    else:
        y_true_smote = y_test
        y_pred_smote = smote_model.predict_proba(X_test)[:, 1]
        print("Stored predictions unavailable/incompatible; recomputed SMOTE predictions from saved model")

    smote_metrics = metric_row("SMOTE calibrated", y_true_smote, y_pred_smote)
    print("SMOTE-calibrated XGBoost metrics:")
    print(f"  Brier Score: {smote_metrics['brier_score']:.6f}")
    print(f"  AUC-ROC: {smote_metrics['auc_roc']:.6f}")
    print(f"  Average Precision: {smote_metrics['avg_precision']:.6f}")

    print_header("Step 5 - Reliability diagram comparison (10 bins)")
    rel_cw = reliability_table(y_test, y_pred_classweight)
    rel_smote = reliability_table(y_true_smote, y_pred_smote)

    rel_compare = rel_cw.merge(rel_smote, on=["bin", "bin_centre"], suffixes=("_classweight", "_smote"))
    print("Bin-by-bin reliability comparison:")
    print(
        rel_compare.to_string(
            index=False,
            formatters={
                "bin_centre": lambda x: f"{x:.2f}",
                "mean_predicted_classweight": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "mean_actual_classweight": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "mean_predicted_smote": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "mean_actual_smote": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "gap_pred_minus_actual_classweight": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "gap_pred_minus_actual_smote": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "overprediction_excess_classweight": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "overprediction_excess_smote": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
            },
        )
    )

    cw_overpred = weighted_overprediction_above_point_two(rel_cw)
    smote_overpred = weighted_overprediction_above_point_two(rel_smote)
    diff = cw_overpred - smote_overpred
    tol = 1e-9
    if diff < -tol:
        overpred_judgement = "better"
    elif diff > tol:
        overpred_judgement = "worse"
    else:
        overpred_judgement = "unchanged"

    print_header("Step 6 - Final comparison summary")
    summary = pd.DataFrame(
        [
            {
                "model": "SMOTE-calibrated",
                "brier_score": smote_metrics["brier_score"],
                "auc_roc": smote_metrics["auc_roc"],
                "avg_precision": smote_metrics["avg_precision"],
                "weighted_overprediction_above_p0.2": smote_overpred,
            },
            {
                "model": "Class-weight-calibrated",
                "brier_score": cw_metrics["brier_score"],
                "auc_roc": cw_metrics["auc_roc"],
                "avg_precision": cw_metrics["avg_precision"],
                "weighted_overprediction_above_p0.2": cw_overpred,
            },
        ]
    )
    print(summary.to_string(index=False, formatters={
        "brier_score": lambda x: f"{x:.6f}",
        "auc_roc": lambda x: f"{x:.6f}",
        "avg_precision": lambda x: f"{x:.6f}",
        "weighted_overprediction_above_p0.2": lambda x: f"{x:.6f}",
    }))
    print(f"Over-prediction above p=0.2 (class-weight vs SMOTE): {overpred_judgement}")

    print_header("Step 7 - Save new artefacts")
    if new_model_path.exists() or new_preds_path.exists():
        raise FileExistsError(
            "Refusing to overwrite existing files: "
            f"{new_model_path} or {new_preds_path}"
        )

    joblib.dump(xgb_classweight_cal, new_model_path)
    new_preds = test_df[["Year", "Round", "Driver", "LapNumber"]].copy()
    new_preds["y_true"] = y_test.to_numpy()
    new_preds["y_pred_xgb_classweight_cal"] = y_pred_classweight
    new_preds.to_csv(new_preds_path, index=False)

    print(f"Saved class-weighted calibrated model: {new_model_path}")
    print(f"Saved class-weighted test predictions: {new_preds_path}")


if __name__ == "__main__":
    main()