"""Phase 2D modeling pipeline.

Loads phase2_features_final.csv and model_feature_list.txt, trains calibrated
Logistic Regression and calibrated XGBoost models, reports diagnostics, and
saves model artifacts.
"""

from pathlib import Path
import ctypes
import importlib
import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


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


def print_metrics(label: str, row: dict) -> None:
    print(label)
    print(f"  Brier Score: {row['brier_score']:.6f}")
    print(f"  AUC-ROC: {row['auc_roc']:.6f}")
    print(f"  Average Precision: {row['avg_precision']:.6f}")


def main() -> None:
    root = Path(__file__).parent.parent
    XGBClassifier = load_xgb_classifier()

    data_path = root / "data" / "processed" / "phase2_features_final.csv"
    feature_list_path = root / "data" / "processed" / "model_feature_list.txt"
    models_dir = root / "data" / "models"

    print_header("Step 0 - Load and validate")
    df = pd.read_csv(data_path)
    df["sc_vsc_next3"] = df["sc_vsc_next3"].astype(int)
    df["sc_vsc_next5"] = df["sc_vsc_next5"].astype("Int64")
    print(f"Loaded dataset shape: {df.shape[0]} x {df.shape[1]}")
    if df.shape != (70549, 69):
        raise ValueError(f"Expected shape (70549, 69), found {df.shape}")

    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensured models directory exists: {models_dir}")

    MODEL_FEATURES = [
        line.strip()
        for line in feature_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded MODEL_FEATURES count: {len(MODEL_FEATURES)}")

    print_header("Step 1 - Two-way temporal split")
    train_mask = df["Year"] <= 2023
    test_mask = df["Year"] == 2024

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

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
    print("Confirmed: zero nulls in all arrays")

    print_header("Step 3 - SMOTE on training set only")
    pre_counts = y_train.value_counts().sort_index()
    print("Class counts before SMOTE:")
    print(pre_counts.to_string())

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    post_counts = pd.Series(y_train_sm).value_counts().sort_index()
    print("Class counts after SMOTE:")
    print(post_counts.to_string())

    print_header("Step 4 - Logistic Regression baseline with calibration")
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_sm_scaled = scaler.transform(X_train_sm)
    X_test_scaled = scaler.transform(X_test)

    lr_base = LogisticRegression(max_iter=2000, random_state=42)
    lr_base.fit(X_train_sm_scaled, y_train_sm)
    lr_cal = CalibratedClassifierCV(lr_base, cv=5, method="sigmoid")
    lr_cal.fit(X_train_sm_scaled, y_train_sm)
    y_pred_lr_cal = lr_cal.predict_proba(X_test_scaled)[:, 1]

    lr_metrics = metric_row("LR baseline (calibrated)", y_test, y_pred_lr_cal)
    print_metrics("LR baseline (calibrated) results:", lr_metrics)

    print_header("Step 5 - XGBoost model")
    xgb_base = XGBClassifier(
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
    )
    xgb_base.fit(X_train_sm, y_train_sm)
    xgb_importances_source = xgb_base

    xgb_cal = CalibratedClassifierCV(xgb_base, cv=5, method="sigmoid")
    xgb_cal.fit(X_train_sm, y_train_sm)
    y_pred_xgb_cal = xgb_cal.predict_proba(X_test)[:, 1]

    xgb_metrics = metric_row("XGBoost (calibrated)", y_test, y_pred_xgb_cal)
    print_metrics("XGBoost (calibrated) results:", xgb_metrics)

    summary_df = pd.DataFrame([lr_metrics, xgb_metrics])
    print("\nSummary comparison table:")
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "brier_score": lambda x: f"{x:.6f}",
                "auc_roc": lambda x: f"{x:.6f}",
                "avg_precision": lambda x: f"{x:.6f}",
            },
        )
    )

    print_header("Step 6 - Calibration table (10 bins)")
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"[{bins[i]:.1f},{bins[i + 1]:.1f}]" for i in range(10)]
    cal_table_df = pd.DataFrame({"y_true": y_test.to_numpy(), "y_prob": y_pred_xgb_cal})
    cal_table_df["bin"] = pd.cut(cal_table_df["y_prob"], bins=bins, labels=labels, include_lowest=True)

    grouped = (
        cal_table_df.groupby("bin", observed=False)
        .agg(mean_predicted=("y_prob", "mean"), mean_actual=("y_true", "mean"), count=("y_true", "size"))
        .reset_index()
    )
    grouped["bin_centre"] = [0.05 + 0.1 * i for i in range(10)]
    grouped = grouped["bin bin_centre mean_predicted mean_actual count".split()]
    print("Reliability bin table:")
    print(
        grouped.to_string(
            index=False,
            formatters={
                "bin_centre": lambda x: f"{x:.2f}",
                "mean_predicted": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "mean_actual": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
            },
        )
    )

    print_header("Step 7 - Feature importance")
    importances = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": xgb_importances_source.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("XGBoost feature importances (ranked):")
    print(importances.to_string(index=False, formatters={"importance": lambda x: f"{x:.6f}"}))

    print_header("Step 8 - Archetype-stratified Brier Scores")
    overall_brier = brier_score_loss(y_test, y_pred_xgb_cal)
    print(f"Overall Brier Score (calibrated XGBoost): {overall_brier:.6f}")

    archetype_names = {0: "Other", 1: "Power", 2: "Street", 3: "Technical"}
    for code in [0, 1, 2, 3]:
        mask = X_test["circuit_archetype_enc"] == code
        n = int(mask.sum())
        if n == 0:
            print(f"Archetype {code} ({archetype_names[code]}): no rows in test set")
            continue
        brier = brier_score_loss(y_test[mask], y_pred_xgb_cal[mask])
        print(f"Archetype {code} ({archetype_names[code]}): Brier={brier:.6f}, n={n}")

    print_header("Step 9 - Sensitivity check on sc_vsc_next5")
    sens_df = df[df["sc_vsc_next5"].notna()].copy()
    sens_train = sens_df[sens_df["Year"] <= 2023].copy()
    sens_test = sens_df[sens_df["Year"] == 2024].copy()

    X_train_5 = sens_train[MODEL_FEATURES].copy()
    y_train_5 = sens_train["sc_vsc_next5"].astype(int).copy()
    X_test_5 = sens_test[MODEL_FEATURES].copy()
    y_test_5 = sens_test["sc_vsc_next5"].astype(int).copy()

    smote_5 = SMOTE(random_state=42)
    X_train_5_sm, y_train_5_sm = smote_5.fit_resample(X_train_5, y_train_5)

    xgb_base_5 = XGBClassifier(
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
    )
    xgb_base_5.fit(X_train_5_sm, y_train_5_sm)
    xgb_cal_5 = CalibratedClassifierCV(xgb_base_5, cv=5, method="sigmoid")
    xgb_cal_5.fit(X_train_5_sm, y_train_5_sm)
    y_pred_5_cal = xgb_cal_5.predict_proba(X_test_5)[:, 1]

    sens_metrics = metric_row("Sensitivity check (sc_vsc_next5)", y_test_5, y_pred_5_cal)
    print("Sensitivity check (sc_vsc_next5, calibrated XGBoost):")
    print(f"  Brier Score: {sens_metrics['brier_score']:.6f}")
    print(f"  AUC-ROC: {sens_metrics['auc_roc']:.6f}")
    print(f"  Average Precision: {sens_metrics['avg_precision']:.6f}")

    print_header("Step 10 - Save artefacts")
    xgb_cal_path = models_dir / "xgb_calibrated.pkl"
    lr_path = models_dir / "lr_baseline.pkl"
    preds_path = models_dir / "test_predictions.csv"

    joblib.dump(xgb_cal, xgb_cal_path)
    joblib.dump(lr_cal, lr_path)

    test_predictions = test_df[["Year", "Round", "Driver", "LapNumber"]].copy()
    test_predictions["y_true"] = y_test.to_numpy()
    test_predictions["y_pred_lr_cal"] = y_pred_lr_cal
    test_predictions["y_pred_xgb_cal"] = y_pred_xgb_cal
    test_predictions.to_csv(preds_path, index=False)

    print(f"Saved calibrated XGBoost model: {xgb_cal_path}")
    print(f"Saved calibrated Logistic Regression model: {lr_path}")
    print(f"Saved test predictions: {preds_path}")


if __name__ == "__main__":
    main()