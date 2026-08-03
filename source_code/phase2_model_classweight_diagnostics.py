"""Diagnostics for class-weighted calibrated XGBoost versus SMOTE baseline."""

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
        "prob_min": float(np.min(y_prob)),
        "prob_max": float(np.max(y_prob)),
    }


def reliability_table(y_true: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, 11)
    labels = [f"[{bins[i]:.1f},{bins[i + 1]:.1f}]" for i in range(10)]
    rel_df = pd.DataFrame({"y_true": y_true.to_numpy(), "y_prob": y_prob})
    rel_df["bin"] = pd.cut(rel_df["y_prob"], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        rel_df.groupby("bin", observed=False)
        .agg(mean_predicted=("y_prob", "mean"), mean_actual=("y_true", "mean"), count=("y_true", "size"))
        .reset_index()
    )
    grouped["bin_centre"] = [0.05 + 0.1 * i for i in range(10)]
    grouped["gap_pred_minus_actual"] = grouped["mean_predicted"] - grouped["mean_actual"]
    grouped["flag"] = ""
    mask = grouped["count"] > 0
    over = mask & (grouped["gap_pred_minus_actual"] > 0.15)
    under = mask & (grouped["gap_pred_minus_actual"] < -0.15)
    grouped.loc[over, "flag"] = "OVER_PREDICTION_|gap|>0.15"
    grouped.loc[under, "flag"] = "UNDER_PREDICTION_|gap|>0.15"
    return grouped[["bin", "bin_centre", "mean_predicted", "mean_actual", "count", "gap_pred_minus_actual", "flag"]]


def extract_calibrated_xgb_importances(calibrated_model: CalibratedClassifierCV) -> np.ndarray:
    """Average feature_importances_ from calibrated fold estimators."""
    fold_importances = []
    for calibrated in calibrated_model.calibrated_classifiers_:
        estimator = None
        for attr in ["estimator", "base_estimator", "classifier", "model"]:
            if hasattr(calibrated, attr):
                candidate = getattr(calibrated, attr)
                if hasattr(candidate, "feature_importances_"):
                    estimator = candidate
                    break
        if estimator is None and hasattr(calibrated, "estimator"):
            candidate = getattr(calibrated, "estimator")
            if hasattr(candidate, "named_steps") and "xgbclassifier" in candidate.named_steps:
                estimator = candidate.named_steps["xgbclassifier"]
        if estimator is None:
            raise ValueError("Could not locate XGBoost estimator inside calibrated fold")
        fold_importances.append(np.asarray(estimator.feature_importances_, dtype=float))
    return np.mean(np.vstack(fold_importances), axis=0)


def get_archetype_code_series(df: pd.DataFrame) -> pd.Series:
    if "circuit_archetype" in df.columns and pd.api.types.is_numeric_dtype(df["circuit_archetype"]):
        return df["circuit_archetype"].astype(int)
    if "circuit_archetype_enc" in df.columns:
        return df["circuit_archetype_enc"].astype(int)
    if "circuit_archetype" in df.columns:
        mapping = {"Other": 0, "Power": 1, "Street": 2, "Technical": 3}
        return df["circuit_archetype"].map(mapping).astype("Int64")
    raise KeyError("No circuit_archetype or circuit_archetype_enc column found")


def is_inverse_ordering(base_rates: pd.Series, briers: pd.Series) -> bool:
    if len(base_rates) < 2:
        return False
    base_order = list(base_rates.sort_values(ascending=False).index)
    brier_order = list(briers.sort_values(ascending=True).index)
    return base_order == brier_order


def main() -> None:
    root = Path(__file__).parent.parent
    XGBClassifier = load_xgb_classifier()

    data_path = root / "data" / "processed" / "phase2_features_final.csv"
    feature_list_path = root / "data" / "processed" / "model_feature_list.txt"
    classweight_model_path = root / "data" / "models" / "xgb_classweight_calibrated.pkl"
    classweight_preds_path = root / "data" / "models" / "classweight_test_predictions.csv"
    smote_preds_path = root / "data" / "models" / "test_predictions.csv"
    next5_save_path = root / "data" / "models" / "xgb_classweight_next5.pkl"

    print_header("Load artifacts and data")
    if not classweight_model_path.exists() or not classweight_preds_path.exists():
        raise FileNotFoundError(
            "Required class-weight artifacts missing: "
            f"{classweight_model_path} and {classweight_preds_path}"
        )
    if not smote_preds_path.exists():
        raise FileNotFoundError(f"Required SMOTE predictions missing: {smote_preds_path}")

    classweight_model = joblib.load(classweight_model_path)
    classweight_preds_df = pd.read_csv(classweight_preds_path)
    smote_preds_df = pd.read_csv(smote_preds_path)

    df = pd.read_csv(data_path)
    df["is_strategic_stop"] = df["is_strategic_stop"].astype(pd.BooleanDtype())
    df["sc_active"] = df["sc_active"].astype(bool)
    df["vsc_active"] = df["vsc_active"].astype(bool)
    df["caution_active"] = df["caution_active"].astype(bool)

    MODEL_FEATURES = [
        line.strip()
        for line in feature_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    train_df = df[df["Year"] <= 2023].copy()
    test_df = df[df["Year"] == 2024].copy()
    X_train = train_df[MODEL_FEATURES].copy()
    X_test = test_df[MODEL_FEATURES].copy()
    y_test = test_df["sc_vsc_next3"].astype(int).copy()

    if len(classweight_preds_df) == len(test_df) and "y_pred_xgb_classweight_cal" in classweight_preds_df.columns:
        y_pred_classweight = classweight_preds_df["y_pred_xgb_classweight_cal"].astype(float).to_numpy()
    else:
        y_pred_classweight = classweight_model.predict_proba(X_test)[:, 1]

    if len(smote_preds_df) == len(test_df) and "y_pred_xgb_cal" in smote_preds_df.columns:
        y_pred_smote = smote_preds_df["y_pred_xgb_cal"].astype(float).to_numpy()
    else:
        raise ValueError("SMOTE test predictions file is incompatible with the 2024 test split")

    print_header("Step 1 - Feature importance")
    avg_importances = extract_calibrated_xgb_importances(classweight_model)
    importance_df = pd.DataFrame(
        {"feature": MODEL_FEATURES, "avg_importance": avg_importances}
    ).sort_values("avg_importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    print("Top 10 features by average importance (class-weight calibrated):")
    print(
        importance_df.head(10).to_string(
            index=False,
            formatters={"avg_importance": lambda x: f"{x:.6f}"},
        )
    )

    year_row = importance_df[importance_df["feature"] == "year_encoded"]
    pace_row = importance_df[importance_df["feature"] == "pace_differential"]
    if year_row.empty or pace_row.empty:
        raise ValueError("Could not find year_encoded or pace_differential in MODEL_FEATURES")

    year_rank = int(year_row["rank"].iloc[0])
    year_val = float(year_row["avg_importance"].iloc[0])
    pace_rank = int(pace_row["rank"].iloc[0])
    pace_val = float(pace_row["avg_importance"].iloc[0])

    print(
        "year_encoded comparison vs SMOTE reference (rank 8, value 0.034): "
        f"rank={year_rank}, value={year_val:.6f}"
    )
    print(
        "pace_differential comparison vs SMOTE reference (lowest, value 0.003): "
        f"rank={pace_rank}, value={pace_val:.6f}"
    )

    print_header("Step 2 - Archetype-stratified Brier/AUC/count")
    archetype_codes = get_archetype_code_series(test_df)
    archetype_names = {0: "Other", 1: "Power", 2: "Street", 3: "Technical"}
    rows = []
    for code in [0, 1, 2, 3]:
        mask = archetype_codes == code
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "code": code,
                    "archetype": archetype_names[code],
                    "rows": 0,
                    "base_rate": np.nan,
                    "brier": np.nan,
                    "auc_roc": np.nan,
                    "smote_brier_ref": {0: 0.100, 1: 0.113, 2: 0.115, 3: 0.054}[code],
                }
            )
            continue
        yt = y_test[mask]
        yp = y_pred_classweight[mask]
        brier = brier_score_loss(yt, yp)
        auc = roc_auc_score(yt, yp) if yt.nunique() > 1 else np.nan
        rows.append(
            {
                "code": code,
                "archetype": archetype_names[code],
                "rows": n,
                "base_rate": float(yt.mean()),
                "brier": float(brier),
                "auc_roc": float(auc) if not pd.isna(auc) else np.nan,
                "smote_brier_ref": {0: 0.100, 1: 0.113, 2: 0.115, 3: 0.054}[code],
            }
        )
    archetype_df = pd.DataFrame(rows)
    print(
        archetype_df.to_string(
            index=False,
            formatters={
                "base_rate": lambda x: "nan" if pd.isna(x) else f"{x:.4%}",
                "brier": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "auc_roc": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "smote_brier_ref": lambda x: f"{x:.3f}",
            },
        )
    )

    valid = archetype_df.dropna(subset=["base_rate", "brier"])
    inverse_pattern_holds = is_inverse_ordering(valid.set_index("code")["base_rate"], valid.set_index("code")["brier"])
    print(
        "Pattern check (Brier ordering inverse of base-rate ordering): "
        + ("holds" if inverse_pattern_holds else "does NOT hold")
    )

    print_header("Step 3 - Full reliability diagram (class-weight next3)")
    rel_cw = reliability_table(y_test, y_pred_classweight)
    print(
        rel_cw.to_string(
            index=False,
            formatters={
                "bin_centre": lambda x: f"{x:.2f}",
                "mean_predicted": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "mean_actual": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "gap_pred_minus_actual": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
            },
        )
    )

    flagged = rel_cw[(rel_cw["flag"] != "") & (rel_cw["count"] > 0)]
    if flagged.empty:
        print("No populated bins exceed |gap| > 0.15")
    else:
        print("Flagged bins with |gap| > 0.15:")
        print(flagged[["bin", "gap_pred_minus_actual", "flag"]].to_string(index=False, formatters={"gap_pred_minus_actual": lambda x: f"{x:.6f}"}))

    print_header("Step 4 - sc_vsc_next5 sensitivity check (class-weight)")
    sens_df = df[df["sc_vsc_next5"].notna()].copy()
    sens_train = sens_df[sens_df["Year"] <= 2023].copy()
    sens_test = sens_df[sens_df["Year"] == 2024].copy()

    X_train_5 = sens_train[MODEL_FEATURES].copy()
    y_train_5 = sens_train["sc_vsc_next5"].astype(int).copy()
    X_test_5 = sens_test[MODEL_FEATURES].copy()
    y_test_5 = sens_test["sc_vsc_next5"].astype(int).copy()

    neg_5 = int((y_train_5 == 0).sum())
    pos_5 = int((y_train_5 == 1).sum())
    if pos_5 == 0:
        raise ValueError("No positive class for sc_vsc_next5 in training split")
    scale_pos_weight_5 = neg_5 / pos_5
    print(f"scale_pos_weight for sc_vsc_next5: {scale_pos_weight_5:.6f}")

    xgb_next5_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight_5,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
    )
    xgb_next5_cal = CalibratedClassifierCV(xgb_next5_base, cv=5, method="sigmoid")
    xgb_next5_cal.fit(X_train_5, y_train_5)
    y_pred_next5 = xgb_next5_cal.predict_proba(X_test_5)[:, 1]

    metrics_next3_cw = metric_row("class-weight next3", y_test, y_pred_classweight)
    metrics_next5_cw = metric_row("class-weight next5", y_test_5, y_pred_next5)
    metrics_next5_smote_ref = {
        "brier_score": 0.143,
        "auc_roc": 0.647,
        "avg_precision": 0.222,
    }

    print("Class-weight next5 metrics:")
    print(f"  Brier Score: {metrics_next5_cw['brier_score']:.6f}")
    print(f"  AUC-ROC: {metrics_next5_cw['auc_roc']:.6f}")
    print(f"  Average Precision: {metrics_next5_cw['avg_precision']:.6f}")
    print("Comparison references:")
    print(
        "  class-weight next3 -> "
        f"Brier={metrics_next3_cw['brier_score']:.6f}, AUC={metrics_next3_cw['auc_roc']:.6f}, "
        f"AP={metrics_next3_cw['avg_precision']:.6f}"
    )
    print(
        "  SMOTE next5 ref -> "
        f"Brier={metrics_next5_smote_ref['brier_score']:.3f}, "
        f"AUC={metrics_next5_smote_ref['auc_roc']:.3f}, AP={metrics_next5_smote_ref['avg_precision']:.3f}"
    )

    next3_stronger = (
        metrics_next3_cw["brier_score"] < metrics_next5_cw["brier_score"]
        and metrics_next3_cw["auc_roc"] > metrics_next5_cw["auc_roc"]
        and metrics_next3_cw["avg_precision"] > metrics_next5_cw["avg_precision"]
    )
    print(
        "Target-strength conclusion under class-weighting: "
        + ("N=3 remains stronger primary target" if next3_stronger else "mixed; N=3 is not uniformly stronger")
    )

    print_header("Step 5 - Consolidated summary table")
    smote_next3_metrics = metric_row("SMOTE next3", y_test, y_pred_smote)
    summary = pd.DataFrame(
        [
            {
                "model": "SMOTE next3",
                "brier": smote_next3_metrics["brier_score"],
                "auc_roc": smote_next3_metrics["auc_roc"],
                "avg_precision": smote_next3_metrics["avg_precision"],
                "prob_min": smote_next3_metrics["prob_min"],
                "prob_max": smote_next3_metrics["prob_max"],
            },
            {
                "model": "class-weight next3",
                "brier": metrics_next3_cw["brier_score"],
                "auc_roc": metrics_next3_cw["auc_roc"],
                "avg_precision": metrics_next3_cw["avg_precision"],
                "prob_min": metrics_next3_cw["prob_min"],
                "prob_max": metrics_next3_cw["prob_max"],
            },
            {
                "model": "class-weight next5",
                "brier": metrics_next5_cw["brier_score"],
                "auc_roc": metrics_next5_cw["auc_roc"],
                "avg_precision": metrics_next5_cw["avg_precision"],
                "prob_min": metrics_next5_cw["prob_min"],
                "prob_max": metrics_next5_cw["prob_max"],
            },
        ]
    )
    print(
        summary.to_string(
            index=False,
            formatters={
                "brier": lambda x: f"{x:.6f}",
                "auc_roc": lambda x: f"{x:.6f}",
                "avg_precision": lambda x: f"{x:.6f}",
                "prob_min": lambda x: f"{x:.6f}",
                "prob_max": lambda x: f"{x:.6f}",
            },
        )
    )

    print_header("Optional save - next5 class-weight model")
    if next5_save_path.exists():
        print(f"Skip save (file already exists, no overwrite): {next5_save_path}")
    else:
        joblib.dump(xgb_next5_cal, next5_save_path)
        print(f"Saved: {next5_save_path}")


if __name__ == "__main__":
    main()