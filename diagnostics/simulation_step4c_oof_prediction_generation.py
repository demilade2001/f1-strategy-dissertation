from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "phase2_features_final.csv"
FEATURE_LIST_PATH = ROOT / "data" / "processed" / "model_feature_list.txt"
HELDOUT_PRED_PATH = ROOT / "data" / "models" / "classweight_test_predictions.csv"
OOF_OUTPUT_PATH = ROOT / "data" / "models" / "classweight_oof_predictions_2022_2023.csv"
MODEL_PATH_REFERENCE = ROOT / "data" / "models" / "xgb_classweight_calibrated.pkl"

HELDOUT_DOC_METRICS = {
    "brier_score": 0.097,
    "auc_roc": 0.718,
    "avg_precision": 0.352,
}

LOCKED_RACES: List[Tuple[int, int, str]] = [
    (2023, 10, "British Grand Prix"),
    (2023, 6, "Monaco Grand Prix"),
    (2022, 17, "Singapore Grand Prix"),
    (2022, 22, "Abu Dhabi Grand Prix"),
    (2022, 13, "Hungarian Grand Prix"),
    (2022, 18, "Japanese Grand Prix"),
    (2023, 7, "Spanish Grand Prix"),
]


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


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
    return grouped[["bin", "bin_centre", "mean_predicted", "mean_actual", "count", "gap_pred_minus_actual"]]


def prob_shape_stats(prob: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(prob)),
        "std": float(np.std(prob, ddof=1)) if len(prob) > 1 else 0.0,
        "p50": float(np.percentile(prob, 50)),
        "p75": float(np.percentile(prob, 75)),
        "p90": float(np.percentile(prob, 90)),
        "p95": float(np.percentile(prob, 95)),
    }


def decision_gate(metrics_table: pd.DataFrame, cal_compare: pd.DataFrame) -> Tuple[bool, str]:
    auc_gap = float(metrics_table.loc[metrics_table["metric"] == "auc_roc", "gap_oof_minus_heldout"].iloc[0])
    brier_gap = float(metrics_table.loc[metrics_table["metric"] == "brier_score", "gap_oof_minus_heldout"].iloc[0])
    ap_gap = float(metrics_table.loc[metrics_table["metric"] == "avg_precision", "gap_oof_minus_heldout"].iloc[0])

    flagged_bins = cal_compare[cal_compare["flag_meaningfully_worse_oof"].eq(True)]
    fail = (
        (auc_gap > 0.10)
        or (brier_gap < -0.02)
        or (ap_gap > 0.08)
        or (len(flagged_bins) >= 2)
    )
    if fail:
        return False, "FAIL: OOF predictions still flagged as unusable by gate thresholds."
    return True, "PASS: OOF predictions are acceptable under the configured gate thresholds."


def main() -> None:
    XGBClassifier = load_xgb_classifier()

    print_header("Step 0 - Exact original training hyperparameters")
    print("Training reference: src/phase2_model_classweight_check.py")
    print(f"Reference artifact trained there: {MODEL_PATH_REFERENCE}")
    print("XGBClassifier hyperparameters (replicated identically):")
    print("- scale_pos_weight = neg_count / pos_count (computed from training data per fit)")
    print("- use_label_encoder = False")
    print("- eval_metric = 'logloss'")
    print("- random_state = 42")
    print("- n_estimators = 300")
    print("- max_depth = 6")
    print("- learning_rate = 0.05")
    print("CalibratedClassifierCV configuration (replicated identically):")
    print("- cv = 5")
    print("- method = 'sigmoid'")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing feature matrix: {DATA_PATH}")
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(f"Missing feature list: {FEATURE_LIST_PATH}")
    if not HELDOUT_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing held-out predictions: {HELDOUT_PRED_PATH}")

    print_header("Step 1 - Load training data and apply preprocessing")
    df = pd.read_csv(DATA_PATH)
    df["is_strategic_stop"] = df["is_strategic_stop"].astype(pd.BooleanDtype())
    df["sc_active"] = df["sc_active"].astype(bool)
    df["vsc_active"] = df["vsc_active"].astype(bool)
    df["caution_active"] = df["caution_active"].astype(bool)

    model_features = [
        line.strip()
        for line in FEATURE_LIST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    train_df = df[df["Year"] <= 2023].copy().reset_index(drop=True)
    print(f"rows_year_le_2023={len(train_df)}")

    missing_features = [f for f in model_features if f not in train_df.columns]
    if missing_features:
        raise KeyError(f"MODEL_FEATURES missing from dataset: {missing_features}")

    X = train_df[model_features].copy()
    y = train_df["sc_vsc_next3"].astype(int).copy()
    x_nulls = int(X.isna().sum().sum())
    y_nulls = int(y.isna().sum())
    print(f"model_features_count={len(model_features)}")
    print(f"X_shape={X.shape}")
    print(f"X_null_cells={x_nulls}")
    print(f"y_null_count={y_nulls}")
    if x_nulls != 0 or y_nulls != 0:
        raise ValueError("Nulls found in Year<=2023 training subset")

    print_header("Step 2 - GroupKFold setup by (Year, Round) and split verification")
    groups = train_df["Year"].astype(int).astype(str) + "_" + train_df["Round"].astype(int).astype(str)
    gkf = GroupKFold(n_splits=5)

    fold_assignments = np.full(len(train_df), -1, dtype=int)
    group_to_fold: Dict[str, int] = {}
    for fold_idx, (_, valid_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        fold_assignments[valid_idx] = fold_idx
        valid_groups = groups.iloc[valid_idx].unique().tolist()
        for g in valid_groups:
            if g in group_to_fold and group_to_fold[g] != fold_idx:
                raise RuntimeError(f"Group {g} appears in multiple folds: {group_to_fold[g]} and {fold_idx}")
            group_to_fold[g] = fold_idx

    if np.any(fold_assignments == -1):
        raise RuntimeError("At least one row was not assigned to a fold")

    sample_races = ["2022_13", "2022_17", "2023_10"]
    print("Sample race fold-purity verification:")
    for race_key in sample_races:
        race_mask = groups.eq(race_key)
        if int(race_mask.sum()) == 0:
            print(f"- race {race_key}: missing in Year<=2023 subset")
            continue
        race_folds = sorted(set(fold_assignments[race_mask.to_numpy()].tolist()))
        print(f"- race {race_key}: rows={int(race_mask.sum())}, assigned_folds={race_folds}")
        if len(race_folds) != 1:
            raise RuntimeError(f"Race {race_key} is split across folds; aborting as requested")

    # Global hard check for all races.
    group_fold_counts = pd.DataFrame({"group": groups, "fold": fold_assignments}).groupby("group")["fold"].nunique()
    split_groups = group_fold_counts[group_fold_counts > 1]
    if not split_groups.empty:
        raise RuntimeError(f"Found races split across folds: {split_groups.index.tolist()[:10]}")
    print("Hard check passed: every (Year, Round) race group maps to exactly one fold.")

    print_header("Step 3 - Train and score each fold (fresh model per fold)")
    oof_pred = np.full(len(train_df), np.nan, dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train_fold = X.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        X_valid_fold = X.iloc[valid_idx].copy()

        neg_count = int((y_train_fold == 0).sum())
        pos_count = int((y_train_fold == 1).sum())
        if pos_count == 0:
            raise ValueError(f"Fold {fold_idx} has zero positives in training partition")
        scale_pos_weight = neg_count / pos_count

        xgb_base = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
        )
        xgb_cal = CalibratedClassifierCV(xgb_base, cv=5, method="sigmoid")
        xgb_cal.fit(X_train_fold, y_train_fold)
        fold_pred = xgb_cal.predict_proba(X_valid_fold)[:, 1]

        oof_pred[valid_idx] = fold_pred

        n_train_groups = int(groups.iloc[train_idx].nunique())
        n_valid_groups = int(groups.iloc[valid_idx].nunique())
        print(
            f"fold={fold_idx} train_rows={len(train_idx)} valid_rows={len(valid_idx)} "
            f"train_races={n_train_groups} valid_races={n_valid_groups} scale_pos_weight={scale_pos_weight:.6f}"
        )

    if np.isnan(oof_pred).any():
        raise RuntimeError("OOF prediction vector contains NaN entries after fold loop")

    print_header("Step 4 - Combine OOF predictions and coverage checks")
    oof_df = train_df[["Year", "Round", "Driver", "LapNumber"]].copy()
    oof_df["y_true"] = y.to_numpy()
    oof_df["y_pred_xgb_classweight_cal"] = oof_pred

    key_cols = ["Year", "Round", "Driver", "LapNumber"]
    duplicate_count = int(oof_df.duplicated(subset=key_cols, keep=False).sum())
    print(f"oof_total_rows={len(oof_df)}")
    print(f"expected_rows_step1={len(train_df)}")
    print(f"row_count_match={len(oof_df) == len(train_df)}")
    print(f"duplicate_key_rows={duplicate_count}")
    if duplicate_count != 0:
        raise RuntimeError("Duplicate (Year,Round,Driver,LapNumber) rows found in combined OOF predictions")

    print_header("Step 5 - OOF vs held-out metrics comparison")
    heldout_df = pd.read_csv(HELDOUT_PRED_PATH)
    heldout_y = heldout_df["y_true"].astype(int)
    heldout_p = heldout_df["y_pred_xgb_classweight_cal"].astype(float).to_numpy()

    oof_metrics = {
        "brier_score": float(brier_score_loss(oof_df["y_true"].astype(int), oof_df["y_pred_xgb_classweight_cal"].to_numpy())),
        "auc_roc": float(roc_auc_score(oof_df["y_true"].astype(int), oof_df["y_pred_xgb_classweight_cal"].to_numpy())),
        "avg_precision": float(average_precision_score(oof_df["y_true"].astype(int), oof_df["y_pred_xgb_classweight_cal"].to_numpy())),
    }
    heldout_metrics_actual = {
        "brier_score": float(brier_score_loss(heldout_y, heldout_p)),
        "auc_roc": float(roc_auc_score(heldout_y, heldout_p)),
        "avg_precision": float(average_precision_score(heldout_y, heldout_p)),
    }

    metrics_table = pd.DataFrame(
        [
            {
                "metric": metric,
                "oof_2022_2023": oof_metrics[metric],
                "heldout_2024_documented": HELDOUT_DOC_METRICS[metric],
                "heldout_2024_actual_file": heldout_metrics_actual[metric],
                "gap_oof_minus_heldout": oof_metrics[metric] - heldout_metrics_actual[metric],
            }
            for metric in ["brier_score", "auc_roc", "avg_precision"]
        ]
    )
    print(
        metrics_table.to_string(
            index=False,
            formatters={
                "oof_2022_2023": lambda x: f"{x:.6f}",
                "heldout_2024_documented": lambda x: f"{x:.3f}",
                "heldout_2024_actual_file": lambda x: f"{x:.6f}",
                "gap_oof_minus_heldout": lambda x: f"{x:+.6f}",
            },
        )
    )

    print_header("Step 6 - Calibration bin comparison (OOF vs held-out 2024)")
    rel_oof = reliability_table(oof_df["y_true"].astype(int), oof_df["y_pred_xgb_classweight_cal"].to_numpy())
    rel_hold = reliability_table(heldout_y, heldout_p)

    cal_compare = rel_oof.merge(rel_hold, on=["bin", "bin_centre"], suffixes=("_oof", "_heldout"))
    cal_compare["abs_gap_oof"] = cal_compare["gap_pred_minus_actual_oof"].abs()
    cal_compare["abs_gap_heldout"] = cal_compare["gap_pred_minus_actual_heldout"].abs()
    cal_compare["flag_meaningfully_worse_oof"] = (
        (cal_compare["count_oof"] > 0)
        & (cal_compare["count_heldout"] > 0)
        & ((cal_compare["abs_gap_oof"] - cal_compare["abs_gap_heldout"]) > 0.05)
    )

    print(
        cal_compare[[
            "bin",
            "count_oof",
            "mean_predicted_oof",
            "mean_actual_oof",
            "gap_pred_minus_actual_oof",
            "count_heldout",
            "mean_predicted_heldout",
            "mean_actual_heldout",
            "gap_pred_minus_actual_heldout",
            "flag_meaningfully_worse_oof",
        ]].to_string(
            index=False,
            formatters={
                "mean_predicted_oof": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "mean_actual_oof": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "gap_pred_minus_actual_oof": lambda x: "nan" if pd.isna(x) else f"{x:+.6f}",
                "mean_predicted_heldout": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "mean_actual_heldout": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "gap_pred_minus_actual_heldout": lambda x: "nan" if pd.isna(x) else f"{x:+.6f}",
            },
        )
    )

    flagged = cal_compare[cal_compare["flag_meaningfully_worse_oof"].eq(True)]
    if flagged.empty:
        print("No bins flagged as meaningfully worse than held-out.")
    else:
        print("Flagged bins (OOF abs-gap exceeds held-out abs-gap by >0.05):")
        print(flagged[["bin", "gap_pred_minus_actual_oof", "gap_pred_minus_actual_heldout"]].to_string(index=False))

    print_header("Step 7 - Distribution shape comparison")
    oof_shape = prob_shape_stats(oof_df["y_pred_xgb_classweight_cal"].to_numpy())
    hold_shape = prob_shape_stats(heldout_p)

    shape_table = pd.DataFrame(
        [
            {
                "stat": stat,
                "oof_2022_2023": oof_shape[stat],
                "heldout_2024": hold_shape[stat],
                "gap_oof_minus_heldout": oof_shape[stat] - hold_shape[stat],
            }
            for stat in ["mean", "std", "p50", "p75", "p90", "p95"]
        ]
    )
    print(
        shape_table.to_string(
            index=False,
            formatters={
                "oof_2022_2023": lambda x: f"{x:.6f}",
                "heldout_2024": lambda x: f"{x:.6f}",
                "gap_oof_minus_heldout": lambda x: f"{x:+.6f}",
            },
        )
    )

    print_header("Step 8 - Decision gate")
    print("Gate thresholds:")
    print("- fail if AUC gap > +0.10")
    print("- fail if Brier gap < -0.02")
    print("- fail if AP gap > +0.08")
    print("- fail if >=2 bins flagged meaningfully worse (abs-gap delta > 0.05)")

    passed, gate_message = decision_gate(metrics_table, cal_compare)
    print(gate_message)

    if not passed:
        print("Step 9 skipped: gate failed, so no 7-race CSV is written.")
        return

    print_header("Step 9 - Filter 7 races and save OOF predictions")
    locked_pairs = {(year, rnd) for year, rnd, _ in LOCKED_RACES}
    final_df = oof_df[oof_df[["Year", "Round"]].apply(lambda r: (int(r["Year"]), int(r["Round"])) in locked_pairs, axis=1)].copy()
    final_df["is_out_of_fold_prediction"] = True
    final_df = final_df[[
        "Year",
        "Round",
        "Driver",
        "LapNumber",
        "y_true",
        "y_pred_xgb_classweight_cal",
        "is_out_of_fold_prediction",
    ]].sort_values(["Year", "Round", "Driver", "LapNumber"]).reset_index(drop=True)

    final_df.to_csv(OOF_OUTPUT_PATH, index=False)
    print(f"saved_oof_locked_races_csv={OOF_OUTPUT_PATH}")
    print(f"saved_row_count={len(final_df)}")


if __name__ == "__main__":
    main()
