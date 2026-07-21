from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "phase2_features_final.csv"
FEATURE_LIST_PATH = ROOT / "data" / "processed" / "model_feature_list.txt"
MODEL_PATH = ROOT / "data" / "models" / "xgb_classweight_calibrated.pkl"
HELDOUT_PRED_PATH = ROOT / "data" / "models" / "classweight_test_predictions.csv"
OUTPUT_PRED_PATH = ROOT / "data" / "models" / "classweight_in_sample_predictions_2022_2023.csv"

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
    """Use the same 10 fixed [0.0,1.0] bins as the original Phase 2 reliability tables."""
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


def decision_gate(metrics_table: pd.DataFrame, cal_compare: pd.DataFrame) -> str:
    """Simple, explicit gate for overconfidence risk."""
    auc_gap = float(metrics_table.loc[metrics_table["metric"] == "auc_roc", "gap_in_sample_minus_heldout"].iloc[0])
    brier_gap = float(metrics_table.loc[metrics_table["metric"] == "brier_score", "gap_in_sample_minus_heldout"].iloc[0])
    ap_gap = float(metrics_table.loc[metrics_table["metric"] == "avg_precision", "gap_in_sample_minus_heldout"].iloc[0])

    flagged_bins = cal_compare[cal_compare["flag_meaningfully_worse_in_sample"].eq(True)]

    high_risk = (
        (auc_gap > 0.10)
        or (brier_gap < -0.02)
        or (ap_gap > 0.08)
        or (len(flagged_bins) >= 2)
    )

    if high_risk:
        return (
            "Decision: in-sample predictions show clear signs of overconfidence versus held-out behavior; "
            "do not use without guardrails/recalibration."
        )
    return (
        "Decision: in-sample predictions look reasonably close to held-out behavior for this use-case; "
        "proceeding may be acceptable with explicit caveat that these are in-sample estimates."
    )


def main() -> None:
    # Keep import behavior aligned with historical scripts (validates xgboost runtime early on macOS).
    _ = load_xgb_classifier()

    print_header("Step 1 - Exact preprocessing path used for classweight_test_predictions.csv")
    print("Source script located: src/phase2_model_classweight_check.py")
    print("Exact sequence replicated for scoring rows:")
    print("1) Load data/processed/phase2_features_final.csv")
    print("2) Cast: is_strategic_stop -> BooleanDtype, sc_active -> bool, vsc_active -> bool, caution_active -> bool")
    print("3) Load MODEL_FEATURES from data/processed/model_feature_list.txt")
    print("4) Confirm all MODEL_FEATURES exist; create X = df[MODEL_FEATURES]")
    print("5) Confirm zero nulls in X and target")
    print("6) Load data/models/xgb_classweight_calibrated.pkl via joblib")
    print("7) Score with calibrated_model.predict_proba(X_subset)[:, 1] -> y_pred_xgb_classweight_cal")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing feature matrix: {DATA_PATH}")
    if not FEATURE_LIST_PATH.exists():
        raise FileNotFoundError(f"Missing model feature list: {FEATURE_LIST_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing calibrated model: {MODEL_PATH}")
    if not HELDOUT_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing held-out predictions: {HELDOUT_PRED_PATH}")

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

    print_header("Step 2 - Locate 7 races in the training feature matrix")
    missing_features = [f for f in model_features if f not in df.columns]
    if missing_features:
        raise KeyError(f"MODEL_FEATURES missing from dataset: {missing_features}")

    locked_pairs = {(year, rnd) for year, rnd, _ in LOCKED_RACES}
    in_scope = df[df[["Year", "Round"]].apply(lambda r: (int(r["Year"]), int(r["Round"])) in locked_pairs, axis=1)].copy()

    if in_scope.empty:
        raise ValueError("No rows found for the 7 requested locked races")

    rows = []
    for year, rnd, name in LOCKED_RACES:
        subset = in_scope[(in_scope["Year"] == year) & (in_scope["Round"] == rnd)]
        rows.append({
            "Year": year,
            "Round": rnd,
            "EventName_expected": name,
            "rows": int(len(subset)),
            "EventName_found": subset["EventName"].dropna().iloc[0] if not subset.empty else "MISSING",
        })
    race_count_df = pd.DataFrame(rows)
    print(race_count_df.to_string(index=False))

    if (race_count_df["rows"] == 0).any():
        raise ValueError("At least one requested (Year, Round) race has zero rows in phase2_features_final.csv")

    x_locked = in_scope[model_features].copy()
    y_locked = in_scope["sc_vsc_next3"].astype(int).copy()

    x_locked_nulls = int(x_locked.isna().sum().sum())
    y_locked_nulls = int(y_locked.isna().sum())
    print(f"MODEL_FEATURES count={len(model_features)}")
    print(f"X_locked shape={x_locked.shape}")
    print(f"X_locked null cells={x_locked_nulls}")
    print(f"y_locked null count={y_locked_nulls}")
    if x_locked_nulls != 0 or y_locked_nulls != 0:
        raise ValueError("Nulls found in in-sample rows; preprocessing parity is broken")

    print_header("Step 3 - Generate in-sample predictions for the 7 races")
    calibrated_model = joblib.load(MODEL_PATH)
    y_pred_locked = calibrated_model.predict_proba(x_locked)[:, 1]

    out_df = in_scope[["Year", "Round", "Driver", "LapNumber"]].copy()
    out_df["y_true"] = y_locked.to_numpy()
    out_df["y_pred_xgb_classweight_cal"] = y_pred_locked
    out_df["is_in_sample"] = True

    out_df = out_df.sort_values(["Year", "Round", "Driver", "LapNumber"]).reset_index(drop=True)
    out_df.to_csv(OUTPUT_PRED_PATH, index=False)

    print(f"Saved in-sample predictions: {OUTPUT_PRED_PATH}")
    print(f"row_count={len(out_df)}")
    print("sample_5_rows:")
    print(out_df.head(5).to_string(index=False))

    print_header("Step 4 - In-sample vs held-out metrics")
    heldout_df = pd.read_csv(HELDOUT_PRED_PATH)
    heldout_y = heldout_df["y_true"].astype(int)
    heldout_p = heldout_df["y_pred_xgb_classweight_cal"].astype(float).to_numpy()

    in_metrics = {
        "brier_score": float(brier_score_loss(out_df["y_true"].astype(int), out_df["y_pred_xgb_classweight_cal"].to_numpy())),
        "auc_roc": float(roc_auc_score(out_df["y_true"].astype(int), out_df["y_pred_xgb_classweight_cal"].to_numpy())),
        "avg_precision": float(average_precision_score(out_df["y_true"].astype(int), out_df["y_pred_xgb_classweight_cal"].to_numpy())),
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
                "in_sample_7_races": in_metrics[metric],
                "heldout_2024_documented": HELDOUT_DOC_METRICS[metric],
                "heldout_2024_actual_file": heldout_metrics_actual[metric],
                "gap_in_sample_minus_heldout": in_metrics[metric] - heldout_metrics_actual[metric],
            }
            for metric in ["brier_score", "auc_roc", "avg_precision"]
        ]
    )
    print(
        metrics_table.to_string(
            index=False,
            formatters={
                "in_sample_7_races": lambda x: f"{x:.6f}",
                "heldout_2024_documented": lambda x: f"{x:.3f}",
                "heldout_2024_actual_file": lambda x: f"{x:.6f}",
                "gap_in_sample_minus_heldout": lambda x: f"{x:+.6f}",
            },
        )
    )

    print_header("Step 5 - Calibration bin comparison (same fixed 10 bins as Chapter 4 pipeline)")
    rel_in = reliability_table(out_df["y_true"].astype(int), out_df["y_pred_xgb_classweight_cal"].to_numpy())
    rel_hold = reliability_table(heldout_y, heldout_p)

    cal_compare = rel_in.merge(rel_hold, on=["bin", "bin_centre"], suffixes=("_in_sample", "_heldout"))
    cal_compare["abs_gap_in_sample"] = cal_compare["gap_pred_minus_actual_in_sample"].abs()
    cal_compare["abs_gap_heldout"] = cal_compare["gap_pred_minus_actual_heldout"].abs()

    # "Meaningfully larger" threshold: abs-gap exceeds held-out by > 0.05 in same bin.
    cal_compare["flag_meaningfully_worse_in_sample"] = (
        (cal_compare["count_in_sample"] > 0)
        & (cal_compare["count_heldout"] > 0)
        & ((cal_compare["abs_gap_in_sample"] - cal_compare["abs_gap_heldout"]) > 0.05)
    )

    print(
        cal_compare[[
            "bin",
            "count_in_sample",
            "mean_predicted_in_sample",
            "mean_actual_in_sample",
            "gap_pred_minus_actual_in_sample",
            "count_heldout",
            "mean_predicted_heldout",
            "mean_actual_heldout",
            "gap_pred_minus_actual_heldout",
            "flag_meaningfully_worse_in_sample",
        ]].to_string(
            index=False,
            formatters={
                "mean_predicted_in_sample": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "mean_actual_in_sample": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "gap_pred_minus_actual_in_sample": lambda x: "nan" if pd.isna(x) else f"{x:+.6f}",
                "mean_predicted_heldout": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "mean_actual_heldout": lambda x: "nan" if pd.isna(x) else f"{x:.6f}",
                "gap_pred_minus_actual_heldout": lambda x: "nan" if pd.isna(x) else f"{x:+.6f}",
            },
        )
    )

    flagged = cal_compare[cal_compare["flag_meaningfully_worse_in_sample"].eq(True)]
    if flagged.empty:
        print("No calibration bins were flagged as meaningfully worse than held-out in the same range.")
    else:
        print("Flagged bins (in-sample abs-gap > held-out abs-gap by >0.05):")
        print(flagged[["bin", "gap_pred_minus_actual_in_sample", "gap_pred_minus_actual_heldout"]].to_string(index=False))

    print_header("Step 6 - Probability distribution shape comparison")
    in_shape = prob_shape_stats(out_df["y_pred_xgb_classweight_cal"].to_numpy())
    hold_shape = prob_shape_stats(heldout_p)

    shape_table = pd.DataFrame(
        [
            {
                "stat": stat,
                "in_sample_7_races": in_shape[stat],
                "heldout_2024": hold_shape[stat],
                "gap_in_sample_minus_heldout": in_shape[stat] - hold_shape[stat],
            }
            for stat in ["mean", "std", "p50", "p75", "p90", "p95"]
        ]
    )
    print(
        shape_table.to_string(
            index=False,
            formatters={
                "in_sample_7_races": lambda x: f"{x:.6f}",
                "heldout_2024": lambda x: f"{x:.6f}",
                "gap_in_sample_minus_heldout": lambda x: f"{x:+.6f}",
            },
        )
    )

    print_header("Step 7 - Decision gate (report only)")
    print("Gate thresholds:")
    print("- Potential overconfidence if AUC gap > +0.10 or Brier gap < -0.02 or AP gap > +0.08")
    print("- Potential overconfidence if >=2 bins are meaningfully worse (abs-gap delta > 0.05)")
    print(decision_gate(metrics_table, cal_compare))
    print("No race_state wiring is performed in this step.")


if __name__ == "__main__":
    main()
