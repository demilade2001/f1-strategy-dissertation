from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source_code.utils import MIDFIELD_CONSTRUCTORS  # noqa: E402


BASE_DF_PATH = REPO_ROOT / "data" / "processed" / "base_df.csv"
FEATURES_PATH = REPO_ROOT / "src" / "features.py"
UTILS_PATH = REPO_ROOT / "src" / "utils.py"
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
DIAGNOSTIC_OUTPUT_DIR = REPO_ROOT / "data" / "diagnostics"

ARCHETYPE_SAMPLED_RACES = {
    "Monaco Grand Prix",
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "Miami Grand Prix",
    "Italian Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Bahrain Grand Prix",
    "Spanish Grand Prix",
    "Hungarian Grand Prix",
    "Abu Dhabi Grand Prix",
    "Japanese Grand Prix",
}


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def read_lines(path: Path, start_line: int, end_line: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = lines[start_line - 1:end_line]
    return "\n".join(chunk)


def print_block(title: str, block: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(block)


def format_bool_counts(series: pd.Series) -> str:
    counts = series.value_counts(dropna=False)
    true_count = int(counts.get(True, 0))
    false_count = int(counts.get(False, 0))
    na_count = int(counts.get(pd.NA, 0))
    return f"True={true_count}, False={false_count}, NA={na_count}"


def logical_midfield_team_count(labels: Iterable[str]) -> int:
    normalized = {"RB" if label == "AlphaTauri" else label for label in labels}
    return len(normalized)


def load_base_df() -> pd.DataFrame:
    if not BASE_DF_PATH.exists():
        raise FileNotFoundError(f"base_df.csv not found at {BASE_DF_PATH}")

    df = pd.read_csv(BASE_DF_PATH)
    df["is_strategic_stop"] = df["is_strategic_stop"].astype(pd.BooleanDtype())

    for col in ["sc_active", "vsc_active"]:
        if col not in df.columns:
            raise KeyError(f"Mandatory cast column missing from base_df.csv: {col}")
        df[col] = df[col].infer_objects(copy=False).fillna(False).astype(bool)

    if "caution_active" in df.columns:
        df["caution_active"] = (
            df["caution_active"].infer_objects(copy=False).fillna(False).astype(bool)
        )
        print("caution_active present in serialized base_df.csv: yes")
    else:
        print("caution_active present in serialized base_df.csv: no")
        print(
            "Diagnostic-only reconstruction: caution_active = sc_active | vsc_active "
            "to satisfy the standing cast policy without modifying base_df.csv."
        )
        df["caution_active"] = (df["sc_active"] | df["vsc_active"]).astype(bool)

    return df


def load_native_pits() -> pd.DataFrame:
    all_pits = []
    for year in [2022, 2023, 2024]:
        pits_path = RAW_DATA_DIR / str(year) / f"f1_{year}_pits.csv"
        pits = pd.read_csv(pits_path)
        if "Year" not in pits.columns:
            pits["Year"] = year
        all_pits.append(pits)
    return pd.concat(all_pits, ignore_index=True)


def reapply_corrected_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pit_merge_keys = ["Year", "Round", "Driver", "LapNumber"]
    pit_native = load_native_pits()[pit_merge_keys + ["TyreCompoundOld", "TyreCompoundNew"]].copy()
    lap_context = df[pit_merge_keys + ["sc_active", "vsc_active"]].copy()

    pit_eval = pit_native.merge(
        lap_context,
        on=pit_merge_keys,
        how="left",
        indicator=True,
    )
    pit_eval["same_compound"] = (
        pit_eval["TyreCompoundOld"].notna()
        & pit_eval["TyreCompoundNew"].notna()
        & (pit_eval["TyreCompoundOld"] == pit_eval["TyreCompoundNew"])
    )
    with pd.option_context("future.no_silent_downcasting", True):
        sc_active_bool = pit_eval["sc_active"].infer_objects(copy=False).fillna(False).astype(bool)
        vsc_active_bool = pit_eval["vsc_active"].infer_objects(copy=False).fillna(False).astype(bool)
    pit_eval["under_caution"] = sc_active_bool | vsc_active_bool
    pit_eval["is_strategic_stop_corrected"] = pd.Series(pd.NA, index=pit_eval.index, dtype="boolean")
    same_compound_mask = pit_eval["same_compound"]
    pit_eval.loc[same_compound_mask, "is_strategic_stop_corrected"] = (
        pit_eval.loc[same_compound_mask, "under_caution"].astype("boolean")
    )

    strategic_assign = (
        pit_eval[pit_merge_keys + ["same_compound", "under_caution", "is_strategic_stop_corrected", "_merge"]]
        .drop_duplicates(pit_merge_keys)
        .copy()
    )
    relabeled = df.merge(strategic_assign, on=pit_merge_keys, how="left")
    return relabeled, pit_eval


def within_10_percent(actual: int, target: int) -> bool:
    if target == 0:
        return actual == 0
    return abs(actual - target) <= 0.10 * target


def print_corrected_label_verification(df: pd.DataFrame, relabeled: pd.DataFrame, pit_eval: pd.DataFrame) -> None:
    print_header("Task 1 - Corrected is_strategic_stop verification")
    print(f"base_df path: {BASE_DF_PATH}")
    print(f"features source: {FEATURES_PATH}")
    print(
        "corrected rule applied to base_df.csv via native raw pits: "
        "same_compound & under_caution, with non-same-compound rows forced to <NA>"
    )

    original_non_null = int(df["is_strategic_stop"].notna().sum())
    original_true = int((df["is_strategic_stop"] == True).sum())
    original_false = int((df["is_strategic_stop"] == False).sum())

    corrected_mask = relabeled["is_strategic_stop_corrected"].notna()
    corrected_true = int((relabeled["is_strategic_stop_corrected"] == True).sum())
    corrected_false = int((relabeled["is_strategic_stop_corrected"] == False).sum())
    corrected_non_null = int(corrected_mask.sum())

    same_compound_true = int(pit_eval["same_compound"].sum())
    same_compound_false = int((pit_eval["same_compound"] == False).sum())
    same_compound_caution_true = int((pit_eval["same_compound"] & pit_eval["under_caution"]).sum())
    same_compound_caution_false = int((pit_eval["same_compound"] & (~pit_eval["under_caution"])).sum())

    print(
        f"serialized counts before fix: True={original_true}, False={original_false}, non_null={original_non_null}"
    )
    print(
        f"corrected counts after re-labelling: True={corrected_true}, False={corrected_false}, non_null={corrected_non_null}"
    )
    print(
        f"native pit-event audit: same_compound=True rows={same_compound_true}, "
        f"same_compound=False rows={same_compound_false}"
    )
    print(
        f"same_compound rows split by caution state: under_caution=True -> {same_compound_caution_true}, "
        f"under_caution=False -> {same_compound_caution_false}"
    )
    print(
        f"merge audit for pit relabelling: matched={int((pit_eval['_merge'] == 'both').sum())}, "
        f"unmatched={int((pit_eval['_merge'] != 'both').sum())}"
    )

    total_close = within_10_percent(corrected_non_null, 583)
    true_close = within_10_percent(corrected_true, 215)
    false_close = within_10_percent(corrected_false, 368)
    print(
        f"comparison to 583 total: observed={corrected_non_null}, delta={corrected_non_null - 583}, "
        f"close_within_10pct={total_close}"
    )
    print(
        f"comparison to 215/368 split: observed={corrected_true}/{corrected_false}, "
        f"delta_true={corrected_true - 215}, delta_false={corrected_false - 368}, "
        f"close_within_10pct={true_close and false_close}"
    )


def normalize_track_status(raw_status: object) -> str:
    if pd.isna(raw_status):
        return ""
    try:
        return str(int(raw_status))
    except (TypeError, ValueError):
        return str(raw_status)


def classify_extreme_row(row: pd.Series) -> str:
    raw_status = normalize_track_status(row.get("TrackStatus"))
    red_flag = "5" in set(raw_status)
    formation_lap = row.get("LapNumber") == 1
    pit_in = pd.notna(row.get("PitInTime"))
    pit_out = pd.notna(row.get("PitOutTime"))
    is_accurate = bool(row.get("IsAccurate")) if pd.notna(row.get("IsAccurate")) else False

    reasons = []
    if red_flag:
        reasons.append("red-flag code present")
    if formation_lap:
        reasons.append("lap 1 / formation-lap-like row")
    if pit_in:
        reasons.append("pit-in lap")
    if pit_out:
        reasons.append("pit-out lap")
    if not reasons:
        reasons.append("timing artefact likely")
    elif not is_accurate and "timing artefact likely" not in reasons:
        reasons.append("timing artefact likely because IsAccurate=False")

    return "; ".join(reasons)


def monaco_hard_subset(df: pd.DataFrame) -> pd.DataFrame:
    subset = df.loc[
        (df["EventName"] == "Monaco Grand Prix") & (df["TyreCompound"] == "HARD")
    ].copy()
    return subset


def degradation_eligible_subset(df: pd.DataFrame) -> pd.DataFrame:
    eligible = df.copy()
    mask = pd.Series(True, index=eligible.index)
    mask &= eligible["compound_known"] == True
    mask &= eligible["IsAccurate"] == True
    mask &= eligible["sc_active"] == False
    mask &= eligible["vsc_active"] == False
    mask &= eligible["PitInTime"].isna()
    mask &= eligible["PitOutTime"].isna()
    mask &= eligible["TyreLife"].notna()
    mask &= eligible["LapTime_s"].notna()
    wet_compounds = {"INTERMEDIATE", "WET"}
    mask &= ~eligible["TyreCompound"].isin(wet_compounds)
    eligible = eligible.loc[mask].copy()
    return eligible


def fit_degradation_slope(group: pd.DataFrame) -> dict[str, object]:
    if len(group) < 5:
        return {
            "n": len(group),
            "model": "insufficient",
            "slope": np.nan,
            "r2_linear": np.nan,
            "r2_quadratic": np.nan,
        }

    x = group["TyreLife"].to_numpy(dtype=float)
    y = group["LapTime_s"].to_numpy(dtype=float)
    if np.isnan(x).any() or np.isnan(y).any():
        return {
            "n": len(group),
            "model": "invalid_data",
            "slope": np.nan,
            "r2_linear": np.nan,
            "r2_quadratic": np.nan,
        }

    coeffs_linear = np.polyfit(x, y, 1)
    y_pred_linear = np.polyval(coeffs_linear, x)
    ss_res_linear = float(np.sum((y - y_pred_linear) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0.0

    coeffs_quadratic = np.polyfit(x, y, 2)
    y_pred_quadratic = np.polyval(coeffs_quadratic, x)
    ss_res_quadratic = float(np.sum((y - y_pred_quadratic) ** 2))
    r2_quadratic = 1 - (ss_res_quadratic / ss_tot) if ss_tot > 0 else 0.0

    if r2_quadratic > r2_linear + 0.02:
        a = float(coeffs_quadratic[0])
        b = float(coeffs_quadratic[1])
        mean_tyre_life = float(x.mean())
        slope = (2 * a * mean_tyre_life) + b
        model = "quadratic"
    else:
        slope = float(coeffs_linear[0])
        model = "linear"

    return {
        "n": len(group),
        "model": model,
        "slope": slope,
        "r2_linear": r2_linear,
        "r2_quadratic": r2_quadratic,
    }


def print_monaco_outlier_diagnostic(df: pd.DataFrame) -> None:
    print_header("Task 2 - Monaco HARD outlier re-run")
    subset = monaco_hard_subset(df)
    print(f"Monaco HARD rows across all seasons: {len(subset)}")

    top10 = subset.sort_values("LapTime_s", ascending=False).head(10).copy()
    top10_display = top10[
        [
            "Year",
            "LapNumber",
            "Driver",
            "TyreLife",
            "LapTime_s",
            "TrackStatus",
            "PitInTime",
            "PitOutTime",
            "sc_active",
            "vsc_active",
        ]
    ]
    print("Top 10 Monaco HARD rows by LapTime_s descending:")
    print(top10_display.to_string(index=False))

    extreme_rows = subset.loc[subset["LapTime_s"] >= 1000].copy()
    if extreme_rows.empty:
        print("No Monaco HARD rows with LapTime_s >= 1000 seconds were found.")
        excluded_indices = []
    else:
        extreme_rows["diagnostic_note"] = extreme_rows.apply(classify_extreme_row, axis=1)
        print("\nRows producing the ~2,500 second extreme-value cluster:")
        print(
            extreme_rows[
                [
                    "Year",
                    "LapNumber",
                    "Driver",
                    "TyreLife",
                    "LapTime_s",
                    "TrackStatus",
                    "PitInTime",
                    "PitOutTime",
                    "sc_active",
                    "vsc_active",
                    "diagnostic_note",
                ]
            ].sort_values("LapTime_s", ascending=False).to_string(index=False)
        )
        excluded_indices = extreme_rows.index.tolist()

    eligible = degradation_eligible_subset(subset)
    original_fit = fit_degradation_slope(eligible)
    corrected_eligible = eligible.loc[~eligible.index.isin(excluded_indices)].copy()
    corrected_fit = fit_degradation_slope(corrected_eligible)

    print("\nDegradation-fit basis: Monaco HARD rows filtered with the same eligibility rules as build_tyre_degradation().")
    print(
        f"Original fit: model={original_fit['model']}, slope={original_fit['slope']}, "
        f"sample_size={original_fit['n']}, r2_linear={original_fit['r2_linear']}, "
        f"r2_quadratic={original_fit['r2_quadratic']}"
    )
    print(
        f"Corrected fit excluding identified extreme rows: model={corrected_fit['model']}, slope={corrected_fit['slope']}, "
        f"sample_size={corrected_fit['n']}, r2_linear={corrected_fit['r2_linear']}, "
        f"r2_quadratic={corrected_fit['r2_quadratic']}"
    )

    DIAGNOSTIC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corrected_artifact = DIAGNOSTIC_OUTPUT_DIR / "monaco_hard_corrected_excluding_extreme_rows.csv"
    corrected_eligible.to_csv(corrected_artifact, index=False)
    print(f"Saved corrected Monaco HARD degradation artefact: {corrected_artifact}")


def main() -> None:
    df = load_base_df()
    relabeled, pit_eval = reapply_corrected_labels(df)
    print_corrected_label_verification(df, relabeled, pit_eval)
    print_monaco_outlier_diagnostic(df)


if __name__ == "__main__":
    main()