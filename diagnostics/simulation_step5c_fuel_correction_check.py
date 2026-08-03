from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation.config import BASE_DF_PATH
from source_code.simulation.monte_carlo import build_actual_strategies, select_midfield_subject_driver
from source_code.simulation.optimiser import (
    ARGMAX_BATCH_SIZE,
    _build_caution_schedule_matrix,
    _enumerate_unconditioned_subject_strategies,
    _precompute_subject_inputs,
    _resolve_probability_by_source,
)
from source_code.simulation.race_model_vectorized import evaluate_strategy_batch, rank_and_score_batch
from source_code.simulation.race_state import load_race_state


RAW_STATS_PATH = ROOT / "data" / "diagnostics" / "deg_rate_full_peryear_stats.csv"
CORRECTED_STATS_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_full_peryear_stats.csv"
OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step5c_fuel_correction_check_output.txt"
FEATURES_PATH = ROOT / "src" / "features.py"

HUNGARY_2022 = {"event_name": "Hungarian Grand Prix", "year": 2022, "round": 13}
ITALY_2024 = {"event_name": "Italian Grand Prix", "year": 2024, "round": 16}

PROB_SOURCE = "lap_level"
INFORMATION_SET = "unconditioned"
LAMBDA_VALUE = 1.0
N_ITERATIONS = 10_000
SEED = 42
EPSILON = 0.01

RAW_HUNGARY_TOP100_SHORT_FRAC = 0.39
RAW_ITALY_MEAN_POINTS = 25.0
RAW_ITALY_WIN_RATE = 1.0


def emit(text: str = "") -> None:
    print(text, flush=True)


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _fit_group(group: pd.DataFrame, y_col: str) -> Dict[str, float]:
    if len(group) == 0:
        return {
            "n_laps": 0,
            "model": "missing",
            "deg_rate_internal": np.nan,
            "deg_rate_final_assigned": np.nan,
            "r2_linear": np.nan,
            "r2_quadratic": np.nan,
            "low_sample": True,
        }

    if len(group) < 5:
        return {
            "n_laps": int(len(group)),
            "model": "insufficient",
            "deg_rate_internal": np.nan,
            "deg_rate_final_assigned": np.nan,
            "r2_linear": np.nan,
            "r2_quadratic": np.nan,
            "low_sample": bool(len(group) < 30),
        }

    x = pd.to_numeric(group["TyreLife"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(group[y_col], errors="coerce").to_numpy(dtype=float)

    if np.isnan(x).any() or np.isnan(y).any():
        return {
            "n_laps": int(len(group)),
            "model": "invalid_data",
            "deg_rate_internal": np.nan,
            "deg_rate_final_assigned": np.nan,
            "r2_linear": np.nan,
            "r2_quadratic": np.nan,
            "low_sample": bool(len(group) < 30),
        }

    coeffs_linear = np.polyfit(x, y, 1)
    y_pred_linear = np.polyval(coeffs_linear, x)
    ss_res_linear = float(np.sum((y - y_pred_linear) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_linear = 1.0 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0.0

    coeffs_quadratic = np.polyfit(x, y, 2)
    y_pred_quadratic = np.polyval(coeffs_quadratic, x)
    ss_res_quadratic = float(np.sum((y - y_pred_quadratic) ** 2))
    r2_quadratic = 1.0 - (ss_res_quadratic / ss_tot) if ss_tot > 0 else 0.0

    if r2_quadratic > r2_linear + 0.02:
        a = float(coeffs_quadratic[0])
        b = float(coeffs_quadratic[1])
        mean_tyre_life = float(x.mean())
        slope = (2.0 * a * mean_tyre_life) + b
        model = "quadratic"
    else:
        slope = float(coeffs_linear[0])
        model = "linear"

    final_assigned = slope
    if np.isfinite(slope) and abs(slope) > 1.0:
        final_assigned = np.nan

    return {
        "n_laps": int(len(group)),
        "model": model,
        "deg_rate_internal": float(slope),
        "deg_rate_final_assigned": float(final_assigned) if pd.notna(final_assigned) else np.nan,
        "r2_linear": float(r2_linear),
        "r2_quadratic": float(r2_quadratic),
        "low_sample": bool(len(group) < 30),
    }


def _raw_style_eligibility_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    mask &= df["compound_known"].eq(True)
    mask &= df["IsAccurate"].eq(True)
    mask &= ~df["sc_active"].fillna(False)
    mask &= ~df["vsc_active"].fillna(False)
    mask &= df["PitInTime"].isna()
    mask &= df["PitOutTime"].isna()
    mask &= df["TyreLife"].notna()
    mask &= df["LapTime_s"].notna()
    wet_compounds = {"INTERMEDIATE", "WET"}
    mask &= ~df["TyreCompound"].isin(wet_compounds)

    outlier_mask = mask.copy()
    for (driver, round_num), group in df[outlier_mask].groupby(["Driver", "Round"]):
        median_time = pd.to_numeric(group["LapTime_s"], errors="coerce").median()
        threshold = median_time * 1.07
        bad_idx = group[pd.to_numeric(group["LapTime_s"], errors="coerce") > threshold].index
        mask.loc[bad_idx] = False

    return mask


def _print_verbatim_corrected_fit_logic() -> None:
    emit("=" * 100)
    emit("Step 1 - Locate corrected-degradation fitting logic in features")
    emit("=" * 100)

    lines = FEATURES_PATH.read_text(encoding="utf-8").splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines, start=1):
        if "# Build eligibility mask same as build_tyre_degradation()" in line:
            start = idx
            break
    if start is None:
        raise RuntimeError("Could not find corrected-fit eligibility block in features.py")

    for idx in range(start, len(lines) + 1):
        if "df['deg_rate_corrected'] = deg_rate_values" in lines[idx - 1]:
            end = idx - 1
            break
    if end is None:
        raise RuntimeError("Could not find corrected-fit block terminator in features.py")

    emit(f"features_py_block_path={FEATURES_PATH}")
    emit(f"features_py_block_lines={start}-{end}")
    emit("verbatim_corrected_fit_logic_begin")
    for line_no in range(start, end + 1):
        emit(lines[line_no - 1])
    emit("verbatim_corrected_fit_logic_end")

    has_grouping = False
    for line_no in range(start, end + 1):
        if "groupby(['TyreCompound', 'EventName', 'Year'])" in lines[line_no - 1]:
            has_grouping = True
            break
    emit(
        "corrected_logic_grouping_confirmation="
        + (
            "Yes, it fits per (TyreCompound, EventName, Year), matching the existing per-year stats grouping keys."
            if has_grouping
            else "No grouping match found."
        )
    )


def _build_corrected_stats(raw_table: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    eligible_mask = _raw_style_eligibility_mask(base_df)
    eligible = base_df[eligible_mask].copy()

    corrected_rows = []
    for row in raw_table.itertuples(index=False):
        compound = str(row.TyreCompound)
        event_name = str(row.EventName)
        year = int(row.Year)
        group = eligible[
            (eligible["TyreCompound"] == compound)
            & (eligible["EventName"] == event_name)
            & (eligible["Year"] == year)
        ].copy()

        fitted = _fit_group(group, "fuel_corrected_laptime")
        corrected_rows.append(
            {
                "TyreCompound": compound,
                "EventName": event_name,
                "Year": year,
                "n_laps": fitted["n_laps"],
                "model": fitted["model"],
                "deg_rate_internal": fitted["deg_rate_internal"],
                "deg_rate_final_assigned": fitted["deg_rate_final_assigned"],
                "r2_linear": fitted["r2_linear"],
                "r2_quadratic": fitted["r2_quadratic"],
                "low_sample": fitted["low_sample"],
            }
        )

    corrected_df = pd.DataFrame(corrected_rows)
    return corrected_df


def _prepare_search_from_state(state: dict, race_ref: Mapping[str, object]) -> Dict[str, object]:
    laps = state["laps"].copy()
    laps["sc_active"] = _to_bool(laps["sc_active"])
    laps["vsc_active"] = _to_bool(laps["vsc_active"])

    subject_driver = select_midfield_subject_driver(laps)
    actual_strategies = build_actual_strategies(laps)
    subject_actual = dict(actual_strategies[subject_driver])
    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())

    prob_by_lap = _resolve_probability_by_source(state, PROB_SOURCE)
    caution_schedules = _build_caution_schedule_matrix(prob_by_lap, N_ITERATIONS, np.random.default_rng(SEED))
    strategies = _enumerate_unconditioned_subject_strategies(race_length, subject_actual)
    precomputed = _precompute_subject_inputs(
        race_state=state,
        subject_driver=subject_driver,
        caution_schedules=caution_schedules,
        lambda_=LAMBDA_VALUE,
    )

    return {
        "race_ref": dict(race_ref),
        "race_state": state,
        "laps": laps,
        "subject_driver": subject_driver,
        "subject_actual": subject_actual,
        "actual_strategies": actual_strategies,
        "race_length": race_length,
        "caution_schedules": caution_schedules,
        "strategies": strategies,
        "precomputed": precomputed,
    }


def _score_strategies(
    search: Mapping[str, object],
    strategies: Sequence[Mapping[str, object]],
    progress_label: str,
) -> Dict[str, np.ndarray]:
    mean_points = np.zeros(len(strategies), dtype=np.float64)
    std_points = np.zeros(len(strategies), dtype=np.float64)
    mean_rank = np.zeros(len(strategies), dtype=np.float64)

    n_batches = (len(strategies) + ARGMAX_BATCH_SIZE - 1) // ARGMAX_BATCH_SIZE
    for batch_idx, start in enumerate(range(0, len(strategies), ARGMAX_BATCH_SIZE), start=1):
        end = min(start + ARGMAX_BATCH_SIZE, len(strategies))
        batch = list(strategies[start:end])
        subject_times = evaluate_strategy_batch(
            subject_strategies=batch,
            caution_schedules=search["caution_schedules"],
            degradation_lookup=search["precomputed"]["degradation_lookup"],
            lambda_=LAMBDA_VALUE,
            baseline_pace_s=float(search["subject_actual"]["baseline_pace_s"]),
            pit_loss_s=float(search["precomputed"]["pit_loss_s"]),
            caution_pit_loss_s=5.0,
            caution_pace_ratio=1.3162357008284764,
        )
        scored = rank_and_score_batch(subject_times, search["precomputed"]["rival_times"])
        mean_points[start:end] = np.asarray(scored["mean_points"], dtype=np.float64)
        std_points[start:end] = np.asarray(scored["std_points"], dtype=np.float64)
        mean_rank[start:end] = np.asarray(scored["mean_rank"], dtype=np.float64)
        if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == n_batches:
            emit(
                f"batch_progress label={progress_label} "
                f"batch={batch_idx}/{n_batches} scored={end}/{len(strategies)}"
            )

    return {
        "mean_points": mean_points,
        "std_points": std_points,
        "mean_rank": mean_rank,
    }


def _evaluate_single_strategy(search: Mapping[str, object], strategy: Mapping[str, object]) -> Dict[str, float]:
    subject_times = evaluate_strategy_batch(
        subject_strategies=[dict(strategy)],
        caution_schedules=search["caution_schedules"],
        degradation_lookup=search["precomputed"]["degradation_lookup"],
        lambda_=LAMBDA_VALUE,
        baseline_pace_s=float(search["subject_actual"]["baseline_pace_s"]),
        pit_loss_s=float(search["precomputed"]["pit_loss_s"]),
        caution_pit_loss_s=5.0,
        caution_pace_ratio=1.3162357008284764,
    )
    scored = rank_and_score_batch(subject_times, search["precomputed"]["rival_times"])
    rank_matrix = np.asarray(scored["rank_matrix"][0], dtype=np.int16)
    points_matrix = np.asarray(scored["points_matrix"][0], dtype=np.float64)
    return {
        "mean_points": float(points_matrix.mean()),
        "win_rate": float(np.mean(rank_matrix == 1)),
        "mean_rank": float(rank_matrix.mean()),
    }


def _strategy_to_key(strategy: Mapping[str, object]) -> Tuple[str, Tuple[Tuple[int, str], ...]]:
    return (
        str(strategy["starting_compound"]),
        tuple((int(l), str(c)) for l, c in strategy.get("stops", [])),
    )


def _load_corrected_race_state(year: int, round_num: int, corrected_stats: pd.DataFrame) -> dict:
    state = load_race_state(year, round_num)
    event_name = str(state["laps"]["EventName"].dropna().iloc[0])
    subset = corrected_stats[
        (corrected_stats["Year"] == int(year))
        & (corrected_stats["EventName"] == event_name)
    ].copy()
    state["deg_stats"] = subset
    return state


def main() -> None:
    emit("=" * 100)
    emit("Step 5c setup")
    emit("=" * 100)
    emit(f"raw_stats_path={RAW_STATS_PATH}")
    emit(f"corrected_stats_output_path={CORRECTED_STATS_PATH}")
    emit(f"report_output_path={OUTPUT_PATH}")

    _print_verbatim_corrected_fit_logic()

    emit("")
    emit("=" * 100)
    emit("Step 2 - Build corrected per-year stats table")
    emit("=" * 100)

    raw_stats = pd.read_csv(RAW_STATS_PATH)
    base_df = pd.read_csv(BASE_DF_PATH)
    corrected_stats = _build_corrected_stats(raw_stats, base_df)
    CORRECTED_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    corrected_stats.to_csv(CORRECTED_STATS_PATH, index=False)

    emit(f"raw_row_count={len(raw_stats)}")
    emit(f"corrected_row_count={len(corrected_stats)}")
    emit(f"row_count_match={len(raw_stats) == len(corrected_stats)}")

    model_diff = corrected_stats[["TyreCompound", "EventName", "Year", "model"]].merge(
        raw_stats[["TyreCompound", "EventName", "Year", "model"]],
        on=["TyreCompound", "EventName", "Year"],
        suffixes=("_corrected", "_raw"),
        how="inner",
    )
    model_diff = model_diff[model_diff["model_corrected"] != model_diff["model_raw"]].copy()

    low_sample_diff = corrected_stats[["TyreCompound", "EventName", "Year", "low_sample"]].merge(
        raw_stats[["TyreCompound", "EventName", "Year", "low_sample"]],
        on=["TyreCompound", "EventName", "Year"],
        suffixes=("_corrected", "_raw"),
        how="inner",
    )
    low_sample_diff = low_sample_diff[low_sample_diff["low_sample_corrected"] != low_sample_diff["low_sample_raw"]].copy()

    emit(f"model_status_flag_diff_count={len(model_diff)}")
    if len(model_diff) > 0:
        emit("model_status_flag_diffs_begin")
        for row in model_diff.to_dict(orient="records"):
            emit(json.dumps(row, sort_keys=True))
        emit("model_status_flag_diffs_end")

    emit(f"low_sample_flag_diff_count={len(low_sample_diff)}")
    if len(low_sample_diff) > 0:
        emit("low_sample_flag_diffs_begin")
        for row in low_sample_diff.to_dict(orient="records"):
            emit(json.dumps(row, sort_keys=True))
        emit("low_sample_flag_diffs_end")

    emit("")
    emit("=" * 100)
    emit("Step 3 - Hungary 2022 and Italian GP 2024 slope shifts (raw vs corrected)")
    emit("=" * 100)

    def print_compound_shift(event_name: str, year: int) -> None:
        raw_sub = raw_stats[(raw_stats["EventName"] == event_name) & (raw_stats["Year"] == year)].copy()
        cor_sub = corrected_stats[(corrected_stats["EventName"] == event_name) & (corrected_stats["Year"] == year)].copy()
        merged = raw_sub.merge(
            cor_sub,
            on=["TyreCompound", "EventName", "Year"],
            suffixes=("_raw", "_corrected"),
            how="outer",
        )
        emit(f"race={event_name} year={year}")
        for compound in ["HARD", "MEDIUM", "SOFT"]:
            row = merged[merged["TyreCompound"] == compound]
            if row.empty:
                emit(f"  {compound}: missing in both")
                continue
            raw_val = row["deg_rate_final_assigned_raw"].iloc[0] if "deg_rate_final_assigned_raw" in row.columns else np.nan
            cor_val = row["deg_rate_final_assigned_corrected"].iloc[0] if "deg_rate_final_assigned_corrected" in row.columns else np.nan
            shift = (cor_val - raw_val) if pd.notna(raw_val) and pd.notna(cor_val) else np.nan
            emit(
                f"  {compound}: raw={raw_val} corrected={cor_val} shift={shift}"
            )

    print_compound_shift(HUNGARY_2022["event_name"], HUNGARY_2022["year"])
    print_compound_shift(ITALY_2024["event_name"], ITALY_2024["year"])

    monza_medium_row = corrected_stats[
        (corrected_stats["EventName"] == ITALY_2024["event_name"])
        & (corrected_stats["Year"] == ITALY_2024["year"])
        & (corrected_stats["TyreCompound"] == "MEDIUM")
    ]
    monza_medium = monza_medium_row["deg_rate_final_assigned"].iloc[0] if not monza_medium_row.empty else np.nan
    if pd.isna(monza_medium):
        monza_state = "missing"
    elif monza_medium > 1e-6:
        monza_state = "positive"
    elif monza_medium < -1e-6:
        monza_state = "negative"
    else:
        monza_state = "near_zero"
    emit(f"italian_gp_2024_medium_corrected_sign_state={monza_state}")

    emit("")
    emit("=" * 100)
    emit("Step 4 - Hungary short-stint concentration under corrected values")
    emit("=" * 100)

    hungary_state_corrected = _load_corrected_race_state(HUNGARY_2022["year"], HUNGARY_2022["round"], corrected_stats)
    hungary_search = _prepare_search_from_state(hungary_state_corrected, HUNGARY_2022)
    hungary_scores = _score_strategies(
        hungary_search,
        hungary_search["strategies"],
        progress_label="hungary_full_search_corrected",
    )

    first_stint_lengths = np.array([
        int(strategy["stops"][0][0]) if len(strategy.get("stops", [])) > 0 else int(hungary_search["race_length"])
        for strategy in hungary_search["strategies"]
    ])
    distribution = pd.Series(first_stint_lengths).value_counts().sort_index()

    emit(f"hungary_strategy_count_corrected={len(hungary_search['strategies'])}")
    emit("hungary_first_stint_length_distribution_corrected_begin")
    for stint_length, count in distribution.items():
        emit(
            f"first_stint_length={int(stint_length)} count={int(count)} share={float(count/len(first_stint_lengths)):.6f}"
        )
    emit("hungary_first_stint_length_distribution_corrected_end")

    top100_idx = np.argsort(-hungary_scores["mean_points"])[:100]
    top100_short_count = int((first_stint_lengths[top100_idx] <= 3).sum())
    top100_short_frac = top100_short_count / 100.0
    emit(f"hungary_top100_short_first_stint_leq3_count_corrected={top100_short_count}")
    emit(f"hungary_top100_short_first_stint_leq3_fraction_corrected={top100_short_frac:.6f}")
    emit(f"hungary_top100_short_first_stint_leq3_fraction_raw_reference={RAW_HUNGARY_TOP100_SHORT_FRAC:.6f}")
    emit(f"hungary_top100_short_first_stint_leq3_fraction_delta_corrected_minus_raw={top100_short_frac - RAW_HUNGARY_TOP100_SHORT_FRAC:.6f}")

    emit("")
    emit("=" * 100)
    emit("Step 5 - Italian checks under corrected values")
    emit("=" * 100)

    italy_state_corrected = _load_corrected_race_state(ITALY_2024["year"], ITALY_2024["round"], corrected_stats)
    italy_search = _prepare_search_from_state(italy_state_corrected, ITALY_2024)

    specific_strategy = {
        "starting_compound": "MEDIUM",
        "stops": [(37, "SOFT")],
        "baseline_pace_s": float(italy_search["subject_actual"]["baseline_pace_s"]),
        "year": int(ITALY_2024["year"]),
        "round": int(ITALY_2024["round"]),
        "event_name": str(ITALY_2024["event_name"]),
    }
    specific_eval = _evaluate_single_strategy(italy_search, specific_strategy)
    emit("italy_specific_strategy=" + json.dumps(specific_strategy, sort_keys=True))
    emit(f"italy_specific_strategy_mean_points_corrected={specific_eval['mean_points']:.6f}")
    emit(f"italy_specific_strategy_win_rate_corrected={specific_eval['win_rate']:.6f}")
    emit(f"italy_specific_strategy_mean_rank_corrected={specific_eval['mean_rank']:.6f}")

    italy_scores = _score_strategies(
        italy_search,
        italy_search["strategies"],
        progress_label="italy_full_search_corrected",
    )
    italy_best_idx = int(np.nanargmax(italy_scores["mean_points"]))
    italy_best_mean = float(italy_scores["mean_points"][italy_best_idx])
    italy_best_std = float(italy_scores["std_points"][italy_best_idx])
    italy_best_rank = float(italy_scores["mean_rank"][italy_best_idx])
    italy_epsilon_mask = italy_scores["mean_points"] >= (italy_best_mean - EPSILON)
    italy_within_epsilon_count = int(np.sum(italy_epsilon_mask))
    best_strategy_corrected = italy_search["strategies"][italy_best_idx]

    best_key = _strategy_to_key(best_strategy_corrected)
    specific_key = _strategy_to_key(specific_strategy)
    same_winner = best_key == specific_key

    emit("italy_corrected_argmax_strategy=" + json.dumps(best_strategy_corrected, sort_keys=True))
    emit(f"italy_corrected_argmax_mean_points={italy_best_mean:.6f}")
    emit(f"italy_corrected_argmax_std_points={italy_best_std:.6f}")
    emit(f"italy_corrected_argmax_mean_rank={italy_best_rank:.6f}")
    emit(f"italy_corrected_argmax_within_epsilon_count={italy_within_epsilon_count}")
    emit(f"italy_corrected_argmax_same_as_specific_37_16={same_winner}")
    emit(f"italy_corrected_margin_argmax_minus_specific_37_16={italy_best_mean - specific_eval['mean_points']:.6f}")

    emit(f"italy_raw_reference_mean_points={RAW_ITALY_MEAN_POINTS:.6f}")
    emit(f"italy_raw_reference_win_rate={RAW_ITALY_WIN_RATE:.6f}")

    emit("")
    emit("=" * 100)
    emit("Step 6 - Decision-ready summary")
    emit("=" * 100)

    hungary_delta = top100_short_frac - RAW_HUNGARY_TOP100_SHORT_FRAC
    italy_mean_delta = specific_eval["mean_points"] - RAW_ITALY_MEAN_POINTS
    italy_win_delta = specific_eval["win_rate"] - RAW_ITALY_WIN_RATE

    if hungary_delta <= -0.10:
        hungary_resolution = "substantial_reduction"
    elif hungary_delta < 0:
        hungary_resolution = "partial_reduction"
    else:
        hungary_resolution = "no_meaningful_reduction"

    if specific_eval["win_rate"] < 0.95 and specific_eval["mean_points"] < 24.0:
        italy_resolution = "substantial_reduction"
    elif specific_eval["win_rate"] < RAW_ITALY_WIN_RATE or specific_eval["mean_points"] < RAW_ITALY_MEAN_POINTS:
        italy_resolution = "partial_reduction"
    else:
        italy_resolution = "no_meaningful_reduction"

    if hungary_resolution.startswith("substantial") and italy_resolution.startswith("substantial"):
        overall = "both_substantially_reduced"
    elif hungary_resolution.startswith("substantial") or italy_resolution.startswith("substantial"):
        overall = "one_substantially_reduced"
    elif hungary_resolution == "partial_reduction" or italy_resolution == "partial_reduction":
        overall = "partial_only"
    else:
        overall = "neither_resolved"

    emit(f"hungary_resolution_assessment={hungary_resolution}")
    emit(f"italy_resolution_assessment={italy_resolution}")
    emit(f"overall_resolution_assessment={overall}")
    emit(f"hungary_short_stint_top100_fraction_delta={hungary_delta:.6f}")
    emit(f"italy_specific_strategy_mean_points_delta={italy_mean_delta:.6f}")
    emit(f"italy_specific_strategy_win_rate_delta={italy_win_delta:.6f}")
    emit("minimum_stint_floor_still_warranted=True")
    emit("minimum_stint_floor_defense_in_depth_reason=Even with fuel-corrected slopes, residual asymmetry and model uncertainty can still create unrealistically short-stint incentives; a physically grounded minimum stint remains necessary as defense in depth.")

    emit("")
    emit(f"saved_corrected_stats_csv={CORRECTED_STATS_PATH}")
    emit(f"report_complete_output_target={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
