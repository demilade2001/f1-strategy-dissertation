from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation.config import (
    BASE_DF_PATH,
    CAUTION_PACE_RATIO,
    CAUTION_PIT_LOSS_S,
    DEG_RATE_STATS_PATH,
    GRID_SPACING,
    LOCKED_ARCHETYPE_RACES,
    MAX_STOPS,
    MIN_STINT_LENGTH_LAPS,
)
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
from source_code.simulation.strategy import is_strategy_feasible

CORRECTED_STATS_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_full_peryear_stats.csv"
SHRUNK_STATS_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_shrunk_full_peryear_stats.csv"

PROB_SOURCE = "lap_level"
N_ITERATIONS = 10_000
LAMBDA_VALUE = 1.0
EPSILON = 0.01
RNG_SEED = 42

FOUR_RACES = [
    {"event_name": "Hungarian Grand Prix", "year": 2022, "round": 13},
    {"event_name": "Italian Grand Prix", "year": 2024, "round": 16},
    {"event_name": "Bahrain Grand Prix", "year": 2024, "round": 1},
    {"event_name": "Azerbaijan Grand Prix", "year": 2024, "round": 17},
]


def emit(text: str = "") -> None:
    print(text, flush=True)


def _selected_r2(row: pd.Series) -> float:
    if row["model"] == "quadratic":
        return float(row["r2_quadratic"])
    return float(row["r2_linear"])


def _is_eligible_fit(row: pd.Series) -> bool:
    if bool(row.get("low_sample", False)):
        return False
    if str(row.get("model", "")) in {"invalid_data", "insufficient", "missing"}:
        return False
    return pd.notna(row.get("deg_rate_final_assigned"))


def _build_shrunk_table(corrected: pd.DataFrame, cutoff: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    locked = pd.DataFrame(LOCKED_ARCHETYPE_RACES)
    table = corrected.copy()
    table["selected_r2"] = table.apply(_selected_r2, axis=1)
    table["is_eligible_fit"] = table.apply(_is_eligible_fit, axis=1)

    locked_rows = table.merge(
        locked[["event_name", "year", "round"]],
        left_on=["EventName", "Year"],
        right_on=["event_name", "year"],
        how="inner",
    ).copy()

    locked_rows["is_thin_r2"] = locked_rows["is_eligible_fit"] & (locked_rows["selected_r2"] < cutoff)

    non_thin_pool = locked_rows[locked_rows["is_eligible_fit"] & ~locked_rows["is_thin_r2"]].copy()

    def compound_target(compound: str) -> float:
        values = non_thin_pool[non_thin_pool["TyreCompound"] == compound]["deg_rate_final_assigned"].dropna()
        return float(values.median()) if not values.empty else np.nan

    table["shrink_target_locked_compound_median"] = np.nan
    table["is_thin_r2"] = False
    table["deg_rate_pre_shrink"] = table["deg_rate_final_assigned"]

    # Apply shrinkage to locked-race rows only; target excludes current race-season row.
    for idx, row in locked_rows.iterrows():
        global_mask = (
            (table["EventName"] == row["EventName"]) &
            (table["Year"] == row["Year"]) &
            (table["TyreCompound"] == row["TyreCompound"])
        )
        thin_flag = bool(row["is_thin_r2"])
        table.loc[global_mask, "is_thin_r2"] = thin_flag
        if not thin_flag:
            continue

        pool = non_thin_pool[
            (non_thin_pool["TyreCompound"] == row["TyreCompound"]) &
            ~(
                (non_thin_pool["EventName"] == row["EventName"]) &
                (non_thin_pool["Year"] == row["Year"])
            )
        ]["deg_rate_final_assigned"].dropna()
        target = float(pool.median()) if not pool.empty else compound_target(str(row["TyreCompound"]))
        table.loc[global_mask, "shrink_target_locked_compound_median"] = target
        if pd.notna(target):
            table.loc[global_mask, "deg_rate_final_assigned"] = target

    locked_eval_rows = table.merge(
        locked[["event_name", "year", "round"]],
        left_on=["EventName", "Year"],
        right_on=["event_name", "year"],
        how="right",
    ).copy()

    return table, locked_eval_rows


def _derive_min_stint(locked_eval_rows: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    base_df = pd.read_csv(BASE_DF_PATH)
    locked = pd.DataFrame(LOCKED_ARCHETYPE_RACES)
    pit_losses = (
        base_df.merge(
            locked[["event_name", "year", "round"]],
            left_on=["EventName", "Year", "Round"],
            right_on=["event_name", "year", "round"],
            how="inner",
        )
        .groupby(["EventName", "Year", "Round"], as_index=False)["pit_loss_s"]
        .median()
    )

    rows: List[Dict[str, object]] = []
    for race in LOCKED_ARCHETYPE_RACES:
        event_name = race["event_name"]
        year = int(race["year"])
        round_num = int(race["round"])

        race_rows = locked_eval_rows[
            (locked_eval_rows["event_name"] == event_name) &
            (locked_eval_rows["year"] == year)
        ].copy()
        race_rows = race_rows[race_rows["deg_rate_final_assigned"].notna()].copy()

        pit_match = pit_losses[
            (pit_losses["EventName"] == event_name) &
            (pit_losses["Year"] == year) &
            (pit_losses["Round"] == round_num)
        ]
        pit_loss = float(pit_match.iloc[0]["pit_loss_s"]) if not pit_match.empty else np.nan

        if len(race_rows) < 2:
            rows.append(
                {
                    "EventName": event_name,
                    "Year": year,
                    "Round": round_num,
                    "available_compounds": int(len(race_rows)),
                    "high_deg_compound": None,
                    "low_deg_compound": None,
                    "spread_high_minus_low": np.nan,
                    "pit_loss_s": pit_loss,
                    "breakeven_laps": np.nan,
                    "breakeven_laps_ceiling": np.nan,
                }
            )
            continue

        high_row = race_rows.loc[race_rows["deg_rate_final_assigned"].idxmax()]
        low_row = race_rows.loc[race_rows["deg_rate_final_assigned"].idxmin()]
        spread = float(high_row["deg_rate_final_assigned"] - low_row["deg_rate_final_assigned"])

        if spread <= 0 or not np.isfinite(pit_loss):
            breakeven = np.nan
        else:
            # Solve spread * T(L) = pit_loss where T(L)=L(L+1)/2.
            breakeven = 0.5 * (-1.0 + math.sqrt(1.0 + (8.0 * pit_loss / spread)))

        rows.append(
            {
                "EventName": event_name,
                "Year": year,
                "Round": round_num,
                "available_compounds": int(len(race_rows)),
                "high_deg_compound": str(high_row["TyreCompound"]),
                "low_deg_compound": str(low_row["TyreCompound"]),
                "spread_high_minus_low": spread,
                "pit_loss_s": pit_loss,
                "breakeven_laps": breakeven,
                "breakeven_laps_ceiling": math.ceil(breakeven) if np.isfinite(breakeven) else np.nan,
            }
        )

    spread_table = pd.DataFrame(rows)
    candidate = spread_table[spread_table["spread_high_minus_low"].notna()].copy()
    worst = candidate.loc[candidate["spread_high_minus_low"].idxmax()].to_dict()
    return spread_table, worst


def _with_subject_metadata(strategy: dict, subject_actual: dict) -> dict:
    enriched = dict(strategy)
    enriched.setdefault("baseline_pace_s", float(subject_actual["baseline_pace_s"]))
    enriched.setdefault("year", int(subject_actual["year"]))
    enriched.setdefault("round", int(subject_actual["round"]))
    enriched.setdefault("event_name", str(subject_actual["event_name"]))
    return enriched


def _run_argmax_with_progress(race_ref: Mapping[str, object]) -> dict:
    race_state = load_race_state(int(race_ref["year"]), int(race_ref["round"]))
    laps = race_state["laps"].copy()
    subject_driver = select_midfield_subject_driver(laps)
    actual_strategies = build_actual_strategies(laps)
    subject_actual = dict(actual_strategies[subject_driver])

    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    prob_by_lap = _resolve_probability_by_source(race_state, PROB_SOURCE)
    caution_schedules = _build_caution_schedule_matrix(prob_by_lap, N_ITERATIONS, np.random.default_rng(RNG_SEED))

    strategies = _enumerate_unconditioned_subject_strategies(race_length, subject_actual)
    if not strategies:
        raise ValueError(f"No feasible strategies after constraints for {race_ref}")

    precomputed = _precompute_subject_inputs(
        race_state=race_state,
        subject_driver=subject_driver,
        caution_schedules=caution_schedules,
        lambda_=LAMBDA_VALUE,
    )

    mean_points = np.zeros(len(strategies), dtype=np.float64)
    std_points = np.zeros(len(strategies), dtype=np.float64)
    mean_rank = np.zeros(len(strategies), dtype=np.float64)

    n_batches = (len(strategies) + ARGMAX_BATCH_SIZE - 1) // ARGMAX_BATCH_SIZE
    for batch_idx, start in enumerate(range(0, len(strategies), ARGMAX_BATCH_SIZE), start=1):
        end = min(start + ARGMAX_BATCH_SIZE, len(strategies))
        batch = strategies[start:end]
        subject_times = evaluate_strategy_batch(
            subject_strategies=batch,
            caution_schedules=caution_schedules,
            degradation_lookup=precomputed["degradation_lookup"],
            lambda_=LAMBDA_VALUE,
            baseline_pace_s=float(subject_actual["baseline_pace_s"]),
            pit_loss_s=float(precomputed["pit_loss_s"]),
            caution_pit_loss_s=CAUTION_PIT_LOSS_S,
            caution_pace_ratio=CAUTION_PACE_RATIO,
        )
        scored = rank_and_score_batch(subject_times, precomputed["rival_times"])
        mean_points[start:end] = np.asarray(scored["mean_points"], dtype=np.float64)
        std_points[start:end] = np.asarray(scored["std_points"], dtype=np.float64)
        mean_rank[start:end] = np.asarray(scored["mean_rank"], dtype=np.float64)

        if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == n_batches:
            emit(
                f"batch_progress race={race_ref['event_name']} year={race_ref['year']} "
                f"batch={batch_idx}/{n_batches} scored={end}/{len(strategies)}"
            )

    best_idx = int(np.argmax(mean_points))
    best_strategy = _with_subject_metadata(strategies[best_idx], subject_actual)
    best_mean_points = float(mean_points[best_idx])

    epsilon_cutoff = best_mean_points - EPSILON
    within_eps = int(np.sum(mean_points >= epsilon_cutoff))

    best_subject_times = evaluate_strategy_batch(
        subject_strategies=[best_strategy],
        caution_schedules=caution_schedules,
        degradation_lookup=precomputed["degradation_lookup"],
        lambda_=LAMBDA_VALUE,
        baseline_pace_s=float(subject_actual["baseline_pace_s"]),
        pit_loss_s=float(precomputed["pit_loss_s"]),
        caution_pit_loss_s=CAUTION_PIT_LOSS_S,
        caution_pace_ratio=CAUTION_PACE_RATIO,
    )
    best_scored = rank_and_score_batch(best_subject_times, precomputed["rival_times"])
    rank_matrix = np.asarray(best_scored["rank_matrix"][0], dtype=np.int16)
    win_rate = float(np.mean(rank_matrix == 1))

    first_stint = int(best_strategy["stops"][0][0]) if best_strategy.get("stops") else race_length

    return {
        "race": dict(race_ref),
        "subject_driver": subject_driver,
        "strategy_count": int(len(strategies)),
        "best_strategy": best_strategy,
        "best_mean_points": best_mean_points,
        "best_std_points": float(std_points[best_idx]),
        "best_mean_rank": float(mean_rank[best_idx]),
        "within_epsilon_count": within_eps,
        "best_win_rate": win_rate,
        "best_first_stint_length": first_stint,
    }


def _actual_strategy_feasibility_for_race(race_ref: Mapping[str, object]) -> dict:
    race_state = load_race_state(int(race_ref["year"]), int(race_ref["round"]))
    laps = race_state["laps"].copy()
    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    actual = build_actual_strategies(laps)

    failures = []
    for driver, strategy in sorted(actual.items()):
        stops = [(int(l), str(c)) for l, c in strategy.get("stops", [])]
        feasible = is_strategy_feasible(
            race_length=race_length,
            starting_compound=str(strategy["starting_compound"]),
            is_wet_race=False,
            stops=stops,
            dry_compounds=["SOFT", "MEDIUM", "HARD"],
            max_stops=MAX_STOPS,
            min_stint_length_laps=MIN_STINT_LENGTH_LAPS,
        )
        if feasible:
            continue

        boundaries = [0] + [int(l) for l, _ in stops] + [race_length]
        stint_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        short_stints = [length for length in stint_lengths if length < MIN_STINT_LENGTH_LAPS]
        failures.append(
            {
                "driver": driver,
                "strategy": strategy,
                "stint_lengths": stint_lengths,
                "short_stints": short_stints,
            }
        )

    return {
        "race": dict(race_ref),
        "driver_count": int(len(actual)),
        "failure_count": int(len(failures)),
        "failures": failures,
    }


def main() -> None:
    emit("=" * 100)
    emit("Step 5d - Combined fix validation")
    emit("=" * 100)
    emit(f"corrected_input_path={CORRECTED_STATS_PATH}")
    emit(f"shrunk_output_path={SHRUNK_STATS_PATH}")
    emit(f"configured_deg_rate_stats_path={DEG_RATE_STATS_PATH}")
    emit(f"configured_min_stint_length_laps={MIN_STINT_LENGTH_LAPS}")

    corrected = pd.read_csv(CORRECTED_STATS_PATH)
    corrected["selected_r2"] = corrected.apply(_selected_r2, axis=1)
    corrected["is_eligible_fit"] = corrected.apply(_is_eligible_fit, axis=1)

    emit("")
    emit("=" * 100)
    emit("Step 1 - R2 distribution and shrinkage rule")
    emit("=" * 100)

    eligible = corrected[corrected["is_eligible_fit"]].copy()
    selected_r2 = eligible["selected_r2"].dropna()

    q10 = float(selected_r2.quantile(0.10))
    q25 = float(selected_r2.quantile(0.25))
    q50 = float(selected_r2.quantile(0.50))
    cutoff = q25

    emit(f"all_rows={len(corrected)}")
    emit(f"eligible_rows_for_r2_distribution={len(selected_r2)}")
    emit(f"selected_r2_min={float(selected_r2.min()):.9f}")
    emit(f"selected_r2_p10={q10:.9f}")
    emit(f"selected_r2_p25={q25:.9f}")
    emit(f"selected_r2_p50={q50:.9f}")
    emit(f"selected_r2_mean={float(selected_r2.mean()):.9f}")
    emit("chosen_r2_cutoff_basis=25th_percentile")
    emit(f"chosen_r2_cutoff={cutoff:.9f}")
    emit(
        "cutoff_reasoning=The 10th percentile is too permissive and would miss Italian GP 2024 MEDIUM; "
        "the 25th percentile remains distribution-derived while capturing the thin/noisy tail relevant to the exploit."
    )

    shrunk_table, locked_eval_rows = _build_shrunk_table(corrected, cutoff)
    SHRUNK_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    shrunk_table.to_csv(SHRUNK_STATS_PATH, index=False)

    locked = pd.DataFrame(LOCKED_ARCHETYPE_RACES)
    dry_compounds = ["SOFT", "MEDIUM", "HARD"]

    summary_rows = []
    for race in LOCKED_ARCHETYPE_RACES:
        for compound in dry_compounds:
            row = locked_eval_rows[
                (locked_eval_rows["event_name"] == race["event_name"]) &
                (locked_eval_rows["year"] == race["year"]) &
                (locked_eval_rows["TyreCompound"] == compound)
            ]
            if row.empty:
                summary_rows.append(
                    {
                        "EventName": race["event_name"],
                        "Year": int(race["year"]),
                        "Round": int(race["round"]),
                        "TyreCompound": compound,
                        "model": "no_row",
                        "low_sample": None,
                        "selected_r2": np.nan,
                        "deg_rate_pre_shrink": np.nan,
                        "shrink_target_locked_compound_median": np.nan,
                        "deg_rate_final_assigned": np.nan,
                        "is_thin_r2": False,
                        "status": "missing_fit_row",
                    }
                )
                continue

            r = row.iloc[0]
            model = str(r.get("model", ""))
            low_sample = bool(r.get("low_sample", False)) if pd.notna(r.get("low_sample")) else None
            selected = float(r.get("selected_r2")) if pd.notna(r.get("selected_r2")) else np.nan
            thin = bool(r.get("is_thin_r2", False))

            if low_sample is True:
                status = "excluded_low_sample"
            elif model in {"invalid_data", "insufficient", "missing"}:
                status = f"excluded_{model}"
            elif thin:
                status = "below_cutoff_shrunk"
            else:
                status = "eligible_non_thin"

            summary_rows.append(
                {
                    "EventName": race["event_name"],
                    "Year": int(race["year"]),
                    "Round": int(race["round"]),
                    "TyreCompound": compound,
                    "model": model,
                    "low_sample": low_sample,
                    "selected_r2": selected,
                    "deg_rate_pre_shrink": r.get("deg_rate_pre_shrink"),
                    "shrink_target_locked_compound_median": r.get("shrink_target_locked_compound_median"),
                    "deg_rate_final_assigned": r.get("deg_rate_final_assigned"),
                    "is_thin_r2": thin,
                    "status": status,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    thin_df = summary_df[summary_df["status"] == "below_cutoff_shrunk"].copy()

    emit("locked_race_compound_status_table_begin")
    for row in summary_df.sort_values(["EventName", "Year", "TyreCompound"]).to_dict(orient="records"):
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("locked_race_compound_status_table_end")

    emit(f"locked_compounds_below_cutoff_count={len(thin_df)}")

    italy_medium = summary_df[
        (summary_df["EventName"] == "Italian Grand Prix") &
        (summary_df["Year"] == 2024) &
        (summary_df["TyreCompound"] == "MEDIUM")
    ]
    if italy_medium.empty:
        raise RuntimeError("Italian GP 2024 MEDIUM row missing from summary table")

    italy_row = italy_medium.iloc[0]
    italy_sign_flip = bool(pd.notna(italy_row["deg_rate_final_assigned"]) and italy_row["deg_rate_final_assigned"] > 0)
    emit("italian_gp_2024_medium_row=" + json.dumps(italy_row.to_dict(), sort_keys=True, default=str))
    emit(f"italian_gp_2024_medium_shrinkage_flips_positive={italy_sign_flip}")

    emit("")
    emit("=" * 100)
    emit("Step 2 - Formal minimum stint derivation")
    emit("=" * 100)

    spread_table, worst = _derive_min_stint(locked_eval_rows)
    emit("locked_race_spread_table_begin")
    for row in spread_table.sort_values(["EventName", "Year"]).to_dict(orient="records"):
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("locked_race_spread_table_end")

    worst_spread = float(worst["spread_high_minus_low"])
    worst_pit = float(worst["pit_loss_s"])
    worst_breakeven = float(worst["breakeven_laps"])
    recommended_min_stint = int(math.ceil(worst_breakeven))

    emit(
        "worst_case_race_for_spread="
        + json.dumps(
            {
                "EventName": worst["EventName"],
                "Year": int(worst["Year"]),
                "Round": int(worst["Round"]),
                "high_deg_compound": worst["high_deg_compound"],
                "low_deg_compound": worst["low_deg_compound"],
                "spread_high_minus_low": worst_spread,
                "pit_loss_s": worst_pit,
                "breakeven_laps": worst_breakeven,
                "breakeven_laps_ceiling": recommended_min_stint,
            },
            sort_keys=True,
        )
    )
    emit(
        "breakeven_formula=spread*T(L)=pit_loss where T(L)=L(L+1)/2, "
        "so L=(-1+sqrt(1+8*pit_loss/ spread))/2"
    )
    emit(
        f"breakeven_substitution= L=(-1+sqrt(1+8*{worst_pit:.6f}/{worst_spread:.9f}))/2"
    )
    emit(f"recommended_MIN_STINT_LENGTH_LAPS={recommended_min_stint}")

    emit("")
    emit("=" * 100)
    emit("Step 3 - Implementation checks")
    emit("=" * 100)
    emit(f"config_min_stint_matches_derivation={MIN_STINT_LENGTH_LAPS == recommended_min_stint}")
    emit(f"config_deg_stats_path_points_to_shrunk_table={Path(DEG_RATE_STATS_PATH).resolve() == SHRUNK_STATS_PATH.resolve()}")
    emit(f"grid_spacing_used={GRID_SPACING}")
    emit(f"max_stops_used={MAX_STOPS}")

    emit("")
    emit("=" * 100)
    emit("Step 4 - Actual strategy feasibility under constrained enumeration")
    emit("=" * 100)

    feasibility_results = []
    for race in FOUR_RACES:
        emit(f"feasibility_check_race={race['event_name']} year={race['year']} round={race['round']}")
        res = _actual_strategy_feasibility_for_race(race)
        feasibility_results.append(res)
        emit(
            f"actual_strategy_feasibility race={race['event_name']} year={race['year']} "
            f"drivers={res['driver_count']} failures={res['failure_count']}"
        )
        for failure in res["failures"]:
            emit("actual_strategy_failure=" + json.dumps(failure, sort_keys=True, default=str))

    emit("")
    emit("=" * 100)
    emit("Step 5 - Argmax re-run on four races under fixed pipeline")
    emit("=" * 100)

    argmax_results = []
    for race in FOUR_RACES:
        emit(f"argmax_start race={race['event_name']} year={race['year']} round={race['round']}")
        result = _run_argmax_with_progress(race)
        argmax_results.append(result)
        emit("argmax_result=" + json.dumps(result, sort_keys=True, default=str))

    hungary_result = next(r for r in argmax_results if r["race"]["event_name"] == "Hungarian Grand Prix")
    italy_result = next(r for r in argmax_results if r["race"]["event_name"] == "Italian Grand Prix")

    short_stint_gone = all(r["best_first_stint_length"] >= MIN_STINT_LENGTH_LAPS for r in argmax_results)
    italy_exact_winrate_gone = italy_result["best_win_rate"] < 1.0

    emit(f"short_stint_pattern_gone={short_stint_gone}")
    emit(f"italian_exact_1_0_win_rate_pattern_gone={italy_exact_winrate_gone}")
    emit(f"italian_best_win_rate={italy_result['best_win_rate']:.6f}")

    emit("")
    emit("=" * 100)
    emit("Step 6 - Resolution statement")
    emit("=" * 100)

    all_actual_feasible = all(r["failure_count"] == 0 for r in feasibility_results)
    resolved = bool(short_stint_gone and italy_exact_winrate_gone and all_actual_feasible)

    emit(f"all_actual_strategies_feasible_in_checked_races={all_actual_feasible}")
    emit(f"combined_fix_resolution_status={resolved}")
    emit(f"saved_shrunk_table={SHRUNK_STATS_PATH}")


if __name__ == "__main__":
    main()
