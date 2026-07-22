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

from src.simulation.config import GRID_SPACING, LOCKED_ARCHETYPE_RACES, MAX_STOPS
from src.simulation.monte_carlo import build_actual_strategies, select_midfield_subject_driver
from src.simulation.optimiser import (
    ARGMAX_BATCH_SIZE,
    _build_caution_schedule_matrix,
    _precompute_subject_inputs,
    _resolve_probability_by_source,
)
from src.simulation.race_model_vectorized import evaluate_strategy_batch, rank_and_score_batch
from src.simulation.race_state import load_race_state
from src.simulation.strategy import enumerate_feasible_strategies, is_strategy_feasible

RAW_DEG_PATH = ROOT / "data" / "diagnostics" / "deg_rate_full_peryear_stats.csv"
SHRUNK_DEG_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_shrunk_full_peryear_stats.csv"

RACES_MAIN = [
    {"event_name": "Hungarian Grand Prix", "year": 2022, "round": 13},
    {"event_name": "Italian Grand Prix", "year": 2024, "round": 16},
]
RACES_FEAS = [
    {"event_name": "Hungarian Grand Prix", "year": 2022, "round": 13},
    {"event_name": "Italian Grand Prix", "year": 2024, "round": 16},
    {"event_name": "Bahrain Grand Prix", "year": 2024, "round": 1},
    {"event_name": "Azerbaijan Grand Prix", "year": 2024, "round": 17},
]

PROB_SOURCE = "lap_level"
N_ITERATIONS = 10_000
LAMBDA_VALUE = 1.0
SEED = 42


def emit(text: str = "") -> None:
    print(text, flush=True)


def _strategy_key(strategy: Mapping[str, object]) -> Tuple[str, Tuple[Tuple[int, str], ...]]:
    return (
        str(strategy["starting_compound"]),
        tuple((int(l), str(c)) for l, c in strategy.get("stops", [])),
    )


def _load_race_state_with_deg(race: Mapping[str, object], deg_path: Path) -> dict:
    state = load_race_state(int(race["year"]), int(race["round"]))
    event_name = str(state["laps"]["EventName"].dropna().iloc[0])
    deg_df = pd.read_csv(deg_path)
    state["deg_stats"] = deg_df[
        (deg_df["Year"] == int(race["year"])) &
        (deg_df["EventName"] == event_name)
    ].copy()
    return state


def _enumerate_subject_strategies(race_state: dict, min_stint_length_laps: int) -> dict:
    laps = race_state["laps"].copy()
    subject_driver = select_midfield_subject_driver(laps)
    actual_strategies = build_actual_strategies(laps)
    subject_actual = dict(actual_strategies[subject_driver])

    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    base_space = enumerate_feasible_strategies(
        race_length=race_length,
        starting_compound=str(subject_actual["starting_compound"]),
        is_wet_race=False,
        max_stops=MAX_STOPS,
        min_stint_length_laps=int(min_stint_length_laps),
    )
    if hasattr(base_space, "materialize"):
        materialized = base_space.materialize(limit=2_000_000)
    else:
        materialized = list(base_space)

    grid_candidate_laps = set(range(1, race_length, GRID_SPACING))
    filtered = [
        s for s in materialized
        if all(int(lap) in grid_candidate_laps for lap, _ in s.get("stops", []))
    ]

    enriched = []
    for s in filtered:
        row = dict(s)
        row.setdefault("baseline_pace_s", float(subject_actual["baseline_pace_s"]))
        row.setdefault("year", int(subject_actual["year"]))
        row.setdefault("round", int(subject_actual["round"]))
        row.setdefault("event_name", str(subject_actual["event_name"]))
        enriched.append(row)

    return {
        "subject_driver": subject_driver,
        "subject_actual": subject_actual,
        "race_length": race_length,
        "strategies": enriched,
    }


def _prepare_race_context(race_state: dict, enum_ctx: dict) -> dict:
    prob_by_lap = _resolve_probability_by_source(race_state, PROB_SOURCE)
    caution_schedules = _build_caution_schedule_matrix(prob_by_lap, N_ITERATIONS, np.random.default_rng(SEED))

    precomputed = _precompute_subject_inputs(
        race_state=race_state,
        subject_driver=enum_ctx["subject_driver"],
        caution_schedules=caution_schedules,
        lambda_=LAMBDA_VALUE,
    )

    return {
        "caution_schedules": caution_schedules,
        "precomputed": precomputed,
    }


def _score_argmax(race_state: dict, enum_ctx: dict, prepared_ctx: dict) -> dict:

    strategies: List[dict] = enum_ctx["strategies"]
    if not strategies:
        raise ValueError("No strategies in enumerated strategy set")

    mean_points = np.zeros(len(strategies), dtype=np.float64)
    std_points = np.zeros(len(strategies), dtype=np.float64)
    mean_rank = np.zeros(len(strategies), dtype=np.float64)

    n_batches = (len(strategies) + ARGMAX_BATCH_SIZE - 1) // ARGMAX_BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(strategies), ARGMAX_BATCH_SIZE), start=1):
        end = min(start + ARGMAX_BATCH_SIZE, len(strategies))
        batch = strategies[start:end]
        subject_times = evaluate_strategy_batch(
            subject_strategies=batch,
            caution_schedules=prepared_ctx["caution_schedules"],
            degradation_lookup=prepared_ctx["precomputed"]["degradation_lookup"],
            lambda_=LAMBDA_VALUE,
            baseline_pace_s=float(enum_ctx["subject_actual"]["baseline_pace_s"]),
            pit_loss_s=float(prepared_ctx["precomputed"]["pit_loss_s"]),
            caution_pit_loss_s=5.0,
            caution_pace_ratio=1.3162357008284764,
        )
        scored = rank_and_score_batch(subject_times, prepared_ctx["precomputed"]["rival_times"])
        mean_points[start:end] = np.asarray(scored["mean_points"], dtype=np.float64)
        std_points[start:end] = np.asarray(scored["std_points"], dtype=np.float64)
        mean_rank[start:end] = np.asarray(scored["mean_rank"], dtype=np.float64)
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == n_batches:
            emit(
                f"batch_progress race={enum_ctx['subject_actual']['event_name']} "
                f"floor_eval_batch={batch_idx}/{n_batches} scored={end}/{len(strategies)}"
            )

    first_stints = np.array([
        int(s["stops"][0][0]) if s.get("stops") else int(enum_ctx["race_length"])
        for s in strategies
    ])
    dist = pd.Series(first_stints).value_counts().sort_index()

    best_idx = int(np.argmax(mean_points))
    best_strategy = strategies[best_idx]
    best_first_stint = int(first_stints[best_idx])

    top_k = min(100, len(strategies))
    top_idx = np.argsort(-mean_points)[:top_k]
    top_short_count = int((first_stints[top_idx] <= 3).sum())
    top_short_frac = float(top_short_count / top_k)

    actual_key = _strategy_key(enum_ctx["subject_actual"])
    feasible_keys = {_strategy_key(s) for s in strategies}
    actual_in_set = actual_key in feasible_keys

    # Win rate for argmax strategy.
    best_times = evaluate_strategy_batch(
        subject_strategies=[best_strategy],
        caution_schedules=prepared_ctx["caution_schedules"],
        degradation_lookup=prepared_ctx["precomputed"]["degradation_lookup"],
        lambda_=LAMBDA_VALUE,
        baseline_pace_s=float(enum_ctx["subject_actual"]["baseline_pace_s"]),
        pit_loss_s=float(prepared_ctx["precomputed"]["pit_loss_s"]),
        caution_pit_loss_s=5.0,
        caution_pace_ratio=1.3162357008284764,
    )
    best_scored = rank_and_score_batch(best_times, prepared_ctx["precomputed"]["rival_times"])
    rank_matrix = np.asarray(best_scored["rank_matrix"][0], dtype=np.int16)
    best_win_rate = float(np.mean(rank_matrix == 1))

    return {
        "strategy_count": int(len(strategies)),
        "best_strategy": best_strategy,
        "best_mean_points": float(mean_points[best_idx]),
        "best_std_points": float(std_points[best_idx]),
        "best_mean_rank": float(mean_rank[best_idx]),
        "best_win_rate": best_win_rate,
        "best_first_stint_length": best_first_stint,
        "first_stint_distribution": [
            {
                "first_stint_length": int(k),
                "count": int(v),
                "share": float(v / len(first_stints)),
            }
            for k, v in dist.items()
        ],
        "top_k": int(top_k),
        "top_short_stint_leq3_count": top_short_count,
        "top_short_stint_leq3_fraction": top_short_frac,
        "actual_subject_strategy_in_feasible_set": bool(actual_in_set),
    }


def _evaluate_configuration(
    race: Mapping[str, object],
    deg_path: Path,
    floor: int,
    context_cache: Dict[Tuple[str, int, int, str], dict],
) -> dict:
    cache_key = (str(race["event_name"]), int(race["year"]), int(race["round"]), str(deg_path))
    cached = context_cache.get(cache_key)
    if cached is None:
        state = _load_race_state_with_deg(race, deg_path)
        enum_ctx_floor1 = _enumerate_subject_strategies(state, min_stint_length_laps=1)
        prepared = _prepare_race_context(state, enum_ctx_floor1)
        cached = {
            "race_state": state,
            "prepared": prepared,
        }
        context_cache[cache_key] = cached
    else:
        state = cached["race_state"]

    enum_ctx = _enumerate_subject_strategies(state, min_stint_length_laps=floor)
    scored = _score_argmax(state, enum_ctx, cached["prepared"])
    return {
        "race": dict(race),
        "deg_path": str(deg_path),
        "min_stint_length_laps": int(floor),
        "subject_driver": str(enum_ctx["subject_driver"]),
        **scored,
    }


def _actual_feasibility_rate(race: Mapping[str, object], floor: int) -> dict:
    state = _load_race_state_with_deg(race, SHRUNK_DEG_PATH)
    laps = state["laps"].copy()
    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    actual = build_actual_strategies(laps)

    feasible = 0
    failures = []
    for driver, strategy in sorted(actual.items()):
        ok = is_strategy_feasible(
            race_length=race_length,
            starting_compound=str(strategy["starting_compound"]),
            is_wet_race=False,
            stops=[(int(l), str(c)) for l, c in strategy.get("stops", [])],
            dry_compounds=["SOFT", "MEDIUM", "HARD"],
            max_stops=MAX_STOPS,
            min_stint_length_laps=int(floor),
        )
        if ok:
            feasible += 1
        else:
            failures.append(driver)

    total = len(actual)
    return {
        "race": dict(race),
        "floor": int(floor),
        "feasible_drivers": int(feasible),
        "total_drivers": int(total),
        "feasibility_rate": float(feasible / total) if total else np.nan,
        "failed_drivers": failures,
    }


def _winner_web_note() -> dict:
    # Collected from public web checks before this run; persisted in report for traceability.
    return {
        "race": "Italian Grand Prix 2024",
        "winner": "Charles Leclerc",
        "strategy_note": "One-stop",
        "source_type": "web_reference",
    }


def main() -> None:
    emit("=" * 100)
    emit("Step 5e - Isolate shrinkage vs floor")
    emit("=" * 100)
    emit(f"raw_deg_path={RAW_DEG_PATH}")
    emit(f"shrunk_deg_path={SHRUNK_DEG_PATH}")

    emit("")
    emit("=" * 100)
    emit("Step 1 - Hungary/Italy with shrinkage and no floor (MIN_STINT=1)")
    emit("=" * 100)

    step1_results = {}
    context_cache: Dict[Tuple[str, int, int, str], dict] = {}
    for race in RACES_MAIN:
        emit(f"running_step1 race={race['event_name']} year={race['year']}")
        res = _evaluate_configuration(race, SHRUNK_DEG_PATH, floor=1, context_cache=context_cache)
        step1_results[(race["event_name"], race["year"])] = res
        emit("step1_result_core=" + json.dumps({
            "race": race,
            "strategy_count": res["strategy_count"],
            "best_strategy": res["best_strategy"],
            "best_first_stint_length": res["best_first_stint_length"],
            "best_mean_points": res["best_mean_points"],
            "best_win_rate": res["best_win_rate"],
            "top_short_stint_leq3_fraction": res["top_short_stint_leq3_fraction"],
            "actual_subject_strategy_in_feasible_set": res["actual_subject_strategy_in_feasible_set"],
        }, sort_keys=True))
        emit("first_stint_distribution_begin")
        for row in res["first_stint_distribution"]:
            emit(json.dumps(row, sort_keys=True))
        emit("first_stint_distribution_end")

    emit("")
    emit("=" * 100)
    emit("Step 2 - Side-by-side comparison of configurations (a), (b), (c)")
    emit("=" * 100)

    labels = {
        "a_raw_no_floor": (RAW_DEG_PATH, 1),
        "b_shrunk_no_floor": (SHRUNK_DEG_PATH, 1),
        "c_shrunk_floor19": (SHRUNK_DEG_PATH, 19),
    }

    comparison_rows = []
    for race in RACES_MAIN:
        for label, (path, floor) in labels.items():
            emit(f"running_step2 race={race['event_name']} config={label}")
            res = _evaluate_configuration(race, path, floor=floor, context_cache=context_cache)
            top_metric = (
                float(res["top_short_stint_leq3_fraction"])
                if res["strategy_count"] >= 100
                else None
            )
            comparison_rows.append(
                {
                    "race": race["event_name"],
                    "year": int(race["year"]),
                    "config": label,
                    "strategy_count": int(res["strategy_count"]),
                    "argmax_first_stint_length": int(res["best_first_stint_length"]),
                    "top100_short_stint_fraction": top_metric,
                    "top_k_used": int(res["top_k"]),
                    "italy_win_rate": float(res["best_win_rate"]) if race["event_name"] == "Italian Grand Prix" else None,
                    "italy_mean_points": float(res["best_mean_points"]) if race["event_name"] == "Italian Grand Prix" else None,
                    "actual_subject_strategy_in_feasible_set": bool(res["actual_subject_strategy_in_feasible_set"]),
                }
            )

    emit("step2_comparison_table_begin")
    for row in comparison_rows:
        emit(json.dumps(row, sort_keys=True))
    emit("step2_comparison_table_end")

    emit("")
    emit("=" * 100)
    emit("Step 3 - Minimal floor candidates (2 and 5)")
    emit("=" * 100)

    floor_candidates = [2, 5]
    floor_race_rows = []
    feas_rows = []

    for floor in floor_candidates:
        for race in RACES_MAIN:
            emit(f"running_step3 race={race['event_name']} floor={floor}")
            res = _evaluate_configuration(race, SHRUNK_DEG_PATH, floor=floor, context_cache=context_cache)
            floor_race_rows.append(
                {
                    "race": race["event_name"],
                    "year": int(race["year"]),
                    "floor": int(floor),
                    "strategy_count": int(res["strategy_count"]),
                    "argmax_strategy": res["best_strategy"],
                    "argmax_first_stint_length": int(res["best_first_stint_length"]),
                    "top100_short_stint_fraction": float(res["top_short_stint_leq3_fraction"]) if res["strategy_count"] >= 100 else None,
                    "top_k_used": int(res["top_k"]),
                    "italy_win_rate": float(res["best_win_rate"]) if race["event_name"] == "Italian Grand Prix" else None,
                    "italy_mean_points": float(res["best_mean_points"]) if race["event_name"] == "Italian Grand Prix" else None,
                    "actual_subject_strategy_in_feasible_set": bool(res["actual_subject_strategy_in_feasible_set"]),
                }
            )

        for race in RACES_FEAS:
            feas = _actual_feasibility_rate(race, floor=floor)
            feas_rows.append(feas)

    emit("step3_floor_race_table_begin")
    for row in floor_race_rows:
        emit(json.dumps(row, sort_keys=True))
    emit("step3_floor_race_table_end")

    emit("step3_feasibility_rates_begin")
    for row in feas_rows:
        emit(json.dumps(row, sort_keys=True))
    emit("step3_feasibility_rates_end")

    emit("")
    emit("=" * 100)
    emit("Step 4 - Evidence-based recommendation")
    emit("=" * 100)

    winner_note = _winner_web_note()
    emit("italy_2024_real_world_note=" + json.dumps(winner_note, sort_keys=True))

    # Extract key comparisons for recommendation.
    step2_df = pd.DataFrame(comparison_rows)

    def row_for(race_name: str, config_name: str) -> pd.Series:
        return step2_df[(step2_df["race"] == race_name) & (step2_df["config"] == config_name)].iloc[0]

    italy_b = row_for("Italian Grand Prix", "b_shrunk_no_floor")
    italy_c = row_for("Italian Grand Prix", "c_shrunk_floor19")
    hungary_b = row_for("Hungarian Grand Prix", "b_shrunk_no_floor")
    hungary_c = row_for("Hungarian Grand Prix", "c_shrunk_floor19")

    feas_df = pd.DataFrame(feas_rows)
    floor_feas_summary = (
        feas_df.groupby("floor")["feasibility_rate"]
        .agg(["mean", "min", "max"]) 
        .reset_index()
    )

    emit("floor_feasibility_summary_begin")
    for row in floor_feas_summary.to_dict(orient="records"):
        emit(json.dumps(row, sort_keys=True))
    emit("floor_feasibility_summary_end")

    recommendation = {
        "criterion_feasibility_threshold": 0.80,
        "b_shrunk_no_floor": {
            "hungary_top100_short_frac": float(hungary_b["top100_short_stint_fraction"]),
            "italy_win_rate": float(italy_b["italy_win_rate"]),
        },
        "c_shrunk_floor19": {
            "hungary_top100_short_frac": float(hungary_c["top100_short_stint_fraction"]),
            "italy_win_rate": float(italy_c["italy_win_rate"]),
            "strategy_count_hungary": int(hungary_c["strategy_count"]),
            "strategy_count_italy": int(italy_c["strategy_count"]),
        },
    }

    emit("recommendation_evidence=" + json.dumps(recommendation, sort_keys=True))

    # Determine if any tested configuration satisfies both criteria.
    # We treat exploit suppression as: Hungary top100 short <= 0.10 and Italy win_rate <= 0.80.
    # Feasibility criterion from prompt: average and minimum should stay above threshold across 4 races.
    threshold = 0.80
    config_ok = []

    # b: no floor means actual-strategy feasibility not constrained by floor; still subject strategy inclusion is in table.
    b_exploit_ok = (float(hungary_b["top100_short_stint_fraction"]) <= 0.10) and (float(italy_b["italy_win_rate"]) <= 0.80)
    config_ok.append({"config": "b_shrunk_no_floor", "exploit_ok": b_exploit_ok, "feasibility_ok": True})

    for floor in floor_candidates + [19]:
        eval_df = feas_df[feas_df["floor"] == floor]
        if eval_df.empty:
            continue
        feas_ok = bool((eval_df["feasibility_rate"] >= threshold).all())
        if floor == 19:
            expl_ok = (float(hungary_c["top100_short_stint_fraction"]) <= 0.10) and (float(italy_c["italy_win_rate"]) <= 0.80)
            name = "c_shrunk_floor19"
        else:
            floor_h = pd.DataFrame(floor_race_rows)
            h_row = floor_h[(floor_h["race"] == "Hungarian Grand Prix") & (floor_h["floor"] == floor)].iloc[0]
            i_row = floor_h[(floor_h["race"] == "Italian Grand Prix") & (floor_h["floor"] == floor)].iloc[0]
            expl_ok = (
                (h_row["top100_short_stint_fraction"] is not None and float(h_row["top100_short_stint_fraction"]) <= 0.10)
                and (i_row["italy_win_rate"] is not None and float(i_row["italy_win_rate"]) <= 0.80)
            )
            name = f"shrunk_floor{floor}"
        config_ok.append({"config": name, "exploit_ok": bool(expl_ok), "feasibility_ok": bool(feas_ok)})

    emit("configuration_pass_matrix_begin")
    for row in config_ok:
        emit(json.dumps(row, sort_keys=True))
    emit("configuration_pass_matrix_end")

    any_both = any(row["exploit_ok"] and row["feasibility_ok"] for row in config_ok)
    emit(f"any_tested_configuration_meets_both_criteria={any_both}")
    if not any_both:
        emit("recommendation_plain=No tested configuration achieves both strong exploit suppression and >=80% feasibility across all four races.")
    else:
        winners = [row["config"] for row in config_ok if row["exploit_ok"] and row["feasibility_ok"]]
        emit("recommendation_plain=Configurations meeting both criteria: " + ", ".join(winners))


if __name__ == "__main__":
    main()
