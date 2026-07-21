from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.monte_carlo import (
    build_actual_strategies,
    build_probability_sources,
    run_monte_carlo,
    select_midfield_subject_driver,
)
from src.simulation.race_model import _build_degradation_lookup, _driver_pit_loss
from src.simulation.race_model_vectorized import (
    evaluate_strategy_batch,
    precompute_rival_times,
    rank_and_score_batch,
)
from src.simulation.race_state import load_race_state
from src.simulation.sc_sampler import sample_caution_schedule
from src.simulation.strategy import enumerate_feasible_strategies

OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step4g_vectorized_correctness_and_runtime_output.txt"

RACE_YEAR = 2022
RACE_ROUND = 13
RACE_NAME = "Hungarian Grand Prix"
GRID_SPACING = 3
MAX_STOPS = 3
N_ITERATIONS = 10_000
SEED = 42
BATCH_SIZE = 2048
NAIVE_SECONDS_PER_STRATEGY = 13.25
MONACO_LOCKED_STRATEGY_COUNT = 73_203


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _build_caution_schedule_matrix(prob_by_lap: Dict[int, float], n_iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    race_length = int(max(prob_by_lap.keys()))
    schedules = np.zeros((n_iterations, race_length), dtype=bool)

    for idx in range(n_iterations):
        schedule = sample_caution_schedule(prob_by_lap, rng)
        for lap in range(1, race_length + 1):
            schedules[idx, lap - 1] = bool(schedule.get(lap, False))
    return schedules


def _is_grid_spaced_strategy(strategy: Dict[str, object], candidate_laps: set) -> bool:
    return all(int(lap) in candidate_laps for lap, _ in strategy.get("stops", []))


def _with_subject_metadata(strategy: Dict[str, object], subject_actual: Dict[str, object]) -> Dict[str, object]:
    enriched = dict(strategy)
    enriched.setdefault("baseline_pace_s", float(subject_actual["baseline_pace_s"]))
    enriched.setdefault("year", int(subject_actual["year"]))
    enriched.setdefault("round", int(subject_actual["round"]))
    enriched.setdefault("event_name", str(subject_actual["event_name"]))
    return enriched


def main() -> None:
    lines: List[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    state = load_race_state(RACE_YEAR, RACE_ROUND)
    race_laps = state["laps"].copy()
    race_laps["sc_active"] = _to_bool(race_laps["sc_active"])
    race_laps["vsc_active"] = _to_bool(race_laps["vsc_active"])

    subject_driver = select_midfield_subject_driver(race_laps)
    strategies_by_driver = build_actual_strategies(race_laps)
    subject_actual = dict(strategies_by_driver[subject_driver])
    subject_laps = race_laps[race_laps["Driver"].astype(str) == subject_driver].copy()

    probability_sources = build_probability_sources(state)
    merged_prob = probability_sources["merged"]
    caution_schedules = _build_caution_schedule_matrix(merged_prob, N_ITERATIONS, SEED)

    emit("=" * 100)
    emit("Step 4g setup")
    emit("=" * 100)
    emit(f"race={RACE_NAME} ({RACE_YEAR} round {RACE_ROUND})")
    emit(f"subject_driver={subject_driver}")
    emit(f"n_iterations={N_ITERATIONS}")
    emit(f"schedule_matrix_shape={tuple(caution_schedules.shape)}")

    race_length = int(pd.to_numeric(race_laps["LapNumber"], errors="coerce").max())
    base_space = enumerate_feasible_strategies(
        race_length=race_length,
        starting_compound=str(subject_actual["starting_compound"]),
        is_wet_race=False,
        max_stops=MAX_STOPS,
    )
    if hasattr(base_space, "materialize"):
        base_materialized = base_space.materialize(limit=2_000_000)
    else:
        base_materialized = list(base_space)

    grid_candidate_laps = set(range(1, race_length, GRID_SPACING))
    grid3_strategies = [s for s in base_materialized if _is_grid_spaced_strategy(s, grid_candidate_laps)]

    emit(f"full_feasible_unfiltered_max_stops3={len(base_materialized)}")
    emit(f"full_feasible_grid3_max_stops3={len(grid3_strategies)}")

    correctness_candidates = [
        _with_subject_metadata(subject_actual, subject_actual),
        *[_with_subject_metadata(s, subject_actual) for s in grid3_strategies[:4]],
    ]

    emit("")
    emit("=" * 100)
    emit("Step 4 - Correctness validation against trusted deterministic path")
    emit("=" * 100)

    deterministic_rows: List[Dict[str, float]] = []
    for idx, candidate in enumerate(correctness_candidates, start=1):
        deterministic = run_monte_carlo(
            race_state=state,
            subject_driver=subject_driver,
            subject_strategy=candidate,
            prob_by_lap=merged_prob,
            lambda_=1.0,
            n_iterations=N_ITERATIONS,
            rng=np.random.default_rng(SEED),
        )
        deterministic_rows.append(
            {
                "strategy_idx": idx,
                "det_mean": float(deterministic["mean_points"]),
                "det_std": float(deterministic["std_points"]),
            }
        )

    degradation_lookup = _build_degradation_lookup(state.get("deg_stats"))
    subject_pit_loss_s = float(_driver_pit_loss(subject_laps))
    rival_times = precompute_rival_times(
        race_state=state,
        subject_driver=subject_driver,
        caution_schedules=caution_schedules,
        lambda_=1.0,
    )

    subject_times_batch = evaluate_strategy_batch(
        subject_strategies=correctness_candidates,
        caution_schedules=caution_schedules,
        degradation_lookup=degradation_lookup,
        lambda_=1.0,
        baseline_pace_s=float(subject_actual["baseline_pace_s"]),
        pit_loss_s=subject_pit_loss_s,
        caution_pit_loss_s=5.0,
        caution_pace_ratio=1.3162357008284764,
    )
    batch_scores = rank_and_score_batch(subject_times_batch, rival_times)

    max_abs_diff = 0.0
    for row in deterministic_rows:
        i = int(row["strategy_idx"]) - 1
        vec_mean = float(batch_scores["mean_points"][i])
        vec_std = float(batch_scores["std_points"][i])
        diff = abs(row["det_mean"] - vec_mean)
        max_abs_diff = max(max_abs_diff, diff)
        emit(
            f"strategy_{row['strategy_idx']}: "
            f"det_mean={row['det_mean']:.6f}, det_std={row['det_std']:.6f}, "
            f"vec_mean={vec_mean:.6f}, vec_std={vec_std:.6f}, "
            f"abs_diff_mean={diff:.6f}"
        )
    emit(f"max_abs_diff_mean_points_across_5={max_abs_diff:.6f}")

    emit("")
    emit("=" * 100)
    emit("Step 5 - Real runtime benchmark (full Hungary grid3 max_stops3)")
    emit("=" * 100)

    full_strategies = [_with_subject_metadata(s, subject_actual) for s in grid3_strategies]

    runtime_start = time.perf_counter()
    mean_points_full = np.zeros(len(full_strategies), dtype=np.float32)
    std_points_full = np.zeros(len(full_strategies), dtype=np.float32)

    for start in range(0, len(full_strategies), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(full_strategies))
        batch = full_strategies[start:end]
        batch_times = evaluate_strategy_batch(
            subject_strategies=batch,
            caution_schedules=caution_schedules,
            degradation_lookup=degradation_lookup,
            lambda_=1.0,
            baseline_pace_s=float(subject_actual["baseline_pace_s"]),
            pit_loss_s=subject_pit_loss_s,
            caution_pit_loss_s=5.0,
            caution_pace_ratio=1.3162357008284764,
        )
        scored = rank_and_score_batch(batch_times, rival_times)
        mean_points_full[start:end] = scored["mean_points"]
        std_points_full[start:end] = scored["std_points"]

    vectorized_seconds = time.perf_counter() - runtime_start

    naive_extrapolated_seconds = NAIVE_SECONDS_PER_STRATEGY * len(full_strategies)
    speedup = naive_extrapolated_seconds / vectorized_seconds if vectorized_seconds > 0 else float("inf")
    sec_per_strategy = vectorized_seconds / len(full_strategies)
    monaco_projected_seconds = sec_per_strategy * MONACO_LOCKED_STRATEGY_COUNT

    emit(f"full_hungary_grid3_max_stops3_strategy_count={len(full_strategies)}")
    emit(f"vectorized_wall_clock_seconds={vectorized_seconds:.6f}")
    emit(f"naive_extrapolated_seconds_using_13.25={naive_extrapolated_seconds:.6f}")
    emit(f"speedup_vs_naive_extrapolated={speedup:.6f}")
    emit(f"vectorized_seconds_per_strategy={sec_per_strategy:.12f}")
    emit(f"projected_monaco_73203_seconds={monaco_projected_seconds:.6f}")
    emit(f"projected_monaco_73203_hours={monaco_projected_seconds / 3600.0:.6f}")

    if monaco_projected_seconds <= 600.0:
        emit("tractability_statement=Largest locked race looks tractable (<=10 minutes projected).")
    elif monaco_projected_seconds <= 3600.0:
        emit("tractability_statement=Largest locked race likely tractable but still expensive (>10 minutes, <=1 hour projected).")
    else:
        emit("tractability_statement=Largest locked race still appears intractable (>1 hour projected).")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
