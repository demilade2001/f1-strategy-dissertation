from __future__ import annotations

import sys
import time
from math import comb
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation.monte_carlo import (
    build_actual_strategies,
    build_probability_sources,
    run_monte_carlo,
    select_midfield_subject_driver,
)
from source_code.simulation.race_state import load_race_state
from source_code.simulation.strategy import enumerate_feasible_strategies


RACE_YEAR = 2022
RACE_ROUND = 13
RACE_NAME = "Hungarian Grand Prix"
GRID_SPACING = 3
MAX_STOPS_SAMPLE = 2
MAX_STOPS_EXTRAPOLATION = 3
SAMPLE_STRATEGY_COUNT = 20
N_ITERATIONS = 10_000
SEED = 42

STEP2H_BENCHMARK_STRATEGY_COUNT = 73_203
STEP2H_BENCHMARK_SCHEDULE_COUNT = 10_000
STEP2H_BENCHMARK_SECONDS = 86.019160
STEP2H_BENCHMARK_RACE = "Monaco Grand Prix 2023"

STEP4_VALIDATION_OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step4_validation_output.txt"


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _is_wet_race_for_subject(subject_laps: pd.DataFrame) -> bool:
    dry_compounds = {"SOFT", "MEDIUM", "HARD"}
    compounds = {
        str(value).upper()
        for value in subject_laps["TyreCompound"].dropna().astype(str).tolist()
    }
    return any(compound not in dry_compounds for compound in compounds)


def _is_grid_spaced_strategy(strategy: Dict[str, object], candidate_laps: set) -> bool:
    stops = strategy.get("stops", [])
    return all(int(lap) in candidate_laps for lap, _ in stops)


def _grid3_size_for_max_stops(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    max_stops: int,
) -> int:
    n_candidates = len(list(range(1, race_length, GRID_SPACING)))
    dry_compounds = 3
    starting_is_dry = str(starting_compound).upper() in {"SOFT", "MEDIUM", "HARD"}

    total = 0
    for stop_count in range(1, max_stops + 1):
        combos = comb(n_candidates, stop_count)
        if is_wet_race or not starting_is_dry:
            compound_sequences = dry_compounds ** stop_count
        else:
            # Only the all-starting-compound sequence is invalid under dry-race
            # two-compound rule for a fixed pit-lap tuple.
            compound_sequences = (dry_compounds ** stop_count) - 1
        total += combos * compound_sequences
    return int(total)


def main() -> None:
    print_header("Step 1 - Re-verify step 4 output file")
    if not STEP4_VALIDATION_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Missing step 4 output file: {STEP4_VALIDATION_OUTPUT_PATH}")

    file_size = STEP4_VALIDATION_OUTPUT_PATH.stat().st_size
    line_count = sum(1 for _ in STEP4_VALIDATION_OUTPUT_PATH.open("r", encoding="utf-8"))
    print(f"step4_output_path={STEP4_VALIDATION_OUTPUT_PATH}")
    print(f"step4_output_size_bytes={file_size}")
    print(f"step4_output_line_count={line_count}")
    print(f"step4_output_non_empty={(file_size > 0 and line_count > 0)}")
    print("previous_step4_commit_empty=False (no step4 re-commit required)")

    print_header("Step 2 - Generate 20 feasible candidate strategies (grid spacing 3, max_stops 2)")
    state = load_race_state(RACE_YEAR, RACE_ROUND)
    race_laps = state["laps"].copy()

    subject_driver = select_midfield_subject_driver(race_laps)
    strategies_by_driver = build_actual_strategies(race_laps)
    subject_actual = dict(strategies_by_driver[subject_driver])

    race_length = int(pd.to_numeric(race_laps["LapNumber"], errors="coerce").max())
    subject_laps = race_laps[race_laps["Driver"].astype(str) == subject_driver].copy()
    starting_compound = str(subject_actual["starting_compound"])
    is_wet_race = _is_wet_race_for_subject(subject_laps)

    base_space = enumerate_feasible_strategies(
        race_length=race_length,
        starting_compound=starting_compound,
        is_wet_race=is_wet_race,
        max_stops=MAX_STOPS_SAMPLE,
    )

    if hasattr(base_space, "materialize"):
        base_materialized = base_space.materialize(limit=500_000)
    else:
        base_materialized = list(base_space)

    grid_candidate_laps = set(range(1, race_length, GRID_SPACING))
    sampled_strategies = [
        strategy
        for strategy in base_materialized
        if _is_grid_spaced_strategy(strategy, grid_candidate_laps)
    ]

    if len(sampled_strategies) < SAMPLE_STRATEGY_COUNT:
        raise ValueError(
            f"Need at least {SAMPLE_STRATEGY_COUNT} grid-spaced strategies, found {len(sampled_strategies)}"
        )

    sampled_strategies = sampled_strategies[:SAMPLE_STRATEGY_COUNT]

    print(f"race={RACE_NAME} ({RACE_YEAR} R{RACE_ROUND})")
    print(f"subject_driver={subject_driver}")
    print(f"subject_starting_compound={starting_compound}")
    print(f"is_wet_race={is_wet_race}")
    print(f"race_length={race_length}")
    print(f"candidate_lap_count_grid3={len(grid_candidate_laps)}")
    print(f"materialized_feasible_count_max2_no_grid_filter={len(base_materialized)}")
    print(f"feasible_count_max2_with_grid3_filter={len([s for s in base_materialized if _is_grid_spaced_strategy(s, grid_candidate_laps)])}")
    print(f"sampled_strategy_count={len(sampled_strategies)}")
    print("sampled_strategy_first3=")
    for idx, strategy in enumerate(sampled_strategies[:3], start=1):
        print(f"  strategy_{idx}={strategy}")

    print_header("Step 3 - Time 20 run_monte_carlo calls at n_iterations=10000")
    probability_sources = build_probability_sources(state)
    prob_by_lap = probability_sources["merged"]

    per_strategy_seconds: List[float] = []
    per_strategy_mean_points: List[float] = []

    total_start = time.perf_counter()
    for idx, candidate_strategy in enumerate(sampled_strategies, start=1):
        rng = np.random.default_rng(SEED)
        start = time.perf_counter()
        result = run_monte_carlo(
            race_state=state,
            subject_driver=subject_driver,
            subject_strategy=candidate_strategy,
            prob_by_lap=prob_by_lap,
            lambda_=1.0,
            n_iterations=N_ITERATIONS,
            rng=rng,
        )
        elapsed = time.perf_counter() - start
        per_strategy_seconds.append(elapsed)
        per_strategy_mean_points.append(float(result["mean_points"]))
        print(
            f"strategy_{idx:02d}_elapsed_seconds={elapsed:.6f} "
            f"strategy_{idx:02d}_mean_points={result['mean_points']:.6f}"
        )

    total_elapsed = time.perf_counter() - total_start
    mean_per_strategy = total_elapsed / len(sampled_strategies)

    full_hungary_grid3_max3_size = _grid3_size_for_max_stops(
        race_length=race_length,
        starting_compound=starting_compound,
        is_wet_race=is_wet_race,
        max_stops=MAX_STOPS_EXTRAPOLATION,
    )
    extrapolated_full_hungary_seconds = mean_per_strategy * full_hungary_grid3_max3_size

    print(f"total_time_for_20_strategies_seconds={total_elapsed:.6f}")
    print(f"mean_time_per_strategy_seconds={mean_per_strategy:.6f}")
    print(f"full_hungary_grid3_max3_feasible_size={full_hungary_grid3_max3_size}")
    print(f"extrapolated_full_hungary_grid3_max3_seconds={extrapolated_full_hungary_seconds:.6f}")

    print_header("Step 4 - Compare against step 2h vectorized benchmark")
    step2h_seconds_per_strategy = STEP2H_BENCHMARK_SECONDS / STEP2H_BENCHMARK_STRATEGY_COUNT
    ratio = mean_per_strategy / step2h_seconds_per_strategy

    print(f"step2h_benchmark_race={STEP2H_BENCHMARK_RACE}")
    print(f"step2h_benchmark_strategy_count={STEP2H_BENCHMARK_STRATEGY_COUNT}")
    print(f"step2h_benchmark_schedule_count={STEP2H_BENCHMARK_SCHEDULE_COUNT}")
    print(f"step2h_benchmark_wall_clock_seconds={STEP2H_BENCHMARK_SECONDS:.6f}")
    print(f"step2h_seconds_per_strategy={step2h_seconds_per_strategy:.12f}")
    print(f"step4e_seconds_per_strategy={mean_per_strategy:.12f}")
    print(f"seconds_per_strategy_ratio_step4e_div_step2h={ratio:.12f}")

    print_header("Step 5 - Report only (no fix)")
    projected_time_73203 = mean_per_strategy * STEP2H_BENCHMARK_STRATEGY_COUNT
    print(f"projected_time_for_73203_strategies_seconds={projected_time_73203:.6f}")
    print(f"projected_time_for_73203_strategies_hours={projected_time_73203 / 3600.0:.6f}")

    if projected_time_73203 <= STEP2H_BENCHMARK_SECONDS * 2:
        conclusion = (
            "Conclusion: simple per-strategy run_monte_carlo loop appears tractable at locked sizes."
        )
    else:
        conclusion = (
            "Conclusion: simple per-strategy run_monte_carlo loop is intractable at locked sizes; "
            "optimiser.py would need vectorized/accelerated evaluation before build."
        )
    print(conclusion)


if __name__ == "__main__":
    main()
