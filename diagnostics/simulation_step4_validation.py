from __future__ import annotations

import time
from pathlib import Path
import sys
from typing import List

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
from src.simulation.race_model import simulate_race_with_driver_laps
from src.simulation.race_state import load_race_state
from src.simulation.sc_sampler import sample_caution_schedule

OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step4_validation_output.txt"
RACE_YEAR = 2022
RACE_ROUND = 13
RACE_NAME = "Hungarian Grand Prix"


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _historical_order_last_valid_position(race_laps: pd.DataFrame) -> List[str]:
    rows = []
    for driver, driver_laps in race_laps.groupby("Driver"):
        valid = driver_laps[driver_laps["Position"].notna()].copy()
        if valid.empty:
            continue
        pick = valid.sort_values("LapNumber").iloc[-1]
        rows.append(
            {
                "Driver": str(driver),
                "LapNumber": int(pick["LapNumber"]),
                "Position": int(pick["Position"]),
            }
        )

    final_df = pd.DataFrame(rows)
    final_df = final_df.sort_values(["Position", "LapNumber", "Driver"], ascending=[True, False, True])
    return final_df["Driver"].tolist()


def _format_result_block(label: str, result: dict) -> List[str]:
    return [
        f"{label}_mean_points={result['mean_points']:.12f}",
        f"{label}_std_points={result['std_points']:.12f}",
        f"{label}_mean_position={result['mean_position']:.12f}",
        f"{label}_n_iterations={result['n_iterations']}",
    ]


def main() -> None:
    lines: List[str] = []

    state = load_race_state(RACE_YEAR, RACE_ROUND)
    race_laps = state["laps"].copy()
    race_laps["sc_active"] = _to_bool(race_laps["sc_active"])
    race_laps["vsc_active"] = _to_bool(race_laps["vsc_active"])

    strategies_by_driver = build_actual_strategies(race_laps)
    subject_driver = select_midfield_subject_driver(race_laps)
    subject_strategy = strategies_by_driver[subject_driver]
    probability_sources = build_probability_sources(state)
    driver_laps_by_driver = {str(driver): driver_laps.copy() for driver, driver_laps in race_laps.groupby("Driver")}

    lines.append("Step 0 - Validation setup")
    lines.append(f"race={RACE_NAME} ({RACE_YEAR} round {RACE_ROUND})")
    lines.append(f"subject_driver={subject_driver}")
    lines.append(f"subject_team={race_laps.loc[race_laps['Driver'].astype(str) == subject_driver, 'Team'].dropna().iloc[-1]}")
    lines.append(f"drivers_in_race={race_laps['Driver'].nunique()}")
    lines.append(f"circuit_sc_rate={float(state['circuit_sc_rate']):.12f}")
    lines.append(f"circuit_vsc_rate={float(state['circuit_vsc_rate']):.12f}")

    # Step 2: direct evidence that rivals are re-simulated rather than replayed.
    lines.append("\nStep 1 - Rival re-simulation evidence")
    rng = np.random.default_rng(42)
    rival_driver = next(driver for driver in sorted(strategies_by_driver) if driver != subject_driver)
    rival_times = []
    for draw_idx in range(1, 4):
        caution_schedule = sample_caution_schedule(probability_sources["merged"], rng)
        result = simulate_race_with_driver_laps(
            state,
            strategies_by_driver,
            caution_schedule,
            lambda_=1.0,
            driver_laps_by_driver=driver_laps_by_driver,
            collect_trace=False,
        )
        rival_time = float(result["driver_results"][rival_driver]["race_time_s"])
        rival_times.append(rival_time)
        lines.append(f"draw_{draw_idx}_rival={rival_driver}_total_race_time_s={rival_time:.12f}")

    if len(set(round(value, 9) for value in rival_times)) == 1:
        raise RuntimeError(
            f"BUG: rival {rival_driver} total race times were identical across three caution-schedule draws; rivals are being replayed, not re-simulated."
        )
    lines.append("rival_re_simulation_check=passed")

    # Step 3: convergence check at 100, 1,000, and 10,000 iterations.
    lines.append("\nStep 2 - Convergence check on Hungary 2022")
    convergence_results = {}
    runtime_10k_seconds = None
    for n_iterations in (100, 1_000, 10_000):
        rng = np.random.default_rng(42)
        start = time.perf_counter()
        result = run_monte_carlo(
            race_state=state,
            subject_driver=subject_driver,
            subject_strategy=subject_strategy,
            prob_by_lap=probability_sources["merged"],
            lambda_=1.0,
            n_iterations=n_iterations,
            rng=rng,
        )
        elapsed = time.perf_counter() - start
        convergence_results[n_iterations] = result
        lines.extend(_format_result_block(f"n_{n_iterations}", result))
        lines.append(f"n_{n_iterations}_wall_clock_seconds={elapsed:.12f}")
        if n_iterations == 10_000:
            runtime_10k_seconds = elapsed

    diff_1k_10k = abs(convergence_results[1_000]["mean_points"] - convergence_results[10_000]["mean_points"])
    rel_diff_1k_10k = diff_1k_10k / abs(convergence_results[10_000]["mean_points"])
    lines.append(f"mean_points_abs_diff_1000_vs_10000={diff_1k_10k:.12f}")
    lines.append(f"mean_points_rel_diff_1000_vs_10000={rel_diff_1k_10k:.12%}")
    lines.append(
        f"mean_stabilized_1000_vs_10000={rel_diff_1k_10k < 0.01}"
    )

    # Step 4: seed stability at 10,000 iterations.
    lines.append("\nStep 3 - Seed stability at 10,000 iterations")
    seed_means = []
    for seed in (42, 43, 44, 45, 46):
        rng = np.random.default_rng(seed)
        result = run_monte_carlo(
            race_state=state,
            subject_driver=subject_driver,
            subject_strategy=subject_strategy,
            prob_by_lap=probability_sources["merged"],
            lambda_=1.0,
            n_iterations=10_000,
            rng=rng,
        )
        seed_means.append(float(result["mean_points"]))
        lines.append(f"seed_{seed}_mean_points={result['mean_points']:.12f}")
    lines.append(f"seed_mean_std={float(np.std(seed_means, ddof=1)):.12f}")

    # Step 5: runtime check.
    lines.append("\nStep 4 - Runtime check for a single-strategy 10,000-iteration run")
    lines.append(f"single_strategy_10k_wall_clock_seconds={runtime_10k_seconds:.12f}")
    lines.append("comparison_to_step2h_full_sweep=This should be dramatically faster than the ~86-second full-strategy sweep because the Monte Carlo loop evaluates one fixed strategy against shared caution schedules, not tens of thousands of candidate strategies.")

    # Step 6: probability source comparison.
    lines.append("\nStep 5 - Probability source distinguishability")
    rng = np.random.default_rng(42)
    merged_result = run_monte_carlo(
        race_state=state,
        subject_driver=subject_driver,
        subject_strategy=subject_strategy,
        prob_by_lap=probability_sources["merged"],
        lambda_=1.0,
        n_iterations=10_000,
        rng=rng,
    )
    rng = np.random.default_rng(42)
    flat_result = run_monte_carlo(
        race_state=state,
        subject_driver=subject_driver,
        subject_strategy=subject_strategy,
        prob_by_lap=probability_sources["flat"],
        lambda_=1.0,
        n_iterations=10_000,
        rng=rng,
    )
    lines.append(f"merged_prob_mean_points={merged_result['mean_points']:.12f}")
    lines.append(f"flat_prob_mean_points={flat_result['mean_points']:.12f}")
    lines.append(f"mean_points_difference_merged_minus_flat={merged_result['mean_points'] - flat_result['mean_points']:.12f}")

    historical_order = _historical_order_last_valid_position(race_laps)
    lines.append("\nStep 6 - Ground truth check")
    lines.append(f"historical_order_first5={' > '.join(historical_order[:5])}")
    lines.append(f"historical_order_last5={' > '.join(historical_order[-5:])}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
