from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CAUTION_PACE_RATIO, CAUTION_PIT_LOSS_S
from .monte_carlo import build_actual_strategies
from .race_model import (
    _build_degradation_lookup,
    _driver_median_green_pace,
    _driver_pit_loss,
    _lookup_degradation_rate,
    _strategy_starting_compound,
    _strategy_stops,
    simulate_car_race,
)


POINTS_LOOKUP = np.array([0, 25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=np.float32)


def _validate_caution_schedules(caution_schedules: np.ndarray) -> np.ndarray:
    arr = np.asarray(caution_schedules)
    if arr.ndim != 2:
        raise ValueError("caution_schedules must be a 2D array (n_iterations, race_length)")
    return arr.astype(bool)


def _matrix_row_to_schedule_map(schedule_row: np.ndarray) -> Dict[int, bool]:
    return {lap_idx + 1: bool(flag) for lap_idx, flag in enumerate(schedule_row)}


def precompute_rival_times(
    race_state: dict,
    subject_driver: str,
    caution_schedules: np.ndarray,
    lambda_: float,
) -> np.ndarray:
    """Simulate each non-subject rival once per schedule under full package physics.

    Returns
    -------
    np.ndarray
        Array of shape (n_rivals, n_iterations) with rival race times in seconds.
    """

    schedules = _validate_caution_schedules(caution_schedules)
    laps = race_state["laps"].copy()
    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    if schedules.shape[1] != race_length:
        raise ValueError(
            f"caution_schedules race_length={schedules.shape[1]} does not match race length {race_length}"
        )

    actual_strategies = build_actual_strategies(laps)
    if subject_driver not in actual_strategies:
        raise ValueError(f"subject_driver {subject_driver} not found in race strategies")

    driver_laps_by_driver = {
        str(driver): driver_laps.copy()
        for driver, driver_laps in laps.groupby("Driver")
    }
    degradation_lookup = _build_degradation_lookup(race_state.get("deg_stats"))

    event_name = str(laps["EventName"].dropna().iloc[0])
    year = int(pd.to_numeric(laps["Year"], errors="coerce").dropna().iloc[0])
    round_num = int(pd.to_numeric(laps["Round"], errors="coerce").dropna().iloc[0])
    race_baseline_pace = _driver_median_green_pace(laps)

    rival_drivers = sorted([driver for driver in actual_strategies if driver != subject_driver])
    n_rivals = len(rival_drivers)
    n_iterations = schedules.shape[0]

    rival_times = np.zeros((n_rivals, n_iterations), dtype=np.float32)
    schedule_maps = [_matrix_row_to_schedule_map(row) for row in schedules]

    for rival_idx, rival_driver in enumerate(rival_drivers):
        strategy = dict(actual_strategies[rival_driver])
        strategy.setdefault("event_name", event_name)
        strategy.setdefault("year", year)
        strategy.setdefault("round", round_num)

        if "baseline_pace_s" not in strategy or pd.isna(strategy.get("baseline_pace_s")):
            driver_laps = driver_laps_by_driver.get(rival_driver)
            try:
                strategy["baseline_pace_s"] = (
                    _driver_median_green_pace(driver_laps)
                    if driver_laps is not None
                    else race_baseline_pace
                )
            except ValueError:
                strategy["baseline_pace_s"] = race_baseline_pace

        pit_loss_s = _driver_pit_loss(driver_laps_by_driver[rival_driver])

        for it_idx, schedule_map in enumerate(schedule_maps):
            result = simulate_car_race(
                strategy=strategy,
                caution_schedule=schedule_map,
                degradation_stats=None,
                race_length=race_length,
                lambda_=lambda_,
                pit_loss_s=pit_loss_s,
                caution_pit_loss_s=CAUTION_PIT_LOSS_S,
                caution_pace_ratio=CAUTION_PACE_RATIO,
                degradation_lookup=degradation_lookup,
                collect_trace=False,
            )
            rival_times[rival_idx, it_idx] = np.float32(result["race_time_s"])

    return rival_times


def evaluate_strategy_batch(
    subject_strategies: List[dict],
    caution_schedules: np.ndarray,
    degradation_lookup: Mapping[Tuple[Any, ...], float],
    lambda_: float,
    baseline_pace_s: float,
    pit_loss_s: float,
    caution_pit_loss_s: float,
    caution_pace_ratio: float,
) -> np.ndarray:
    """Vectorized subject simulation over (n_strategies x n_iterations).

    The update loop runs only over laps. Within each lap, all strategy and
    iteration states are updated via NumPy array operations.
    """

    schedules = _validate_caution_schedules(caution_schedules)
    n_iterations, race_length = schedules.shape
    n_strategies = len(subject_strategies)
    if n_strategies == 0:
        return np.zeros((0, n_iterations), dtype=np.float32)

    # Precompute strategy encoding (outside the lap loop).
    parsed_stops = [_strategy_stops(strategy) for strategy in subject_strategies]
    max_stops = max((len(stops) for stops in parsed_stops), default=0)

    compound_set = set()
    for strategy, stops in zip(subject_strategies, parsed_stops):
        compound_set.add(_strategy_starting_compound(strategy))
        for _, compound in stops:
            compound_set.add(str(compound))
    compounds = sorted(compound_set)
    compound_to_idx = {compound: idx for idx, compound in enumerate(compounds)}

    stop_laps = np.full((n_strategies, max_stops), race_length + 1, dtype=np.int16)
    stop_compound_idx = np.full((n_strategies, max_stops), -1, dtype=np.int16)
    starting_compound_idx = np.zeros(n_strategies, dtype=np.int16)

    rates_by_strategy_compound = np.zeros((n_strategies, len(compounds)), dtype=np.float32)

    for s_idx, (strategy, stops) in enumerate(zip(subject_strategies, parsed_stops)):
        starting_compound = _strategy_starting_compound(strategy)
        starting_compound_idx[s_idx] = compound_to_idx[starting_compound]

        for stop_idx, (pit_lap, compound) in enumerate(stops):
            stop_laps[s_idx, stop_idx] = int(pit_lap)
            stop_compound_idx[s_idx, stop_idx] = compound_to_idx[str(compound)]

        for compound, comp_idx in compound_to_idx.items():
            rates_by_strategy_compound[s_idx, comp_idx] = np.float32(
                _lookup_degradation_rate(strategy, degradation_lookup, compound)
            )

    # State arrays over strategies x iterations.
    cumulative_time = np.zeros((n_strategies, n_iterations), dtype=np.float64)
    effective_age = np.ones((n_strategies, n_iterations), dtype=np.float32)

    baseline = np.float32(baseline_pace_s)
    pit_loss_green = np.float32(pit_loss_s)
    pit_loss_caution = np.float32(caution_pit_loss_s)
    caution_ratio = np.float32(caution_pace_ratio)
    lambda_val = np.float32(lambda_)

    strategy_indices = np.arange(n_strategies, dtype=np.int32)

    for lap in range(1, race_length + 1):
        caution_mask = schedules[:, lap - 1][None, :]  # 1 x n_iterations

        if max_stops == 0:
            has_stopped = np.zeros(n_strategies, dtype=bool)
            current_compound_idx = starting_compound_idx
            is_pit_lap = np.zeros((n_strategies, 1), dtype=bool)
        else:
            stops_completed = (stop_laps <= lap).sum(axis=1)  # n_strategies
            has_stopped = stops_completed > 0
            last_stop_idx = np.maximum(stops_completed - 1, 0)

            current_compound_idx = np.where(
                has_stopped,
                stop_compound_idx[strategy_indices, last_stop_idx],
                starting_compound_idx,
            )

            is_pit_lap = (stop_laps == lap).any(axis=1)[:, None]

        degradation_rate = rates_by_strategy_compound[strategy_indices, current_compound_idx][:, None]

        lap_time_green = baseline + (degradation_rate * effective_age)
        lap_time_caution = baseline * caution_ratio
        lap_time = np.where(caution_mask, lap_time_caution, lap_time_green)

        pit_add = np.where(caution_mask, pit_loss_caution, pit_loss_green)
        lap_time = lap_time + np.where(is_pit_lap, pit_add, np.float32(0.0))

        cumulative_time += lap_time

        effective_age = np.where(
            is_pit_lap,
            np.float32(1.0),
            effective_age + np.where(caution_mask, lambda_val, np.float32(1.0)),
        )

    return cumulative_time


def rank_and_score_batch(subject_times: np.ndarray, rival_times: np.ndarray) -> dict:
    """Rank subject versus fixed rivals and map to points without re-simulation."""

    subj = np.asarray(subject_times, dtype=np.float32)
    rivals = np.asarray(rival_times, dtype=np.float32)

    if subj.ndim != 2:
        raise ValueError("subject_times must be 2D (n_strategies, n_iterations)")
    if rivals.ndim != 2:
        raise ValueError("rival_times must be 2D (n_rivals, n_iterations)")
    if subj.shape[1] != rivals.shape[1]:
        raise ValueError("subject_times and rival_times must have the same n_iterations")

    n_strategies, _ = subj.shape

    rank = np.ones_like(subj, dtype=np.int16)
    for r_idx in range(rivals.shape[0]):
        rank += (rivals[r_idx][None, :] < subj).astype(np.int16)

    valid_rank = np.where(rank <= 10, rank, 0)
    points = POINTS_LOOKUP[valid_rank]

    mean_points = points.mean(axis=1)
    std_points = points.std(axis=1, ddof=1) if points.shape[1] > 1 else np.zeros(n_strategies, dtype=np.float32)
    mean_rank = rank.mean(axis=1)

    return {
        "mean_points": mean_points.astype(np.float32),
        "std_points": std_points.astype(np.float32),
        "mean_rank": mean_rank.astype(np.float32),
        "points_matrix": points,
        "rank_matrix": rank,
    }
