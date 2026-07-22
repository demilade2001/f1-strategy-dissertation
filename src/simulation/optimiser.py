from __future__ import annotations

from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

from .config import CAUTION_PACE_RATIO, CAUTION_PIT_LOSS_S, GRID_SPACING, MAX_STOPS
from .monte_carlo import build_actual_strategies, build_probability_sources
from .race_model import _build_degradation_lookup, _driver_pit_loss
from .race_model_vectorized import evaluate_strategy_batch, precompute_rival_times, rank_and_score_batch
from .sc_sampler import sample_caution_schedule
from .strategy import enumerate_feasible_strategies


ARGMAX_BATCH_SIZE = 2048


def _build_caution_schedule_matrix(prob_by_lap: Mapping[int, float], n_iterations: int, rng) -> np.ndarray:
    race_length = int(max(prob_by_lap.keys()))
    schedules = np.zeros((int(n_iterations), race_length), dtype=bool)

    for idx in range(int(n_iterations)):
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


def _resolve_probability_by_source(race_state: dict, prob_source: str) -> Dict[int, float]:
    sources = build_probability_sources(race_state)
    if prob_source == "lap_level":
        return dict(sources["merged"])
    if prob_source == "static_prior":
        return dict(sources["flat"])
    raise ValueError("prob_source must be one of {'lap_level', 'static_prior'}")


def _enumerate_unconditioned_subject_strategies(
    race_length: int,
    subject_actual: Dict[str, object],
) -> List[Dict[str, object]]:
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
    filtered = [s for s in base_materialized if _is_grid_spaced_strategy(s, grid_candidate_laps)]
    return [_with_subject_metadata(s, subject_actual) for s in filtered]


def _precompute_subject_inputs(
    race_state: dict,
    subject_driver: str,
    caution_schedules: np.ndarray,
    lambda_: float,
) -> dict:
    laps = race_state["laps"].copy()
    subject_laps = laps[laps["Driver"].astype(str) == str(subject_driver)].copy()
    return {
        "pit_loss_s": float(_driver_pit_loss(subject_laps)),
        "degradation_lookup": _build_degradation_lookup(race_state.get("deg_stats")),
        "rival_times": precompute_rival_times(
            race_state=race_state,
            subject_driver=subject_driver,
            caution_schedules=caution_schedules,
            lambda_=float(lambda_),
        ),
    }

def argmax_strategy(
    race_state: dict,
    subject_driver: str,
    prob_source: str,
    information_set: str,
    lambda_: float,
    n_iterations: int,
    rng,
    epsilon: float = 0.01,
) -> dict:
    if information_set != "unconditioned":
        raise NotImplementedError(
            "Only information_set='unconditioned' is implemented; rival-conditioned information sets are not yet built."
        )

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    laps = race_state["laps"].copy()
    actual_strategies = build_actual_strategies(laps)
    if subject_driver not in actual_strategies:
        raise ValueError(f"subject_driver {subject_driver} not found in race strategies")

    subject_actual = dict(actual_strategies[subject_driver])
    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())

    prob_by_lap = _resolve_probability_by_source(race_state, prob_source)
    caution_schedules = _build_caution_schedule_matrix(prob_by_lap, int(n_iterations), rng)

    subject_strategies = _enumerate_unconditioned_subject_strategies(race_length, subject_actual)
    if not subject_strategies:
        raise ValueError("No feasible unconditioned subject strategies found")

    precomputed = _precompute_subject_inputs(
        race_state=race_state,
        subject_driver=subject_driver,
        caution_schedules=caution_schedules,
        lambda_=float(lambda_),
    )

    mean_points = np.zeros(len(subject_strategies), dtype=np.float64)
    std_points = np.zeros(len(subject_strategies), dtype=np.float64)
    mean_rank = np.zeros(len(subject_strategies), dtype=np.float64)

    for start in range(0, len(subject_strategies), ARGMAX_BATCH_SIZE):
        end = min(start + ARGMAX_BATCH_SIZE, len(subject_strategies))
        batch = subject_strategies[start:end]
        subject_times = evaluate_strategy_batch(
            subject_strategies=batch,
            caution_schedules=caution_schedules,
            degradation_lookup=precomputed["degradation_lookup"],
            lambda_=float(lambda_),
            baseline_pace_s=float(subject_actual["baseline_pace_s"]),
            pit_loss_s=float(precomputed["pit_loss_s"]),
            caution_pit_loss_s=CAUTION_PIT_LOSS_S,
            caution_pace_ratio=CAUTION_PACE_RATIO,
        )
        scored = rank_and_score_batch(subject_times, precomputed["rival_times"])
        mean_points[start:end] = np.asarray(scored["mean_points"], dtype=np.float64)
        std_points[start:end] = np.asarray(scored["std_points"], dtype=np.float64)
        mean_rank[start:end] = np.asarray(scored["mean_rank"], dtype=np.float64)

    max_mean_points = float(np.max(mean_points))
    best_idx = int(np.argmax(mean_points))
    epsilon_cutoff = max_mean_points - float(epsilon)
    within_eps_idx = np.flatnonzero(mean_points >= epsilon_cutoff)

    near_optimal = []
    for idx in within_eps_idx.tolist():
        near_optimal.append(
            {
                "strategy_index": int(idx),
                "mean_points": float(mean_points[idx]),
                "std_points": float(std_points[idx]),
                "mean_rank": float(mean_rank[idx]),
                "strategy": subject_strategies[idx],
            }
        )

    near_optimal.sort(key=lambda row: (-row["mean_points"], row["strategy_index"]))

    return {
        "prob_source": str(prob_source),
        "information_set": str(information_set),
        "subject_driver": str(subject_driver),
        "n_iterations": int(n_iterations),
        "epsilon": float(epsilon),
        "grid_spacing": int(GRID_SPACING),
        "max_stops": int(MAX_STOPS),
        "strategy_count": int(len(subject_strategies)),
        "best_strategy_index": best_idx,
        "best_strategy": subject_strategies[best_idx],
        "best_mean_points": max_mean_points,
        "best_std_points": float(std_points[best_idx]),
        "best_mean_rank": float(mean_rank[best_idx]),
        "within_epsilon_count": int(len(near_optimal)),
        "within_epsilon_strategies": near_optimal,
    }


__all__ = ["argmax_strategy"]
