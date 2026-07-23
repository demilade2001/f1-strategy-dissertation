from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .config import (
    CAUTION_PACE_RATIO,
    CAUTION_PIT_LOSS_S,
    GRID_SPACING,
    INCLUDE_BIG_THREE_AS_RIVALS,
    MAX_STOPS,
    PROXIMITY_MARGIN_S,
)
from .monte_carlo import build_actual_strategies, build_probability_sources
from .race_model import _build_degradation_lookup, _driver_pit_loss
from .race_model_vectorized import evaluate_strategy_batch, precompute_rival_times, rank_and_score_batch
from .sc_sampler import sample_caution_schedule
from .strategy import enumerate_feasible_strategies


ARGMAX_BATCH_SIZE = 2048
BIG_THREE_TEAMS = {"Red Bull Racing", "Mercedes", "Ferrari"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


def _strategy_key(strategy: Mapping[str, object]) -> Tuple[str, Tuple[Tuple[int, str], ...]]:
    return (
        str(strategy["starting_compound"]),
        tuple((int(lap), str(compound)) for lap, compound in strategy.get("stops", [])),
    )


def _strategy_has_stop_lap(strategy: Mapping[str, object], lap: int) -> bool:
    target = int(lap)
    return any(int(stop_lap) == target for stop_lap, _ in strategy.get("stops", []))


def _is_wet_race(actual_strategies: Mapping[str, Mapping[str, object]]) -> bool:
    for strategy in actual_strategies.values():
        starting = str(strategy.get("starting_compound", ""))
        if starting in WET_COMPOUNDS:
            return True
        for _, compound in strategy.get("stops", []):
            if str(compound) in WET_COMPOUNDS:
                return True
    return False


def _driver_team_map(race_state: dict) -> Dict[str, str]:
    rows = race_state.get("drivers_with_teams", [])
    mapping: Dict[str, str] = {}
    for row in rows:
        driver = str(row.get("Driver", ""))
        team = str(row.get("Team", ""))
        if driver:
            mapping[driver] = team
    return mapping


def _eligible_rivals(
    race_state: dict,
    subject_driver: str,
    include_big_three_as_rivals: bool = INCLUDE_BIG_THREE_AS_RIVALS,
) -> Set[str]:
    team_map = _driver_team_map(race_state)
    subject_team = team_map.get(str(subject_driver), "")
    rivals: Set[str] = set()
    for driver, team in team_map.items():
        if driver == str(subject_driver):
            continue
        if subject_team and team == subject_team:
            continue
        if (not include_big_three_as_rivals) and (team in BIG_THREE_TEAMS):
            continue
        rivals.add(driver)
    return rivals


def _stops_by_driver(actual_strategies: Mapping[str, Mapping[str, object]]) -> Dict[str, Set[int]]:
    out: Dict[str, Set[int]] = {}
    for driver, strategy in actual_strategies.items():
        out[str(driver)] = {int(lap) for lap, _ in strategy.get("stops", [])}
    return out


def detect_rival_trigger_events(
    race_state: dict,
    subject_driver: str,
    proximity_margin_s: float = PROXIMITY_MARGIN_S,
    include_big_three_as_rivals: bool = INCLUDE_BIG_THREE_AS_RIVALS,
) -> Dict[str, object]:
    laps = race_state["laps"].copy()
    subject_rows = laps[laps["Driver"].astype(str) == str(subject_driver)].copy()
    if subject_rows.empty:
        raise ValueError(f"Subject driver {subject_driver} not found in race laps")

    actual_strategies = build_actual_strategies(laps)
    if str(subject_driver) not in actual_strategies:
        raise ValueError(f"Subject driver {subject_driver} not found in actual strategies")

    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    rivals_all = _eligible_rivals(
        race_state=race_state,
        subject_driver=str(subject_driver),
        include_big_three_as_rivals=bool(include_big_three_as_rivals),
    )

    subject_stop_laps = {int(lap) for lap, _ in actual_strategies[str(subject_driver)].get("stops", [])}
    rival_stop_laps = _stops_by_driver(actual_strategies)

    trigger_events: List[Dict[str, object]] = []
    threat_basis: Dict[int, Dict[str, object]] = {}
    for _, row in subject_rows.sort_values("LapNumber").iterrows():
        lap = int(row["LapNumber"])
        if lap < 1 or lap >= race_length:
            continue

        under_flag = bool(row.get("undercut_threat_flag", False))
        over_flag = bool(row.get("overcut_threat_flag", False))
        under_idx = float(row.get("undercut_threat_index", np.nan))
        over_idx = float(row.get("overcut_threat_index", np.nan))

        # threat indexes are dimensionless age/gap ratios in this project, not plain seconds.
        # use the explicit threat flags to gate rival-eligibility on each lap.
        threat_active = bool(under_flag or over_flag)

        if not threat_active:
            # Fallback branch retained for forward-compatibility if threat index semantics change.
            if np.isfinite(under_idx) and under_idx <= float(proximity_margin_s):
                threat_active = True
            if np.isfinite(over_idx) and over_idx <= float(proximity_margin_s):
                threat_active = True

        rivals_on_lap = set(rivals_all) if threat_active else set()
        threat_basis[lap] = {
            "lap": int(lap),
            "undercut_threat_flag": under_flag,
            "overcut_threat_flag": over_flag,
            "undercut_threat_index": under_idx,
            "overcut_threat_index": over_idx,
            "rival_set_size": int(len(rivals_on_lap)),
            "rival_set": sorted(rivals_on_lap),
        }

        if not rivals_on_lap:
            continue

        if lap in subject_stop_laps:
            continue

        triggered_by = sorted(
            [
                rival
                for rival in rivals_on_lap
                if int(lap) in rival_stop_laps.get(str(rival), set())
            ]
        )
        if triggered_by:
            trigger_events.append(
                {
                    "lap": int(lap),
                    "triggered_by": triggered_by,
                    "rival_set_size": int(len(rivals_on_lap)),
                }
            )

    return {
        "race_length": int(race_length),
        "subject_driver": str(subject_driver),
        "rivals_eligible": sorted(rivals_all),
        "subject_stop_laps": sorted(subject_stop_laps),
        "trigger_events": trigger_events,
        "threat_basis_by_lap": threat_basis,
    }


def build_reactive_candidate_sets(
    race_state: dict,
    subject_driver: str,
    trigger_events: Sequence[Mapping[str, object]],
    min_stint_length_laps: int,
    grid_spacing: int = GRID_SPACING,
) -> Dict[str, object]:
    laps = race_state["laps"].copy()
    actual_strategies = build_actual_strategies(laps)
    if str(subject_driver) not in actual_strategies:
        raise ValueError(f"Subject driver {subject_driver} not found in race strategies")

    subject_actual = dict(actual_strategies[str(subject_driver)])
    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    wet_race = _is_wet_race(actual_strategies)

    full_space = enumerate_feasible_strategies(
        race_length=race_length,
        starting_compound=str(subject_actual["starting_compound"]),
        is_wet_race=wet_race,
        max_stops=MAX_STOPS,
        min_stint_length_laps=int(min_stint_length_laps),
    )
    if hasattr(full_space, "materialize"):
        materialized = full_space.materialize(limit=2_000_000)
    else:
        materialized = list(full_space)

    standard_candidate_laps = set(range(1, race_length, int(grid_spacing)))

    def enrich(strategy: Dict[str, object]) -> Dict[str, object]:
        row = dict(strategy)
        row.setdefault("baseline_pace_s", float(subject_actual["baseline_pace_s"]))
        row.setdefault("year", int(subject_actual["year"]))
        row.setdefault("round", int(subject_actual["round"]))
        row.setdefault("event_name", str(subject_actual["event_name"]))
        return row

    standard: List[Dict[str, object]] = []
    standard_keys: Set[Tuple[str, Tuple[Tuple[int, str], ...]]] = set()
    for strategy in materialized:
        stops = strategy.get("stops", [])
        if all(int(lap) in standard_candidate_laps for lap, _ in stops):
            es = enrich(strategy)
            key = _strategy_key(es)
            standard.append(es)
            standard_keys.add(key)

    reactive_only: List[Dict[str, object]] = []
    reactive_keys: Set[Tuple[str, Tuple[Tuple[int, str], ...]]] = set()
    trigger_laps = sorted({int(e["lap"]) for e in trigger_events})

    augmented_laps_used: List[int] = []
    skipped_already_grid: List[int] = []
    for trigger_lap in trigger_laps:
        augmented_lap = int(trigger_lap) + 1
        if augmented_lap <= 0 or augmented_lap >= race_length:
            continue
        if augmented_lap in standard_candidate_laps:
            skipped_already_grid.append(augmented_lap)
            continue

        augmented_laps_used.append(augmented_lap)
        allowed = set(standard_candidate_laps)
        allowed.add(augmented_lap)

        for strategy in materialized:
            stops = [(int(lap), str(comp)) for lap, comp in strategy.get("stops", [])]
            if not all(lap in allowed for lap, _ in stops):
                continue
            if not any(lap == augmented_lap for lap, _ in stops):
                continue

            es = enrich(strategy)
            key = _strategy_key(es)
            if key in standard_keys or key in reactive_keys:
                continue
            reactive_only.append(es)
            reactive_keys.add(key)

    augmented = list(standard)
    augmented.extend(reactive_only)

    return {
        "race_length": int(race_length),
        "wet_race": bool(wet_race),
        "subject_actual": subject_actual,
        "standard_strategies": standard,
        "reactive_only_strategies": reactive_only,
        "augmented_strategies": augmented,
        "standard_count": int(len(standard)),
        "reactive_only_count": int(len(reactive_only)),
        "augmented_count": int(len(augmented)),
        "trigger_laps": trigger_laps,
        "augmented_laps_used": sorted(set(augmented_laps_used)),
        "skipped_already_grid_laps": sorted(set(skipped_already_grid)),
    }


def _score_strategy_pool(
    race_state: dict,
    subject_driver: str,
    strategies: Sequence[Mapping[str, object]],
    prob_source: str,
    lambda_: float,
    n_iterations: int,
    rng,
) -> Dict[str, object]:
    if not strategies:
        raise ValueError("No strategies were provided for scoring")

    laps = race_state["laps"].copy()
    actual_strategies = build_actual_strategies(laps)
    subject_actual = dict(actual_strategies[str(subject_driver)])

    prob_by_lap = _resolve_probability_by_source(race_state, prob_source)
    caution_schedules = _build_caution_schedule_matrix(prob_by_lap, int(n_iterations), rng)
    precomputed = _precompute_subject_inputs(
        race_state=race_state,
        subject_driver=str(subject_driver),
        caution_schedules=caution_schedules,
        lambda_=float(lambda_),
    )

    mean_points = np.zeros(len(strategies), dtype=np.float64)
    std_points = np.zeros(len(strategies), dtype=np.float64)
    mean_rank = np.zeros(len(strategies), dtype=np.float64)
    win_rate = np.zeros(len(strategies), dtype=np.float64)

    for start in range(0, len(strategies), ARGMAX_BATCH_SIZE):
        end = min(start + ARGMAX_BATCH_SIZE, len(strategies))
        batch = [dict(s) for s in strategies[start:end]]
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

        rank_matrix = np.asarray(scored["rank_matrix"], dtype=np.int16)
        win_rate[start:end] = np.mean(rank_matrix == 1, axis=1)

    return {
        "mean_points": mean_points,
        "std_points": std_points,
        "mean_rank": mean_rank,
        "win_rate": win_rate,
    }


def compute_rival_conditioned_benchmarks(
    race_state: dict,
    subject_driver: str,
    prob_source: str,
    lambda_: float,
    n_iterations: int,
    rng,
    min_stint_length_laps: int,
    proximity_margin_s: float = PROXIMITY_MARGIN_S,
    include_big_three_as_rivals: bool = INCLUDE_BIG_THREE_AS_RIVALS,
) -> Dict[str, object]:
    trigger_data = detect_rival_trigger_events(
        race_state=race_state,
        subject_driver=str(subject_driver),
        proximity_margin_s=float(proximity_margin_s),
        include_big_three_as_rivals=bool(include_big_three_as_rivals),
    )

    candidate_sets = build_reactive_candidate_sets(
        race_state=race_state,
        subject_driver=str(subject_driver),
        trigger_events=trigger_data["trigger_events"],
        min_stint_length_laps=int(min_stint_length_laps),
        grid_spacing=int(GRID_SPACING),
    )

    augmented = candidate_sets["augmented_strategies"]
    scoring = _score_strategy_pool(
        race_state=race_state,
        subject_driver=str(subject_driver),
        strategies=augmented,
        prob_source=str(prob_source),
        lambda_=float(lambda_),
        n_iterations=int(n_iterations),
        rng=rng,
    )

    standard_keys = {_strategy_key(s) for s in candidate_sets["standard_strategies"]}
    augmented_keys = [_strategy_key(s) for s in augmented]

    standard_idx = [idx for idx, key in enumerate(augmented_keys) if key in standard_keys]
    augmented_idx = list(range(len(augmented)))

    def best_from_indices(indices: Sequence[int]) -> Optional[Dict[str, object]]:
        if not indices:
            return None
        idx_arr = np.asarray(list(indices), dtype=np.int64)
        local = idx_arr[np.argmax(scoring["mean_points"][idx_arr])]
        i = int(local)
        return {
            "strategy": augmented[i],
            "mean_points": float(scoring["mean_points"][i]),
            "std_points": float(scoring["std_points"][i]),
            "mean_rank": float(scoring["mean_rank"][i]),
            "win_rate": float(scoring["win_rate"][i]),
            "index": i,
        }

    bench_a = best_from_indices(standard_idx)
    bench_b = best_from_indices(augmented_idx)
    benchmark_r_by_trigger_lap: Dict[int, Optional[Dict[str, object]]] = {}
    for event in trigger_data["trigger_events"]:
        added_lap = int(event["lap"]) + 1
        r_idx = [idx for idx, strategy in enumerate(augmented) if _strategy_has_stop_lap(strategy, added_lap)]
        benchmark_r_by_trigger_lap[int(event["lap"])] = best_from_indices(r_idx)

    r_union_idx = [
        idx
        for idx, strategy in enumerate(augmented)
        if any(_strategy_has_stop_lap(strategy, int(event["lap"]) + 1) for event in trigger_data["trigger_events"])
    ]
    bench_r = best_from_indices(r_union_idx)

    return {
        "trigger_data": trigger_data,
        "candidate_sets": candidate_sets,
        "benchmark_a": bench_a,
        "benchmark_b": bench_b,
        "benchmark_r_anchoring": bench_r,
        "benchmark_r_anchoring_by_trigger_lap": benchmark_r_by_trigger_lap,
        "standard_index_count": int(len(standard_idx)),
        "reactive_index_count": int(len(r_union_idx)),
        "augmented_index_count": int(len(augmented_idx)),
    }


def compute_benchmarks_for_trigger_events(
    race_state: dict,
    subject_driver: str,
    trigger_events: Sequence[Mapping[str, object]],
    prob_source: str,
    lambda_: float,
    n_iterations: int,
    rng,
    min_stint_length_laps: int,
) -> Dict[str, object]:
    candidate_sets = build_reactive_candidate_sets(
        race_state=race_state,
        subject_driver=str(subject_driver),
        trigger_events=trigger_events,
        min_stint_length_laps=int(min_stint_length_laps),
        grid_spacing=int(GRID_SPACING),
    )

    augmented = candidate_sets["augmented_strategies"]
    scoring = _score_strategy_pool(
        race_state=race_state,
        subject_driver=str(subject_driver),
        strategies=augmented,
        prob_source=str(prob_source),
        lambda_=float(lambda_),
        n_iterations=int(n_iterations),
        rng=rng,
    )

    standard_keys = {_strategy_key(s) for s in candidate_sets["standard_strategies"]}
    augmented_keys = [_strategy_key(s) for s in augmented]

    standard_idx = [idx for idx, key in enumerate(augmented_keys) if key in standard_keys]
    augmented_idx = list(range(len(augmented)))

    def best_from_indices(indices: Sequence[int]) -> Optional[Dict[str, object]]:
        if not indices:
            return None
        idx_arr = np.asarray(list(indices), dtype=np.int64)
        local = idx_arr[np.argmax(scoring["mean_points"][idx_arr])]
        i = int(local)
        return {
            "strategy": augmented[i],
            "mean_points": float(scoring["mean_points"][i]),
            "std_points": float(scoring["std_points"][i]),
            "mean_rank": float(scoring["mean_rank"][i]),
            "win_rate": float(scoring["win_rate"][i]),
            "index": i,
        }

    benchmark_r_by_trigger_lap: Dict[int, Optional[Dict[str, object]]] = {}
    r_union_idx: Set[int] = set()
    for event in trigger_events:
        event_lap = int(event["lap"])
        added_lap = event_lap + 1
        r_idx = [idx for idx, strategy in enumerate(augmented) if _strategy_has_stop_lap(strategy, added_lap)]
        benchmark_r_by_trigger_lap[event_lap] = best_from_indices(r_idx)
        r_union_idx.update(r_idx)

    return {
        "candidate_sets": candidate_sets,
        "benchmark_a": best_from_indices(standard_idx),
        "benchmark_b": best_from_indices(augmented_idx),
        "benchmark_r_anchoring": best_from_indices(sorted(r_union_idx)),
        "benchmark_r_anchoring_by_trigger_lap": benchmark_r_by_trigger_lap,
        "standard_index_count": int(len(standard_idx)),
        "reactive_index_count": int(len(r_union_idx)),
        "augmented_index_count": int(len(augmented_idx)),
    }


def score_subject_strategy_pool(
    race_state: dict,
    subject_driver: str,
    strategies: Sequence[Mapping[str, object]],
    prob_source: str,
    lambda_: float,
    n_iterations: int,
    rng,
) -> Dict[str, object]:
    return _score_strategy_pool(
        race_state=race_state,
        subject_driver=str(subject_driver),
        strategies=strategies,
        prob_source=str(prob_source),
        lambda_=float(lambda_),
        n_iterations=int(n_iterations),
        rng=rng,
    )


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
    is_wet_race: bool,
) -> List[Dict[str, object]]:
    base_space = enumerate_feasible_strategies(
        race_length=race_length,
        starting_compound=str(subject_actual["starting_compound"]),
        is_wet_race=bool(is_wet_race),
        max_stops=MAX_STOPS,
    )
    if hasattr(base_space, "materialize"):
        base_materialized = base_space.materialize(limit=2_000_000)
    else:
        base_materialized = list(base_space)

    grid_candidate_laps = set(range(1, race_length, GRID_SPACING))
    filtered = [s for s in base_materialized if _is_grid_spaced_strategy(s, grid_candidate_laps)]
    out = [_with_subject_metadata(s, subject_actual) for s in filtered]

    # Preserve realizability: if the historical subject strategy is no-stop,
    # ensure it exists in the candidate pool even when constrained grid/dry
    # enumeration would otherwise exclude it.
    actual_stops = [(int(lap), str(comp)) for lap, comp in subject_actual.get("stops", [])]
    if len(actual_stops) == 0:
        actual_candidate = _with_subject_metadata(
            {
                "starting_compound": str(subject_actual["starting_compound"]),
                "stops": [],
            },
            subject_actual,
        )
        existing_keys = {_strategy_key(s) for s in out}
        if _strategy_key(actual_candidate) not in existing_keys:
            out.append(actual_candidate)

    return out


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
    wet_race = _is_wet_race(actual_strategies)

    prob_by_lap = _resolve_probability_by_source(race_state, prob_source)
    caution_schedules = _build_caution_schedule_matrix(prob_by_lap, int(n_iterations), rng)

    subject_strategies = _enumerate_unconditioned_subject_strategies(
        race_length,
        subject_actual,
        is_wet_race=wet_race,
    )
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


__all__ = [
    "argmax_strategy",
    "detect_rival_trigger_events",
    "build_reactive_candidate_sets",
    "compute_rival_conditioned_benchmarks",
    "compute_benchmarks_for_trigger_events",
    "score_subject_strategy_pool",
]
