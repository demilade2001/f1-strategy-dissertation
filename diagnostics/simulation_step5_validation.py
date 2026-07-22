from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.config import GRID_SPACING, LOCKED_ARCHETYPE_RACES, MAX_STOPS
from src.simulation.monte_carlo import (
    build_actual_strategies,
    build_probability_sources,
    select_midfield_subject_driver,
)
from src.simulation.optimiser import argmax_strategy
from src.simulation.race_model import _build_degradation_lookup, _driver_pit_loss
from src.simulation.race_model_vectorized import (
    evaluate_strategy_batch,
    precompute_rival_times,
    rank_and_score_batch,
)
from src.simulation.race_state import load_race_state
from src.simulation.sc_sampler import sample_caution_schedule

OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step5_validation_output.txt"

HUNGARY_2022 = {"event_name": "Hungarian Grand Prix", "year": 2022, "round": 13}
MONACO_CLASS_RACE = {"event_name": "Monaco Grand Prix", "year": 2023, "round": 6}
MONZA_CLASS_RACE = {"event_name": "Italian Grand Prix", "year": 2024, "round": 16}

LAMBDA_VALUE = 1.0
N_ITERATIONS = 10_000
SEED = 42
EPSILON = 0.01


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _build_caution_schedule_matrix(prob_by_lap: Mapping[int, float], n_iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    race_length = int(max(prob_by_lap.keys()))
    schedules = np.zeros((n_iterations, race_length), dtype=bool)

    for idx in range(n_iterations):
        schedule = sample_caution_schedule(prob_by_lap, rng)
        for lap in range(1, race_length + 1):
            schedules[idx, lap - 1] = bool(schedule.get(lap, False))
    return schedules


def _strategy_key(strategy: Dict[str, object]) -> Tuple[str, Tuple[Tuple[int, str], ...]]:
    starting = str(strategy["starting_compound"])
    stops = tuple((int(lap), str(compound)) for lap, compound in strategy.get("stops", []))
    return starting, stops


def _with_subject_metadata(strategy: Dict[str, object], subject_actual: Dict[str, object]) -> Dict[str, object]:
    enriched = dict(strategy)
    enriched.setdefault("baseline_pace_s", float(subject_actual["baseline_pace_s"]))
    enriched.setdefault("year", int(subject_actual["year"]))
    enriched.setdefault("round", int(subject_actual["round"]))
    enriched.setdefault("event_name", str(subject_actual["event_name"]))
    return enriched


def _resolve_prob_by_source(race_state: dict, prob_source: str) -> Dict[int, float]:
    sources = build_probability_sources(race_state)
    if prob_source == "lap_level":
        return dict(sources["merged"])
    if prob_source == "static_prior":
        return dict(sources["flat"])
    raise ValueError("prob_source must be one of {'lap_level', 'static_prior'}")


def _score_subject_strategies(
    race_state: dict,
    subject_driver: str,
    subject_strategies: Sequence[Dict[str, object]],
    prob_source: str,
    lambda_: float,
    n_iterations: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    if not subject_strategies:
        raise ValueError("subject_strategies cannot be empty")

    laps = race_state["laps"].copy()
    prob_by_lap = _resolve_prob_by_source(race_state, prob_source)
    caution_schedules = _build_caution_schedule_matrix(prob_by_lap, int(n_iterations), int(seed))

    rival_times = precompute_rival_times(
        race_state=race_state,
        subject_driver=subject_driver,
        caution_schedules=caution_schedules,
        lambda_=float(lambda_),
    )

    subject_laps = laps[laps["Driver"].astype(str) == str(subject_driver)].copy()
    pit_loss_s = float(_driver_pit_loss(subject_laps))
    degradation_lookup = _build_degradation_lookup(race_state.get("deg_stats"))

    subject_times = evaluate_strategy_batch(
        subject_strategies=list(subject_strategies),
        caution_schedules=caution_schedules,
        degradation_lookup=degradation_lookup,
        lambda_=float(lambda_),
        baseline_pace_s=float(subject_strategies[0]["baseline_pace_s"]),
        pit_loss_s=pit_loss_s,
        caution_pit_loss_s=5.0,
        caution_pace_ratio=1.3162357008284764,
    )
    scored = rank_and_score_batch(subject_times, rival_times)
    return {
        "mean_points": np.asarray(scored["mean_points"], dtype=np.float64),
        "std_points": np.asarray(scored["std_points"], dtype=np.float64),
        "mean_rank": np.asarray(scored["mean_rank"], dtype=np.float64),
    }


def _is_clearly_implausible(strategy: Dict[str, object], race_length: int) -> bool:
    stops = [(int(lap), str(compound)) for lap, compound in strategy.get("stops", [])]
    if len(stops) < MAX_STOPS:
        return False

    compounds = [compound for _, compound in stops]
    early_or_late = any(lap <= 3 or lap >= (race_length - 2) for lap, _ in stops)
    repeated_compounds_only = len(set(compounds)) == 1
    return bool(early_or_late and repeated_compounds_only)


def _run_argmax_for_race(
    race_ref: Mapping[str, object],
    prob_source: str,
    information_set: str,
    lambda_: float,
    n_iterations: int,
    seed: int,
    epsilon: float,
) -> Dict[str, object]:
    state = load_race_state(int(race_ref["year"]), int(race_ref["round"]))
    laps = state["laps"].copy()
    laps["sc_active"] = _to_bool(laps["sc_active"])
    laps["vsc_active"] = _to_bool(laps["vsc_active"])

    subject_driver = select_midfield_subject_driver(laps)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    result = argmax_strategy(
        race_state=state,
        subject_driver=subject_driver,
        prob_source=prob_source,
        information_set=information_set,
        lambda_=lambda_,
        n_iterations=n_iterations,
        rng=rng,
        epsilon=epsilon,
    )
    elapsed = time.perf_counter() - t0

    result["wall_clock_seconds"] = float(elapsed)
    result["subject_driver"] = str(subject_driver)
    result["race_event_name"] = str(race_ref["event_name"])
    result["race_year"] = int(race_ref["year"])
    result["race_round"] = int(race_ref["round"])
    return result


def main() -> None:
    lines: List[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 100)
    emit("Step 1 - Confirm shared defaults")
    emit("=" * 100)
    emit(f"GRID_SPACING={GRID_SPACING}")
    emit(f"MAX_STOPS={MAX_STOPS}")

    hungary_state = load_race_state(HUNGARY_2022["year"], HUNGARY_2022["round"])
    hungary_laps = hungary_state["laps"].copy()
    hungary_laps["sc_active"] = _to_bool(hungary_laps["sc_active"])
    hungary_laps["vsc_active"] = _to_bool(hungary_laps["vsc_active"])
    hungary_subject = select_midfield_subject_driver(hungary_laps)

    emit("")
    emit("=" * 100)
    emit("Step 3 - Hungary 2022 argmax (lap_level, unconditioned)")
    emit("=" * 100)

    hungary_lap_level = _run_argmax_for_race(
        race_ref=HUNGARY_2022,
        prob_source="lap_level",
        information_set="unconditioned",
        lambda_=LAMBDA_VALUE,
        n_iterations=N_ITERATIONS,
        seed=SEED,
        epsilon=EPSILON,
    )

    emit(
        f"race={hungary_lap_level['race_event_name']} "
        f"({hungary_lap_level['race_year']} round {hungary_lap_level['race_round']})"
    )
    emit(f"subject_driver={hungary_lap_level['subject_driver']}")
    emit(f"best_strategy={json.dumps(hungary_lap_level['best_strategy'], sort_keys=True)}")
    emit(f"best_mean_points={hungary_lap_level['best_mean_points']:.6f}")
    emit(f"within_epsilon_count={hungary_lap_level['within_epsilon_count']}")
    emit("within_epsilon_strategies_begin")
    for row in hungary_lap_level["within_epsilon_strategies"]:
        emit(json.dumps(row, sort_keys=True))
    emit("within_epsilon_strategies_end")
    emit(f"wall_clock_seconds={hungary_lap_level['wall_clock_seconds']:.6f}")

    emit("")
    emit("=" * 100)
    emit("Step 4 - Plausibility check against actual history")
    emit("=" * 100)

    actual_strategies = build_actual_strategies(hungary_laps)
    subject_actual = dict(actual_strategies[hungary_subject])
    actual_strategy = _with_subject_metadata(subject_actual, subject_actual)

    scored = _score_subject_strategies(
        race_state=hungary_state,
        subject_driver=hungary_subject,
        subject_strategies=[actual_strategy],
        prob_source="lap_level",
        lambda_=LAMBDA_VALUE,
        n_iterations=N_ITERATIONS,
        seed=SEED,
    )
    actual_mean = float(scored["mean_points"][0])
    best_mean = float(hungary_lap_level["best_mean_points"])
    gap = best_mean - actual_mean

    emit(f"actual_strategy={json.dumps(actual_strategy, sort_keys=True)}")
    emit(f"actual_strategy_mean_points={actual_mean:.6f}")
    emit(f"argmax_mean_points={best_mean:.6f}")
    emit(f"actual_gap_to_argmax={gap:.6f}")

    if gap <= EPSILON:
        relation = "within epsilon-band of argmax"
    else:
        relation = "meaningfully below argmax"
    emit(f"actual_vs_argmax_classification={relation}")

    race_length = int(pd.to_numeric(hungary_laps["LapNumber"], errors="coerce").max())
    implausible_near_actual = []
    for row in hungary_lap_level["within_epsilon_strategies"]:
        strategy = row["strategy"]
        if _is_clearly_implausible(strategy, race_length):
            if abs(float(row["mean_points"]) - actual_mean) <= EPSILON:
                implausible_near_actual.append(row)

    if implausible_near_actual:
        emit("implausible_near_actual_begin")
        for row in implausible_near_actual:
            emit(json.dumps(row, sort_keys=True))
        emit("implausible_near_actual_end")
        raise RuntimeError(
            "BUG: actual strategy is almost equal to clearly-implausible near-optimal candidate(s); investigate enumeration/evaluation before proceeding"
        )
    emit("implausible_near_actual=none")

    emit("")
    emit("=" * 100)
    emit("Step 5 - Repeat Hungary with static_prior")
    emit("=" * 100)

    hungary_static = _run_argmax_for_race(
        race_ref=HUNGARY_2022,
        prob_source="static_prior",
        information_set="unconditioned",
        lambda_=LAMBDA_VALUE,
        n_iterations=N_ITERATIONS,
        seed=SEED,
        epsilon=EPSILON,
    )

    emit(f"lap_level_best_strategy={json.dumps(hungary_lap_level['best_strategy'], sort_keys=True)}")
    emit(f"lap_level_best_mean_points={hungary_lap_level['best_mean_points']:.6f}")
    emit(f"static_prior_best_strategy={json.dumps(hungary_static['best_strategy'], sort_keys=True)}")
    emit(f"static_prior_best_mean_points={hungary_static['best_mean_points']:.6f}")

    same_strategy = _strategy_key(hungary_lap_level["best_strategy"]) == _strategy_key(hungary_static["best_strategy"])
    same_mean = abs(float(hungary_lap_level["best_mean_points"]) - float(hungary_static["best_mean_points"])) <= 1e-12
    emit(f"lap_level_vs_static_prior_same_strategy={same_strategy}")
    emit(f"lap_level_vs_static_prior_same_mean={same_mean}")

    if same_strategy and same_mean:
        raise RuntimeError(
            "BUG: lap_level and static_prior argmax outputs are identical; B and R_SC should be distinguishable"
        )

    emit("")
    emit("=" * 100)
    emit("Step 6 - Runtime sanity check across race sizes")
    emit("=" * 100)

    runtime_races = [MONZA_CLASS_RACE, MONACO_CLASS_RACE]
    for race_ref in runtime_races:
        result = _run_argmax_for_race(
            race_ref=race_ref,
            prob_source="lap_level",
            information_set="unconditioned",
            lambda_=LAMBDA_VALUE,
            n_iterations=N_ITERATIONS,
            seed=SEED,
            epsilon=EPSILON,
        )
        emit(
            f"race={result['race_event_name']} ({result['race_year']} round {result['race_round']})"
        )
        emit(f"subject_driver={result['subject_driver']}")
        emit(f"strategy_count={result['strategy_count']}")
        emit(f"best_mean_points={result['best_mean_points']:.6f}")
        emit(f"wall_clock_seconds={result['wall_clock_seconds']:.6f}")

    emit("")
    emit("used_locked_archetype_races_for_runtime_check_begin")
    lock_df = pd.DataFrame(LOCKED_ARCHETYPE_RACES)
    subset = lock_df[
        (lock_df["event_name"].isin([MONZA_CLASS_RACE["event_name"], MONACO_CLASS_RACE["event_name"]]))
        & (lock_df["year"].isin([MONZA_CLASS_RACE["year"], MONACO_CLASS_RACE["year"]]))
    ][["archetype", "event_name", "year", "round"]]
    emit(subset.to_string(index=False))
    emit("used_locked_archetype_races_for_runtime_check_end")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
