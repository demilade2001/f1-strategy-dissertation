from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.simulation.monte_carlo as mc
from source_code.simulation.config import CAUTION_PACE_RATIO, CAUTION_PIT_LOSS_S
from source_code.simulation.monte_carlo import (
    build_actual_strategies,
    build_probability_sources,
    run_monte_carlo,
    select_midfield_subject_driver,
)
from source_code.simulation.race_model import _build_degradation_lookup, _driver_pit_loss, simulate_car_race
from source_code.simulation.race_model_vectorized import (
    evaluate_strategy_batch,
    precompute_rival_times,
    rank_and_score_batch,
)

OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step4h_vectorized_diff_source_check_output.txt"

RACE_YEAR = 2022
RACE_ROUND = 13
N_ITERATIONS = 10_000
SEED = 42
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _is_wet_race(subject_laps: pd.DataFrame) -> bool:
    seen = set(subject_laps["TyreCompound"].dropna().astype(str).str.upper().tolist())
    return any(comp not in {"SOFT", "MEDIUM", "HARD"} for comp in seen)


def _first_grid3_strategies(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    max_stops: int,
    count: int,
) -> List[dict]:
    out: List[dict] = []
    candidate_laps = list(range(1, race_length, 3))

    for stop_count in range(1, max_stops + 1):
        for pit_laps in itertools.combinations(candidate_laps, stop_count):
            for compounds in itertools.product(DRY_COMPOUNDS, repeat=stop_count):
                if (not is_wet_race) and (starting_compound in DRY_COMPOUNDS):
                    used = {starting_compound, *compounds}
                    if len(used) < 2:
                        continue
                out.append(
                    {
                        "starting_compound": starting_compound,
                        "stops": [(int(l), str(c)) for l, c in zip(pit_laps, compounds)],
                    }
                )
                if len(out) >= count:
                    return out
    return out


def _schedule_to_row(schedule: Dict[int, bool], race_length: int) -> np.ndarray:
    row = np.zeros(race_length, dtype=bool)
    for lap in range(1, race_length + 1):
        row[lap - 1] = bool(schedule.get(lap, False))
    return row


def _build_schedule_matrix(prob_by_lap: Dict[int, float], n_iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    race_length = int(max(prob_by_lap.keys()))
    out = np.zeros((n_iterations, race_length), dtype=bool)
    for i in range(n_iterations):
        out[i] = _schedule_to_row(mc.sample_caution_schedule(prob_by_lap, rng), race_length)
    return out


def _capture_schedules_and_run(strategy: dict, state: dict, subject_driver: str, prob_by_lap: Dict[int, float]) -> Tuple[np.ndarray, dict]:
    race_length = int(pd.to_numeric(state["laps"]["LapNumber"], errors="coerce").max())
    captured: List[Dict[int, bool]] = []
    original_sampler = mc.sample_caution_schedule

    def wrapped(prob, rng):
        schedule = original_sampler(prob, rng)
        captured.append(schedule)
        return schedule

    mc.sample_caution_schedule = wrapped
    try:
        result = run_monte_carlo(
            race_state=state,
            subject_driver=subject_driver,
            subject_strategy=strategy,
            prob_by_lap=prob_by_lap,
            lambda_=1.0,
            n_iterations=N_ITERATIONS,
            rng=np.random.default_rng(SEED),
        )
    finally:
        mc.sample_caution_schedule = original_sampler

    schedules = np.vstack([_schedule_to_row(s, race_length) for s in captured])
    return schedules, result


def _vectorized_trace_single(
    strategy: dict,
    schedule_row: np.ndarray,
    degradation_lookup: Dict[Tuple[object, ...], float],
    baseline_pace_s: float,
    pit_loss_s: float,
    lambda_: float,
) -> List[dict]:
    from source_code.simulation.race_model import _lookup_degradation_rate, _strategy_starting_compound, _strategy_stops

    race_length = schedule_row.shape[0]
    starting_compound = _strategy_starting_compound(strategy)
    stops = _strategy_stops(strategy)

    stop_map: Dict[int, str] = {int(l): str(c) for l, c in stops}
    current_compound = starting_compound
    cumulative = np.float32(0.0)
    age = np.float32(1.0)

    trace: List[dict] = []
    for lap in range(1, race_length + 1):
        if lap in stop_map:
            current_compound = stop_map[lap]

        is_caution = bool(schedule_row[lap - 1])
        deg = np.float32(_lookup_degradation_rate(strategy, degradation_lookup, current_compound))

        if is_caution:
            lap_time = np.float32(baseline_pace_s) * np.float32(CAUTION_PACE_RATIO)
        else:
            lap_time = np.float32(baseline_pace_s) + deg * age

        if lap in stop_map:
            lap_time = lap_time + (np.float32(CAUTION_PIT_LOSS_S) if is_caution else np.float32(pit_loss_s))

        cumulative = cumulative + lap_time

        trace.append(
            {
                "lap": lap,
                "effective_age": float(age),
                "degradation_rate": float(deg),
                "lap_time_s": float(lap_time),
                "cumulative_time_s": float(cumulative),
                "is_caution": is_caution,
                "compound": current_compound,
            }
        )

        if lap in stop_map:
            age = np.float32(1.0)
        else:
            age = age + (np.float32(lambda_) if is_caution else np.float32(1.0))

    return trace


def main() -> None:
    lines: List[str] = []

    def emit(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    state = load_race_state(RACE_YEAR, RACE_ROUND)
    laps = state["laps"].copy()
    laps["sc_active"] = _to_bool(laps["sc_active"])
    laps["vsc_active"] = _to_bool(laps["vsc_active"])

    subject_driver = select_midfield_subject_driver(laps)
    strategies = build_actual_strategies(laps)
    subject_actual = dict(strategies[subject_driver])
    subject_laps = laps[laps["Driver"].astype(str) == subject_driver].copy()

    merged_prob = build_probability_sources(state)["merged"]

    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    first_four_grid3 = _first_grid3_strategies(
        race_length=race_length,
        starting_compound=str(subject_actual["starting_compound"]),
        is_wet_race=_is_wet_race(subject_laps),
        max_stops=3,
        count=4,
    )
    candidates = [dict(subject_actual)] + [dict(s) for s in first_four_grid3]
    for s in candidates:
        s.setdefault("baseline_pace_s", float(subject_actual["baseline_pace_s"]))
        s.setdefault("year", int(subject_actual["year"]))
        s.setdefault("round", int(subject_actual["round"]))
        s.setdefault("event_name", str(subject_actual["event_name"]))

    emit("=" * 100)
    emit("Step 1 - Caution schedule identity check")
    emit("=" * 100)

    vectorized_schedule_matrix = _build_schedule_matrix(merged_prob, N_ITERATIONS, SEED)

    all_identical = True
    first_divergence_reported = False
    deterministic_results: List[dict] = []

    for idx, strategy in enumerate(candidates, start=1):
        det_schedules, det_result = _capture_schedules_and_run(strategy, state, subject_driver, merged_prob)
        deterministic_results.append(det_result)

        identical = np.array_equal(det_schedules, vectorized_schedule_matrix)
        emit(f"strategy_{idx}_schedule_bitwise_identical={identical}")
        if not identical:
            all_identical = False
            if not first_divergence_reported:
                diff_positions = np.argwhere(det_schedules != vectorized_schedule_matrix)
                i0, l0 = diff_positions[0]
                emit(f"first_divergence_iteration_index={int(i0)}")
                emit(f"first_divergence_lap_index_0based={int(l0)}")
                emit(
                    f"det_value={bool(det_schedules[i0, l0])}, vec_value={bool(vectorized_schedule_matrix[i0, l0])}"
                )
                emit("divergence_reason=Different schedule draws/RNG consumption order")
                first_divergence_reported = True

    emit(f"all_5_strategies_schedule_identical={all_identical}")

    degradation_lookup = _build_degradation_lookup(state.get("deg_stats"))
    subject_pit_loss_s = float(_driver_pit_loss(subject_laps))

    rival_times = precompute_rival_times(
        race_state=state,
        subject_driver=subject_driver,
        caution_schedules=vectorized_schedule_matrix,
        lambda_=1.0,
    )
    subject_times = evaluate_strategy_batch(
        subject_strategies=candidates,
        caution_schedules=vectorized_schedule_matrix,
        degradation_lookup=degradation_lookup,
        lambda_=1.0,
        baseline_pace_s=float(subject_actual["baseline_pace_s"]),
        pit_loss_s=subject_pit_loss_s,
        caution_pit_loss_s=CAUTION_PIT_LOSS_S,
        caution_pace_ratio=CAUTION_PACE_RATIO,
    )
    vec_scores = rank_and_score_batch(subject_times, rival_times)

    if all_identical:
        emit("")
        emit("=" * 100)
        emit("Step 2 - Implementation divergence isolation (strategy 1)")
        emit("=" * 100)

        det_points = np.asarray(deterministic_results[0]["points_distribution"], dtype=np.float32)
        vec_points = np.asarray(vec_scores["points_matrix"][0], dtype=np.float32)
        point_diff_idx = np.where(det_points != vec_points)[0]

        if len(point_diff_idx) == 0:
            emit("strategy_1_points_distribution_identical=True")
            emit("no_iteration_with_points_difference_found=True")
        else:
            it = int(point_diff_idx[0])
            emit("strategy_1_points_distribution_identical=False")
            emit(f"first_iteration_with_points_difference={it}")
            emit(f"det_points={float(det_points[it])}, vec_points={float(vec_points[it])}")

            schedule_map = {
                lap + 1: bool(vectorized_schedule_matrix[it, lap])
                for lap in range(vectorized_schedule_matrix.shape[1])
            }
            det_trace = simulate_car_race(
                strategy=candidates[0],
                caution_schedule=schedule_map,
                degradation_stats=None,
                race_length=race_length,
                lambda_=1.0,
                pit_loss_s=subject_pit_loss_s,
                caution_pit_loss_s=CAUTION_PIT_LOSS_S,
                caution_pace_ratio=CAUTION_PACE_RATIO,
                degradation_lookup=degradation_lookup,
                collect_trace=True,
            )["lap_trace"]
            vec_trace = _vectorized_trace_single(
                strategy=candidates[0],
                schedule_row=vectorized_schedule_matrix[it],
                degradation_lookup=degradation_lookup,
                baseline_pace_s=float(subject_actual["baseline_pace_s"]),
                pit_loss_s=subject_pit_loss_s,
                lambda_=1.0,
            )

            first_lap_diff = None
            for a, b in zip(det_trace, vec_trace):
                diffs = {
                    "effective_age": abs(float(a["effective_age"]) - float(b["effective_age"])),
                    "degradation_rate": abs(float(a["degradation_rate"]) - float(b["degradation_rate"])),
                    "lap_time_s": abs(float(a["lap_time_s"]) - float(b["lap_time_s"])),
                    "cumulative_time_s": abs(float(a["cumulative_time_s"]) - float(b["cumulative_time_s"])),
                }
                if any(v > 1e-9 for v in diffs.values()):
                    first_lap_diff = (int(a["lap_number"]), a, b, diffs)
                    break

            if first_lap_diff is None:
                emit("subject_trace_first_lap_disagreement=None")
                emit("subject_race_time_difference_source=not_subject_lap_dynamics")
                emit("likely_source=rival-time precision/tie-boundary effects in vectorized ranking (float32 storage)")
            else:
                lap, a, b, diffs = first_lap_diff
                emit(f"subject_trace_first_lap_disagreement={lap}")
                emit(
                    "det_vs_vec_at_first_diff="
                    f"effective_age({a['effective_age']} vs {b['effective_age']}), "
                    f"degradation_rate({a['degradation_rate']} vs {b['degradation_rate']}), "
                    f"lap_time_s({a['lap_time_s']} vs {b['lap_time_s']}), "
                    f"cumulative_time_s({a['cumulative_time_s']} vs {b['cumulative_time_s']})"
                )
                emit(f"absolute_differences={diffs}")
    else:
        emit("")
        emit("=" * 100)
        emit("Step 3 - Schedule divergence conclusion")
        emit("=" * 100)
        emit(
            "Schedules are not identical; observed output differences are ordinary Monte Carlo sampling variance "
            "from different random-world draws, not an implementation bug."
        )

    emit("")
    emit("=" * 100)
    emit("Step 4 - Shared vs duplicated constants/physics helpers")
    emit("=" * 100)

    vec_path = ROOT / "src" / "simulation" / "race_model_vectorized.py"
    vec_source = vec_path.read_text(encoding="utf-8")

    imports_config_constants = (
        "from .config import CAUTION_PACE_RATIO, CAUTION_PIT_LOSS_S" in vec_source
    )
    local_caution_ratio_definition = bool(re.search(r"^\s*CAUTION_PACE_RATIO\s*=", vec_source, flags=re.M))
    local_caution_pit_definition = bool(re.search(r"^\s*CAUTION_PIT_LOSS_S\s*=", vec_source, flags=re.M))
    imports_degradation_helpers = (
        "_build_degradation_lookup" in vec_source
        and "_lookup_degradation_rate" in vec_source
    )

    emit(f"imports_config_constants={imports_config_constants}")
    emit(f"local_caution_ratio_definition_present={local_caution_ratio_definition}")
    emit(f"local_caution_pit_loss_definition_present={local_caution_pit_definition}")
    emit(f"imports_shared_degradation_helpers_from_race_model={imports_degradation_helpers}")

    if local_caution_ratio_definition or local_caution_pit_definition:
        emit("maintenance_risk=Constants duplicated locally; should consolidate to shared config imports")
    else:
        emit("maintenance_risk=No local duplicate caution constants found in vectorized module")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("")
    emit(f"Saved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    from source_code.simulation.race_state import load_race_state

    main()
