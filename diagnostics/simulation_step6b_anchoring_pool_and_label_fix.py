from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import config as sim_config  # noqa: E402
from src.simulation.monte_carlo import select_midfield_subject_driver  # noqa: E402
from src.simulation.optimiser import (  # noqa: E402
    build_reactive_candidate_sets,
    compute_benchmarks_for_trigger_events,
    detect_rival_trigger_events,
    score_subject_strategy_pool,
)
from src.simulation.race_state import load_race_state  # noqa: E402

PROB_SOURCE = "lap_level"
LAMBDA_VALUE = 1.0
V3_PATH = ROOT / "data" / "diagnostics" / "deg_rate_corrected_shrunk_v3_full_peryear_stats.csv"
# Runtime tradeoff for this diagnostics gate only. The committed default in config
# remains the validated value; this cap keeps the 12-race gate tractable.
GATE_RUNTIME_N_ITERATIONS = 1000


def emit(text: str = "") -> None:
    print(text, flush=True)


def _load_state_with_v3(race: Mapping[str, object], v3: pd.DataFrame) -> dict:
    state = load_race_state(int(race["year"]), int(race["round"]))
    event_name = str(state["laps"]["EventName"].dropna().iloc[0])
    state["deg_stats"] = v3[(v3["Year"] == int(race["year"])) & (v3["EventName"] == event_name)].copy()
    return state


def _serialize_benchmark(bench: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if bench is None:
        return None
    return {
        "strategy": bench["strategy"],
        "mean_points": float(bench["mean_points"]),
        "win_rate": float(bench["win_rate"]),
        "mean_rank": float(bench["mean_rank"]),
        "std_points": float(bench["std_points"]),
    }


def _strategy_has_stop_lap(strategy: Mapping[str, object], lap: int) -> bool:
    target = int(lap)
    return any(int(stop_lap) == target for stop_lap, _ in strategy.get("stops", []))


def main() -> None:
    emit("=" * 100)
    emit("Step 6b - Anchoring pool and labeling fix")
    emit("=" * 100)

    emit("step1_config_values_begin")
    emit(f"config_path={Path(sim_config.__file__).resolve()}")
    emit(f"config_min_stint_length_laps={sim_config.MIN_STINT_LENGTH_LAPS}")
    emit(f"config_n_iterations={sim_config.N_ITERATIONS}")
    emit("step1_config_values_end")

    min_stint_is_2 = int(sim_config.MIN_STINT_LENGTH_LAPS) == 2
    emit(f"step1_min_stint_is_2={min_stint_is_2}")
    emit(
        "step1_runtime_iteration_choice="
        "Using gate_runtime_n_iterations=1000 as a deliberate diagnostics runtime tradeoff; "
        "the validated committed default remains config_n_iterations=10000."
    )
    emit(f"gate_runtime_n_iterations={GATE_RUNTIME_N_ITERATIONS}")

    v3 = pd.read_csv(V3_PATH)

    emit("")
    emit("=" * 100)
    emit("Step 2 - R_anchoring candidate pool fix and Hungary L=3 check")
    emit("=" * 100)

    hungary = next(
        r
        for r in sim_config.LOCKED_ARCHETYPE_RACES
        if r["event_name"] == "Hungarian Grand Prix" and int(r["year"]) == 2022
    )
    hungary_state = _load_state_with_v3(hungary, v3)
    hungary_subject = select_midfield_subject_driver(hungary_state["laps"])
    hungary_triggers = detect_rival_trigger_events(
        race_state=hungary_state,
        subject_driver=hungary_subject,
        proximity_margin_s=sim_config.PROXIMITY_MARGIN_S,
        include_big_three_as_rivals=sim_config.INCLUDE_BIG_THREE_AS_RIVALS,
    )
    hu_benchmarks = compute_benchmarks_for_trigger_events(
        race_state=hungary_state,
        subject_driver=hungary_subject,
        trigger_events=hungary_triggers["trigger_events"],
        prob_source=PROB_SOURCE,
        lambda_=LAMBDA_VALUE,
        n_iterations=GATE_RUNTIME_N_ITERATIONS,
        rng=np.random.default_rng(sim_config.RNG_SEED),
        min_stint_length_laps=int(sim_config.MIN_STINT_LENGTH_LAPS),
    )

    r_by_lap = hu_benchmarks.get("benchmark_r_anchoring_by_trigger_lap", {})
    h_l3 = r_by_lap.get(3)
    hu_augmented = list(hu_benchmarks["candidate_sets"]["augmented_strategies"])
    hu_l4_count = sum(1 for s in hu_augmented if _strategy_has_stop_lap(s, 4))

    emit("hungary_benchmark_a=" + json.dumps(_serialize_benchmark(hu_benchmarks["benchmark_a"]), sort_keys=True, default=str))
    emit("hungary_benchmark_b=" + json.dumps(_serialize_benchmark(hu_benchmarks["benchmark_b"]), sort_keys=True, default=str))
    emit("hungary_benchmark_r_anchoring=" + json.dumps(_serialize_benchmark(hu_benchmarks["benchmark_r_anchoring"]), sort_keys=True, default=str))
    emit("hungary_trigger_l3_r_anchoring=" + json.dumps(_serialize_benchmark(h_l3), sort_keys=True, default=str))
    emit(f"hungary_trigger_l3_added_lap=4")
    emit(f"hungary_trigger_l3_pool_count_stop_at_4={hu_l4_count}")
    emit(f"hungary_trigger_l3_computable={h_l3 is not None}")

    emit("")
    emit("=" * 100)
    emit("Step 3/4 - Full 12-race gate with corrected unidentifiable labeling")
    emit("=" * 100)

    gate_rows: List[Dict[str, object]] = []
    trigger_count_by_archetype: Dict[str, int] = defaultdict(int)
    total_trigger_events = 0

    for race in sim_config.LOCKED_ARCHETYPE_RACES:
        state = _load_state_with_v3(race, v3)
        subject = select_midfield_subject_driver(state["laps"])
        trigger_data = detect_rival_trigger_events(
            race_state=state,
            subject_driver=subject,
            proximity_margin_s=sim_config.PROXIMITY_MARGIN_S,
            include_big_three_as_rivals=sim_config.INCLUDE_BIG_THREE_AS_RIVALS,
        )
        events = trigger_data["trigger_events"]
        trigger_count_by_archetype[str(race["archetype"])] += int(len(events))
        total_trigger_events += int(len(events))

        emit(
            "gate_race_trigger_summary="
            + json.dumps(
                {
                    "race": race,
                    "subject_driver": subject,
                    "trigger_event_count": len(events),
                },
                sort_keys=True,
                default=str,
            )
        )

        if not events:
            continue

        sets_all = build_reactive_candidate_sets(
            race_state=state,
            subject_driver=subject,
            trigger_events=events,
            min_stint_length_laps=int(sim_config.MIN_STINT_LENGTH_LAPS),
            grid_spacing=sim_config.GRID_SPACING,
        )
        augmented = list(sets_all["augmented_strategies"])
        scored = score_subject_strategy_pool(
            race_state=state,
            subject_driver=subject,
            strategies=augmented,
            prob_source=PROB_SOURCE,
            lambda_=LAMBDA_VALUE,
            n_iterations=GATE_RUNTIME_N_ITERATIONS,
            rng=np.random.default_rng(sim_config.RNG_SEED),
        )

        mean_points = np.asarray(scored["mean_points"], dtype=np.float64)
        b_value = float(np.max(mean_points)) if mean_points.size else math.nan

        for event in events:
            added_lap = int(event["lap"]) + 1
            r_idx = [idx for idx, strategy in enumerate(augmented) if _strategy_has_stop_lap(strategy, added_lap)]
            if not r_idx:
                r_value = math.nan
                diff = math.nan
                computable = False
                unidentifiable = False
            else:
                r_value = float(np.max(mean_points[np.asarray(r_idx, dtype=np.int64)]))
                diff = abs(b_value - r_value)
                computable = True
                unidentifiable = bool(diff < float(sim_config.UNIDENTIFIABLE_EPSILON))

            gate_rows.append(
                {
                    "race": race,
                    "subject_driver": subject,
                    "trigger_lap": int(event["lap"]),
                    "triggered_by": event["triggered_by"],
                    "added_lap_for_r": int(added_lap),
                    "b_mean_points": b_value,
                    "r_anchoring_mean_points": r_value,
                    "abs_B_minus_R": diff,
                    "unidentifiable_strict": unidentifiable,
                    "computable": computable,
                    "r_pool_count": int(len(r_idx)),
                }
            )

    emit("gate_trigger_rows_begin")
    for row in gate_rows:
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("gate_trigger_rows_end")

    gate_df = pd.DataFrame(gate_rows)
    computable_df = gate_df[gate_df["computable"].eq(True)].copy() if not gate_df.empty else gate_df.copy()
    identifiable_df = computable_df[computable_df["unidentifiable_strict"].eq(False)].copy() if not computable_df.empty else computable_df.copy()

    diffs_identifiable = (
        identifiable_df["abs_B_minus_R"].dropna().astype(float).values
        if not identifiable_df.empty
        else np.array([])
    )

    if diffs_identifiable.size > 0:
        min_diff = float(np.min(diffs_identifiable))
        median_diff = float(np.median(diffs_identifiable))
        max_diff = float(np.max(diffs_identifiable))
        pct = {
            "p05": float(np.percentile(diffs_identifiable, 5)),
            "p25": float(np.percentile(diffs_identifiable, 25)),
            "p50": float(np.percentile(diffs_identifiable, 50)),
            "p75": float(np.percentile(diffs_identifiable, 75)),
            "p95": float(np.percentile(diffs_identifiable, 95)),
        }
    else:
        min_diff = median_diff = max_diff = math.nan
        pct = {"p05": math.nan, "p25": math.nan, "p50": math.nan, "p75": math.nan, "p95": math.nan}

    total_count = int(total_trigger_events)
    computable_count = int(len(computable_df))
    computable_rate = float(computable_count / total_count) if total_count else math.nan

    unidentifiable_count = int(computable_df["unidentifiable_strict"].sum()) if not computable_df.empty else 0
    unidentifiable_fraction_computable = float(unidentifiable_count / computable_count) if computable_count else math.nan
    unidentifiable_fraction_total = float(unidentifiable_count / total_count) if total_count else math.nan

    emit(f"gate_total_trigger_events={total_count}")
    emit(f"gate_computable_trigger_events={computable_count}")
    emit(f"gate_computable_trigger_rate={computable_rate}")
    emit(f"gate_unidentifiable_count_strict={unidentifiable_count}")
    emit(f"gate_unidentifiable_fraction_computable={unidentifiable_fraction_computable}")
    emit(f"gate_unidentifiable_fraction_total={unidentifiable_fraction_total}")
    emit(f"gate_identifiable_computable_count={int(len(identifiable_df))}")
    emit(f"gate_abs_diff_identifiable_only_min={min_diff}")
    emit(f"gate_abs_diff_identifiable_only_median={median_diff}")
    emit(f"gate_abs_diff_identifiable_only_max={max_diff}")
    emit("gate_abs_diff_identifiable_only_percentiles=" + json.dumps(pct, sort_keys=True, default=str))

    emit(
        "gate_trigger_count_by_archetype="
        + json.dumps(
            {
                "Power": int(trigger_count_by_archetype.get("Power", 0)),
                "Street": int(trigger_count_by_archetype.get("Street", 0)),
                "Technical": int(trigger_count_by_archetype.get("Technical", 0)),
            },
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
