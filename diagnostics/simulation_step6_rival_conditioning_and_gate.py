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

from src.simulation.config import (  # noqa: E402
    GRID_SPACING,
    INCLUDE_BIG_THREE_AS_RIVALS,
    LOCKED_ARCHETYPE_RACES,
    MIN_STINT_LENGTH_LAPS,
    N_ITERATIONS,
    PROXIMITY_MARGIN_S,
    PROXIMITY_MARGIN_SENSITIVITY_S,
    RNG_SEED,
    UNIDENTIFIABLE_EPSILON,
)
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
FEATURES_PATH = ROOT / "src" / "features.py"
BACKSTOP_FLOOR = 2
RUNTIME_N_ITERATIONS = 1000


def emit(text: str = "") -> None:
    print(text, flush=True)


def _load_state_with_v3(race: Mapping[str, object], v3: pd.DataFrame) -> dict:
    state = load_race_state(int(race["year"]), int(race["round"]))
    event_name = str(state["laps"]["EventName"].dropna().iloc[0])
    state["deg_stats"] = v3[(v3["Year"] == int(race["year"])) & (v3["EventName"] == event_name)].copy()
    return state


def _threat_logic_excerpt() -> List[str]:
    lines = FEATURES_PATH.read_text(encoding="utf-8").splitlines()
    start = 820
    end = 894
    out = []
    for i in range(start, end + 1):
        if i <= len(lines):
            out.append(f"{i:04d}: {lines[i - 1]}")
    return out


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


def main() -> None:
    emit("=" * 100)
    emit("Step 6 - Rival conditioning and identifiability gate")
    emit("=" * 100)
    emit(f"v3_path={V3_PATH}")
    emit(f"config_min_stint_length_laps={MIN_STINT_LENGTH_LAPS}")
    emit(f"runtime_min_stint_length_laps={BACKSTOP_FLOOR}")
    emit(f"grid_spacing={GRID_SPACING}")
    emit(f"n_iterations={N_ITERATIONS}")
    emit(f"runtime_n_iterations={RUNTIME_N_ITERATIONS}")
    emit(f"proximity_margin_s={PROXIMITY_MARGIN_S}")
    emit(f"include_big_three_as_rivals={INCLUDE_BIG_THREE_AS_RIVALS}")
    emit(f"unidentifiable_epsilon={UNIDENTIFIABLE_EPSILON}")

    v3 = pd.read_csv(V3_PATH)

    emit("")
    emit("=" * 100)
    emit("Step 1 - Threat index logic audit")
    emit("=" * 100)
    emit("features_logic_verbatim_begin")
    for line in _threat_logic_excerpt():
        emit(line)
    emit("features_logic_verbatim_end")

    emit("step1_conclusion_a_identity=Threat columns do not encode specific rival driver identity; they store only scalar index values and boolean flags.")
    emit("step1_conclusion_b_scale=Index is a dimensionless ratio age_diff/max(gap,0.5), capped at 20.0. It is not a time-gap in seconds.")
    emit("step1_conclusion_c_scope=Computed per subject-driver-lap against all nearby cars in the same lap group, then collapsed to the maximum threat value (single nearest effective threat retained implicitly by max).")
    emit(
        "step1_required_columns_for_identity_recovery=Need per-race/per-lap rows with Driver, Team, Position, LapTime_s, TyreLife, Year, Round, LapNumber to reconstruct which rival(s) satisfied the threat condition."
    )

    emit("")
    emit("=" * 100)
    emit("Step 2 - Rival set definition and sanity samples")
    emit("=" * 100)

    hungary = next(r for r in LOCKED_ARCHETYPE_RACES if r["event_name"] == "Hungarian Grand Prix" and int(r["year"]) == 2022)
    hungary_state = _load_state_with_v3(hungary, v3)
    hungary_subject = select_midfield_subject_driver(hungary_state["laps"])
    hungary_triggers = detect_rival_trigger_events(
        race_state=hungary_state,
        subject_driver=hungary_subject,
        proximity_margin_s=PROXIMITY_MARGIN_S,
        include_big_three_as_rivals=INCLUDE_BIG_THREE_AS_RIVALS,
    )

    basis = hungary_triggers["threat_basis_by_lap"]
    sample_laps = [int(k) for k, v in basis.items() if int(v.get("rival_set_size", 0)) > 0][:3]
    if len(sample_laps) < 3:
        sample_laps = sorted(list(basis.keys()))[:3]

    emit(f"step2_subject_driver={hungary_subject}")
    emit(f"step2_include_big_three_as_rivals={INCLUDE_BIG_THREE_AS_RIVALS}")
    for lap in sample_laps:
        row = basis[int(lap)]
        emit("step2_rival_set_sample=" + json.dumps({
            "race": hungary,
            "subject_driver": hungary_subject,
            "lap": int(lap),
            "undercut_threat_flag": bool(row.get("undercut_threat_flag", False)),
            "overcut_threat_flag": bool(row.get("overcut_threat_flag", False)),
            "undercut_threat_index": row.get("undercut_threat_index"),
            "overcut_threat_index": row.get("overcut_threat_index"),
            "rivals": row.get("rival_set", []),
        }, sort_keys=True, default=str))

    emit("")
    emit("=" * 100)
    emit("Step 3 - Trigger detection (Hungary 2022)")
    emit("=" * 100)
    emit("hungary_trigger_events_begin")
    for evt in hungary_triggers["trigger_events"]:
        emit(json.dumps(evt, sort_keys=True, default=str))
    emit("hungary_trigger_events_end")

    emit("")
    emit("=" * 100)
    emit("Step 4 - Augmented set size for benchmark (b)")
    emit("=" * 100)

    rng_hu = np.random.default_rng(RNG_SEED)
    hu_benchmarks = compute_benchmarks_for_trigger_events(
        race_state=hungary_state,
        subject_driver=hungary_subject,
        trigger_events=hungary_triggers["trigger_events"],
        prob_source=PROB_SOURCE,
        lambda_=LAMBDA_VALUE,
        n_iterations=RUNTIME_N_ITERATIONS,
        rng=rng_hu,
        min_stint_length_laps=BACKSTOP_FLOOR,
    )

    emit("hungary_augmented_size_summary=" + json.dumps({
        "race": hungary,
        "subject_driver": hungary_subject,
        "standard_count_a": hu_benchmarks["candidate_sets"]["standard_count"],
        "reactive_only_count": hu_benchmarks["candidate_sets"]["reactive_only_count"],
        "augmented_count_b": hu_benchmarks["candidate_sets"]["augmented_count"],
        "trigger_laps": hu_benchmarks["candidate_sets"]["trigger_laps"],
        "augmented_laps_used": hu_benchmarks["candidate_sets"]["augmented_laps_used"],
        "skipped_already_grid_laps": hu_benchmarks["candidate_sets"]["skipped_already_grid_laps"],
    }, sort_keys=True, default=str))

    emit("")
    emit("=" * 100)
    emit("Step 5 - Compute (a), (b), R_anchoring for Hungary 2022")
    emit("=" * 100)

    bench_a = hu_benchmarks["benchmark_a"]
    bench_b = hu_benchmarks["benchmark_b"]
    bench_r = hu_benchmarks["benchmark_r_anchoring"]

    emit("hungary_benchmark_a=" + json.dumps(_serialize_benchmark(bench_a), sort_keys=True, default=str))
    emit("hungary_benchmark_b=" + json.dumps(_serialize_benchmark(bench_b), sort_keys=True, default=str))
    emit("hungary_benchmark_r_anchoring=" + json.dumps(_serialize_benchmark(bench_r), sort_keys=True, default=str))

    b_dominates_a = False
    if (bench_a is not None) and (bench_b is not None):
        b_dominates_a = float(bench_b["mean_points"]) >= float(bench_a["mean_points"]) - 1e-12
    emit(f"hungary_b_weakly_dominates_a={b_dominates_a}")
    if not b_dominates_a:
        emit("hungary_dominance_bug=Benchmark (b) should weakly dominate (a), but mean_points(b) < mean_points(a).")

    emit("")
    emit("=" * 100)
    emit("Step 6 - Full 12-race identifiability gate")
    emit("=" * 100)

    gate_rows: List[Dict[str, object]] = []
    trigger_count_by_archetype: Dict[str, int] = defaultdict(int)
    total_trigger_events = 0

    for race in LOCKED_ARCHETYPE_RACES:
        state = _load_state_with_v3(race, v3)
        subject = select_midfield_subject_driver(state["laps"])
        trigger_data = detect_rival_trigger_events(
            race_state=state,
            subject_driver=subject,
            proximity_margin_s=PROXIMITY_MARGIN_S,
            include_big_three_as_rivals=INCLUDE_BIG_THREE_AS_RIVALS,
        )

        events = trigger_data["trigger_events"]
        trigger_count_by_archetype[str(race["archetype"])] += int(len(events))
        total_trigger_events += int(len(events))

        emit("gate_race_trigger_summary=" + json.dumps({
            "race": race,
            "subject_driver": subject,
            "trigger_event_count": len(events),
        }, sort_keys=True, default=str))

        if not events:
            continue

        # Score the race once over the union of standard + all reactive candidates.
        sets_all = build_reactive_candidate_sets(
            race_state=state,
            subject_driver=subject,
            trigger_events=events,
            min_stint_length_laps=BACKSTOP_FLOOR,
            grid_spacing=GRID_SPACING,
        )
        augmented = list(sets_all["augmented_strategies"])
        standard_count = int(sets_all["standard_count"])
        reactive = list(sets_all["reactive_only_strategies"])

        rng_race = np.random.default_rng(RNG_SEED)
        scored = score_subject_strategy_pool(
            race_state=state,
            subject_driver=subject,
            strategies=augmented,
            prob_source=PROB_SOURCE,
            lambda_=LAMBDA_VALUE,
            n_iterations=RUNTIME_N_ITERATIONS,
            rng=rng_race,
        )

        mean_points = np.asarray(scored["mean_points"], dtype=np.float64)

        standard_indices = np.arange(standard_count, dtype=np.int64)
        a_best = float(np.max(mean_points[standard_indices])) if standard_indices.size else math.nan

        standard_candidate_laps = set(range(1, int(sets_all["race_length"]), GRID_SPACING))

        # Map each reactive added lap -> indices in augmented pool.
        reactive_indices_by_added_lap: Dict[int, List[int]] = defaultdict(list)
        for local_idx, strategy in enumerate(reactive, start=standard_count):
            stops = [(int(l), str(c)) for l, c in strategy.get("stops", [])]
            extra_laps = sorted({lap for lap, _ in stops if lap not in standard_candidate_laps})
            if len(extra_laps) != 1:
                continue
            reactive_indices_by_added_lap[int(extra_laps[0])].append(int(local_idx))

        for event in events:
            added_lap = int(event["lap"]) + 1
            reactive_idx = np.asarray(reactive_indices_by_added_lap.get(added_lap, []), dtype=np.int64)
            if reactive_idx.size == 0:
                diff = math.nan
                identifiable = False
                computable = False
                b_value = math.nan
                r_value = math.nan
            else:
                r_value = float(np.max(mean_points[reactive_idx]))
                b_value = float(max(a_best, r_value)) if np.isfinite(a_best) else r_value
                diff = abs(b_value - r_value)
                identifiable = bool(diff < float(UNIDENTIFIABLE_EPSILON))
                computable = True

            row = {
                "race": race,
                "subject_driver": subject,
                "trigger_lap": int(event["lap"]),
                "triggered_by": event["triggered_by"],
                "b_mean_points": b_value,
                "r_anchoring_mean_points": r_value,
                "abs_B_minus_R": diff,
                "identifiable_strict": identifiable,
                "computable": computable,
                "reactive_only_count": int(reactive_idx.size),
            }
            gate_rows.append(row)

    emit("gate_trigger_rows_begin")
    for row in gate_rows:
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("gate_trigger_rows_end")

    gate_df = pd.DataFrame(gate_rows)
    computable_df = gate_df[gate_df["computable"].eq(True)].copy() if not gate_df.empty else gate_df.copy()

    diffs = computable_df["abs_B_minus_R"].dropna().astype(float).values if not computable_df.empty else np.array([])
    if diffs.size > 0:
        min_diff = float(np.min(diffs))
        median_diff = float(np.median(diffs))
        max_diff = float(np.max(diffs))
        pct = {
            "p05": float(np.percentile(diffs, 5)),
            "p25": float(np.percentile(diffs, 25)),
            "p50": float(np.percentile(diffs, 50)),
            "p75": float(np.percentile(diffs, 75)),
            "p95": float(np.percentile(diffs, 95)),
        }
        bins = np.array([0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0, np.inf])
        hist_counts, _ = np.histogram(diffs, bins=bins)
        hist_rows = []
        for i in range(len(hist_counts)):
            lo = bins[i]
            hi = bins[i + 1]
            hist_rows.append({"bin_lo": float(lo), "bin_hi": float(hi) if np.isfinite(hi) else "inf", "count": int(hist_counts[i])})
    else:
        min_diff = median_diff = max_diff = math.nan
        pct = {"p05": math.nan, "p25": math.nan, "p50": math.nan, "p75": math.nan, "p95": math.nan}
        hist_rows = []

    identifiable_count = int(computable_df["identifiable_strict"].sum()) if not computable_df.empty else 0
    computable_count = int(len(computable_df))
    identifiable_fraction = float(identifiable_count / computable_count) if computable_count else math.nan

    emit(f"gate_total_trigger_events={total_trigger_events}")
    emit(f"gate_computable_trigger_events={computable_count}")
    emit(f"gate_abs_diff_min={min_diff}")
    emit(f"gate_abs_diff_median={median_diff}")
    emit(f"gate_abs_diff_max={max_diff}")
    emit("gate_abs_diff_percentiles=" + json.dumps(pct, sort_keys=True, default=str))
    emit("gate_abs_diff_histogram_begin")
    for row in hist_rows:
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("gate_abs_diff_histogram_end")
    emit(f"gate_identifiable_count_strict={identifiable_count}")
    emit(f"gate_identifiable_fraction_strict={identifiable_fraction}")

    emit("gate_trigger_count_by_archetype=" + json.dumps({
        "Street": int(trigger_count_by_archetype.get("Street", 0)),
        "Power": int(trigger_count_by_archetype.get("Power", 0)),
        "Technical": int(trigger_count_by_archetype.get("Technical", 0)),
    }, sort_keys=True, default=str))

    emit("")
    emit("=" * 100)
    emit("Step 7 - Proximity margin sensitivity (trigger-count only)")
    emit("=" * 100)

    sensitivity = []
    for margin in PROXIMITY_MARGIN_SENSITIVITY_S:
        margin_total = 0
        for race in LOCKED_ARCHETYPE_RACES:
            state = _load_state_with_v3(race, v3)
            subject = select_midfield_subject_driver(state["laps"])
            td = detect_rival_trigger_events(
                race_state=state,
                subject_driver=subject,
                proximity_margin_s=float(margin),
                include_big_three_as_rivals=INCLUDE_BIG_THREE_AS_RIVALS,
            )
            margin_total += int(len(td["trigger_events"]))
        sensitivity.append({"proximity_margin_s": float(margin), "trigger_event_count": int(margin_total)})

    emit("step7_sensitivity_counts_begin")
    for row in sensitivity:
        emit(json.dumps(row, sort_keys=True, default=str))
    emit("step7_sensitivity_counts_end")


if __name__ == "__main__":
    main()
