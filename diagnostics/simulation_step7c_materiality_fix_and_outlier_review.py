from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import config as sim_config  # noqa: E402
from src.simulation.bias import compute_r_b  # noqa: E402
from src.simulation.monte_carlo import build_actual_strategies  # noqa: E402
from src.simulation.optimiser import (  # noqa: E402
    argmax_strategy,
    build_reactive_candidate_sets,
    detect_rival_trigger_events,
    score_subject_strategy_pool,
)
from src.simulation.race_state import load_race_state  # noqa: E402
from src.simulation.references import r_conservatism  # noqa: E402


DIAG_DIR = ROOT / "data" / "diagnostics"
OUTPUT_PATH = DIAG_DIR / "simulation_step7c_materiality_fix_and_outlier_review_output.txt"
CACHE_PATH = DIAG_DIR / "phase3_pilot_reference_cache.json"
DEG_CORRECTED_PATH = DIAG_DIR / "deg_rate_corrected_shrunk_v3_full_peryear_stats.csv"

N_ITERATIONS = 10_000
EPSILON = float(sim_config.UNIDENTIFIABLE_EPSILON)
MATERIALITY = float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R)
LAMBDA_VALUE = 1.0


def emit(out_handle, text: str = "") -> None:
    print(text, flush=True)
    out_handle.write(text + "\n")
    out_handle.flush()


def _race_key(race: Mapping[str, Any], subject_driver: str) -> Tuple[int, int, str, str]:
    return (int(race["year"]), int(race["round"]), str(race["event_name"]), str(subject_driver))


def _first_stop_lap(strategy: Mapping[str, Any]) -> int | None:
    stops = strategy.get("stops", [])
    if not stops:
        return None
    return int(stops[0][0])


def _load_race_state_with_corrected_deg(race: Mapping[str, Any]) -> dict:
    state = load_race_state(int(race["year"]), int(race["round"]))
    if DEG_CORRECTED_PATH.exists():
        deg = pd.read_csv(DEG_CORRECTED_PATH)
        event_name = str(state["laps"]["EventName"].dropna().iloc[0])
        state["deg_stats"] = deg[(deg["Year"] == int(race["year"])) & (deg["EventName"] == event_name)].copy()
    return state


def _subject_context(race_state: Mapping[str, Any], subject_driver: str) -> Dict[str, Any]:
    laps = race_state["laps"].copy()
    sub = laps[laps["Driver"].astype(str) == str(subject_driver)].copy()
    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())

    max_lap = int(pd.to_numeric(sub["LapNumber"], errors="coerce").max()) if not sub.empty else None
    final_position = None
    if not sub.empty and "Position" in sub.columns:
        sorted_sub = sub.sort_values("LapNumber")
        try:
            final_position = int(pd.to_numeric(sorted_sub["Position"], errors="coerce").dropna().iloc[-1])
        except Exception:
            final_position = None

    compound_col = "Compound" if "Compound" in sub.columns else None
    used_wet = False
    if compound_col is not None:
        used_wet = bool(sub[compound_col].astype(str).str.upper().isin(["WET", "INTERMEDIATE"]).any())

    caution_laps = int((laps["sc_active"].astype(bool) | laps["vsc_active"].astype(bool)).sum())

    return {
        "race_length_laps": race_length,
        "subject_max_lap": max_lap,
        "completed_full_distance": bool(max_lap is not None and max_lap >= race_length),
        "observed_final_position": final_position,
        "used_wet_or_intermediate": used_wet,
        "caution_lap_count": caution_laps,
    }


def _compute_reference_cache_for_race(race: Mapping[str, Any]) -> Dict[str, Any]:
    state = _load_race_state_with_corrected_deg(race)
    laps = state["laps"]
    subject_driver = str(race["subject_driver"])

    actual_by_driver = build_actual_strategies(laps)
    if subject_driver not in actual_by_driver:
        raise ValueError(f"Subject driver {subject_driver} missing from actual strategies for {race}")

    actual_strategy = dict(actual_by_driver[subject_driver])
    race_length_laps = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())

    # Use the vectorized strategy scoring path for X on a single actual strategy.
    x_scored = score_subject_strategy_pool(
        race_state=state,
        subject_driver=subject_driver,
        strategies=[actual_strategy],
        prob_source="lap_level",
        lambda_=LAMBDA_VALUE,
        n_iterations=N_ITERATIONS,
        rng=np.random.default_rng(sim_config.RNG_SEED),
    )
    x_mean_points = float(np.asarray(x_scored["mean_points"], dtype=np.float64)[0])

    b_result = argmax_strategy(
        race_state=state,
        subject_driver=subject_driver,
        prob_source="lap_level",
        information_set="unconditioned",
        lambda_=LAMBDA_VALUE,
        n_iterations=N_ITERATIONS,
        rng=np.random.default_rng(sim_config.RNG_SEED),
        epsilon=0.01,
    )

    r_sc_result = argmax_strategy(
        race_state=state,
        subject_driver=subject_driver,
        prob_source="static_prior",
        information_set="unconditioned",
        lambda_=LAMBDA_VALUE,
        n_iterations=N_ITERATIONS,
        rng=np.random.default_rng(sim_config.RNG_SEED),
        epsilon=0.01,
    )

    trigger_data = detect_rival_trigger_events(
        race_state=state,
        subject_driver=subject_driver,
        proximity_margin_s=sim_config.PROXIMITY_MARGIN_S,
        include_big_three_as_rivals=sim_config.INCLUDE_BIG_THREE_AS_RIVALS,
    )
    events = trigger_data["trigger_events"]

    candidate_sets = build_reactive_candidate_sets(
        race_state=state,
        subject_driver=subject_driver,
        trigger_events=events,
        min_stint_length_laps=int(sim_config.MIN_STINT_LENGTH_LAPS),
        grid_spacing=sim_config.GRID_SPACING,
    )
    augmented = list(candidate_sets["augmented_strategies"])

    scored = score_subject_strategy_pool(
        race_state=state,
        subject_driver=subject_driver,
        strategies=augmented,
        prob_source="lap_level",
        lambda_=LAMBDA_VALUE,
        n_iterations=N_ITERATIONS,
        rng=np.random.default_rng(sim_config.RNG_SEED),
    )
    mean_points = np.asarray(scored["mean_points"], dtype=np.float64)
    b_anchor = float(np.max(mean_points)) if mean_points.size else math.nan

    anchoring_pairs: List[Dict[str, Any]] = []
    for event in events:
        trigger_lap = int(event["lap"])
        added_lap = trigger_lap + 1
        r_idx = [
            idx
            for idx, strategy in enumerate(augmented)
            if any(int(stop_lap) == int(added_lap) for stop_lap, _ in strategy.get("stops", []))
        ]
        if not r_idx:
            r_value = math.nan
            computable = False
        else:
            r_value = float(np.max(mean_points[np.asarray(r_idx, dtype=np.int64)]))
            computable = True

        anchoring_pairs.append(
            {
                "trigger_lap": trigger_lap,
                "triggered_by": event["triggered_by"],
                "added_lap_for_r": added_lap,
                "B_anchoring_mean_points": b_anchor,
                "R_anchoring_mean_points": r_value,
                "computable": computable,
                "r_pool_count": int(len(r_idx)),
            }
        )

    row = {
        "race": {
            "year": int(race["year"]),
            "round": int(race["round"]),
            "event_name": str(race["event_name"]),
            "archetype": str(race["archetype"]),
        },
        "subject_driver": subject_driver,
        "race_length_laps": race_length_laps,
        "context": _subject_context(state, subject_driver),
        "X_actual_strategy": actual_strategy,
        "X_actual_first_pit_lap": _first_stop_lap(actual_strategy),
        "X_actual_mean_points": x_mean_points,
        "B_unconditioned_strategy": dict(b_result["best_strategy"]),
        "B_unconditioned_first_pit_lap": _first_stop_lap(dict(b_result["best_strategy"])),
        "B_unconditioned_mean_points": float(b_result["best_mean_points"]),
        "R_SC_static_prior_strategy": dict(r_sc_result["best_strategy"]),
        "R_SC_static_prior_first_pit_lap": _first_stop_lap(dict(r_sc_result["best_strategy"])),
        "R_SC_static_prior_mean_points": float(r_sc_result["best_mean_points"]),
        "anchoring_pairs": anchoring_pairs,
    }

    return row


def _rb_stats(rows: List[Mapping[str, Any]]) -> Dict[str, float]:
    vals = [float(r["r_b"]) for r in rows if not pd.isna(r.get("r_b"))]
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"min": math.nan, "median": math.nan, "max": math.nan}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _summarize_bias(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    strict_count = int(sum(bool(r.get("unidentifiable_strict", False)) for r in rows))
    materiality_count = int(sum(bool(r.get("unidentifiable_materiality", False)) for r in rows))
    identifiable = [
        r
        for r in rows
        if (not bool(r.get("unidentifiable_strict", False)))
        and (not bool(r.get("unidentifiable_materiality", False)))
        and (not pd.isna(r.get("r_b")))
    ]
    return {
        "total_rows": total,
        "strict_unidentifiable_count": strict_count,
        "materiality_unidentifiable_count": materiality_count,
        "materially_identifiable_count": len(identifiable),
        "r_b_distribution_materially_identifiable": _rb_stats(identifiable),
    }


def _context_tags(context: Mapping[str, Any], x_actual_mean_points: float) -> List[str]:
    tags: List[str] = []
    if not bool(context.get("completed_full_distance", True)):
        tags.append("dnf_or_not_full_distance")
    if bool(context.get("used_wet_or_intermediate", False)):
        tags.append("wet_or_intermediate_complication")
    final_pos = context.get("observed_final_position")
    if isinstance(final_pos, int) and final_pos >= 15:
        tags.append("poor_finish_position")
    if float(x_actual_mean_points) <= 1.0:
        tags.append("very_low_actual_expected_points")
    if not tags:
        tags.append("no_obvious_context_marker")
    return tags


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        emit(out, "=" * 100)
        emit(out, "Step 7c - Materiality fix and outlier review")
        emit(out, "=" * 100)
        emit(out, f"n_iterations={N_ITERATIONS}")
        emit(out, f"epsilon_strict={EPSILON}")
        emit(out, f"materiality_threshold_abs_B_minus_R={MATERIALITY}")

        emit(out, "")
        emit(out, "Step 1+2 - Rebuild full pilot reference cache at n=10000")
        cache_rows: List[Dict[str, Any]] = []
        race_start = time.perf_counter()
        worker_count = max(1, min(4, (os.cpu_count() or 1)))
        emit(out, f"parallel_workers={worker_count}")

        race_inputs: List[Dict[str, Any]] = []
        race_index: Dict[Tuple[int, int, str, str], int] = {}

        for idx, race in enumerate(sim_config.LOCKED_ARCHETYPE_RACES):
            race_info = dict(race)
            if not race_info.get("subject_driver"):
                # Keep driver mapping aligned with prior Step 7 artifacts.
                from src.simulation.monte_carlo import select_midfield_subject_driver

                st = _load_race_state_with_corrected_deg(race_info)
                race_info["subject_driver"] = str(select_midfield_subject_driver(st["laps"]))

            key = _race_key(race_info, str(race_info["subject_driver"]))
            race_index[key] = idx
            race_inputs.append(race_info)

        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            future_to_key = {
                pool.submit(_compute_reference_cache_for_race, race_info): _race_key(race_info, str(race_info["subject_driver"]))
                for race_info in race_inputs
            }

            ordered: List[Dict[str, Any] | None] = [None] * len(race_inputs)
            for future in as_completed(future_to_key):
                row = future.result()
                key = _race_key(row["race"], row["subject_driver"])
                ordered[race_index[key]] = row
                emit(
                    out,
                    "pilot_cache_row_summary="
                    + json.dumps(
                        {
                            "race": row["race"],
                            "subject_driver": row["subject_driver"],
                            "X_actual_mean_points": row["X_actual_mean_points"],
                            "B_unconditioned_mean_points": row["B_unconditioned_mean_points"],
                            "R_SC_static_prior_mean_points": row["R_SC_static_prior_mean_points"],
                            "anchoring_pair_count": len(row["anchoring_pairs"]),
                        },
                        sort_keys=True,
                        default=str,
                    ),
                )

        cache_rows = [row for row in ordered if row is not None]

        wall_clock_s = time.perf_counter() - race_start
        CACHE_PATH.write_text(json.dumps(cache_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        emit(out, f"pilot_cache_written={CACHE_PATH}")
        emit(out, f"pilot_cache_row_count={len(cache_rows)}")
        emit(out, f"pilot_cache_recompute_wall_clock_seconds={wall_clock_s:.3f}")

        emit(out, "")
        emit(out, "Step 3 - Recompute Conservatism, Anchoring, SC with strict+materiality flags")

        conservatism_rows: List[Dict[str, Any]] = []
        anchoring_rows: List[Dict[str, Any]] = []
        sc_rows: List[Dict[str, Any]] = []

        for row in cache_rows:
            race = row["race"]
            subject_driver = row["subject_driver"]

            cons_ref = r_conservatism(
                {"race_length_laps": int(row["race_length_laps"]), "subject_final_compound": None},
                str(subject_driver),
            )
            X_lap = row["X_actual_first_pit_lap"]
            B_lap = row["B_unconditioned_first_pit_lap"]
            R_lap = cons_ref["reference_lap"]
            if X_lap is not None and B_lap is not None and R_lap is not None:
                m = compute_r_b(float(X_lap), float(B_lap), float(R_lap), EPSILON, MATERIALITY)
                conservatism_rows.append(
                    {
                        "race": race,
                        "subject_driver": subject_driver,
                        "X": float(X_lap),
                        "B": float(B_lap),
                        "R": float(R_lap),
                        "abs_B_minus_R": float(m["abs_B_minus_R_b"]),
                        "r_b": m["r_b"],
                        "unidentifiable_strict": bool(m["unidentifiable_strict"]),
                        "unidentifiable_materiality": bool(m["unidentifiable_materiality"]),
                    }
                )

            X_points = float(row["X_actual_mean_points"])
            B_sc = float(row["B_unconditioned_mean_points"])
            R_sc = float(row["R_SC_static_prior_mean_points"])
            m_sc = compute_r_b(X_points, B_sc, R_sc, EPSILON, MATERIALITY)
            sc_rows.append(
                {
                    "race": race,
                    "subject_driver": subject_driver,
                    "X": X_points,
                    "B": B_sc,
                    "R": R_sc,
                    "abs_B_minus_R": float(m_sc["abs_B_minus_R_b"]),
                    "r_b": m_sc["r_b"],
                    "unidentifiable_strict": bool(m_sc["unidentifiable_strict"]),
                    "unidentifiable_materiality": bool(m_sc["unidentifiable_materiality"]),
                    "context": row["context"],
                }
            )

            for ap in row["anchoring_pairs"]:
                if not bool(ap.get("computable", False)):
                    continue
                B_a = float(ap["B_anchoring_mean_points"])
                R_a = float(ap["R_anchoring_mean_points"])
                m_a = compute_r_b(X_points, B_a, R_a, EPSILON, MATERIALITY)
                anchoring_rows.append(
                    {
                        "race": race,
                        "subject_driver": subject_driver,
                        "trigger_lap": int(ap["trigger_lap"]),
                        "triggered_by": ap["triggered_by"],
                        "X": X_points,
                        "B": B_a,
                        "R": R_a,
                        "abs_B_minus_R": float(m_a["abs_B_minus_R_b"]),
                        "r_b": m_a["r_b"],
                        "unidentifiable_strict": bool(m_a["unidentifiable_strict"]),
                        "unidentifiable_materiality": bool(m_a["unidentifiable_materiality"]),
                        "context": row["context"],
                    }
                )

        cons_summary = _summarize_bias(conservatism_rows)
        anch_summary = _summarize_bias(anchoring_rows)
        sc_summary = _summarize_bias(sc_rows)

        emit(out, "conservatism_summary=" + json.dumps(cons_summary, sort_keys=True, default=str))
        emit(out, "anchoring_summary=" + json.dumps(anch_summary, sort_keys=True, default=str))
        emit(out, "sc_underweighting_summary=" + json.dumps(sc_summary, sort_keys=True, default=str))

        emit(out, "")
        emit(out, "Step 4 - Surviving materially-identifiable outliers (|r_b| > 5)")

        outliers: List[Dict[str, Any]] = []
        for row in anchoring_rows:
            if row["unidentifiable_strict"] or row["unidentifiable_materiality"] or pd.isna(row["r_b"]):
                continue
            if abs(float(row["r_b"])) > 5.0:
                ctx = dict(row["context"])
                tags = _context_tags(ctx, float(row["X"]))
                outliers.append(
                    {
                        "bias_type": "Anchoring",
                        "race": row["race"],
                        "subject_driver": row["subject_driver"],
                        "trigger_lap": row["trigger_lap"],
                        "triggered_by": row["triggered_by"],
                        "X_actual_mean_points": row["X"],
                        "B_mean_points": row["B"],
                        "R_mean_points": row["R"],
                        "abs_B_minus_R": row["abs_B_minus_R"],
                        "r_b": row["r_b"],
                        "context": ctx,
                        "context_tags": tags,
                    }
                )

        for row in sc_rows:
            if row["unidentifiable_strict"] or row["unidentifiable_materiality"] or pd.isna(row["r_b"]):
                continue
            if abs(float(row["r_b"])) > 5.0:
                ctx = dict(row["context"])
                tags = _context_tags(ctx, float(row["X"]))
                outliers.append(
                    {
                        "bias_type": "SC_Underweighting",
                        "race": row["race"],
                        "subject_driver": row["subject_driver"],
                        "trigger_lap": None,
                        "triggered_by": None,
                        "X_actual_mean_points": row["X"],
                        "B_mean_points": row["B"],
                        "R_mean_points": row["R"],
                        "abs_B_minus_R": row["abs_B_minus_R"],
                        "r_b": row["r_b"],
                        "context": ctx,
                        "context_tags": tags,
                    }
                )

        emit(out, f"surviving_outlier_count={len(outliers)}")
        for item in outliers:
            emit(out, "surviving_outlier_row=" + json.dumps(item, sort_keys=True, default=str))

        emit(out, "")
        emit(out, "Step 5 - Final stopping decision")
        if outliers and all("no_obvious_context_marker" not in o["context_tags"] for o in outliers):
            emit(
                out,
                "stopping_decision=Surviving outliers show plausible real-world context markers (wet/partial-distance/poor-finish patterns); treat as genuine extreme deviations and stop.",
            )
        elif outliers:
            emit(
                out,
                "stopping_decision=Some surviving outliers lack discernible context patterns at n=10000; accept as residual limitation of race-level Anchoring/SC scoping and stop.",
            )
        else:
            emit(out, "stopping_decision=No materially-identifiable |r_b|>5 outliers remain; stop.")

    print(f"output_file={OUTPUT_PATH}")
    print(f"output_file_bytes={OUTPUT_PATH.stat().st_size if OUTPUT_PATH.exists() else 0}")


if __name__ == "__main__":
    main()
