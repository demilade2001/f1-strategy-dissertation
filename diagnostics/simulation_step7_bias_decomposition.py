from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import config as sim_config  # noqa: E402
from src.simulation.bias import compute_r_b, partition_cost  # noqa: E402
from src.simulation.monte_carlo import build_actual_strategies, build_probability_sources, run_monte_carlo  # noqa: E402
from src.simulation.optimiser import argmax_strategy, detect_rival_trigger_events  # noqa: E402
from src.simulation.race_state import load_race_state  # noqa: E402
from src.simulation.references import r_conservatism  # noqa: E402


DIAG_DIR = ROOT / "data" / "diagnostics"
OUTPUT_PATH = DIAG_DIR / "simulation_step7_bias_decomposition_output.txt"
CACHE_PATH = DIAG_DIR / "phase3_pilot_reference_cache.json"
EPSILON = float(sim_config.UNIDENTIFIABLE_EPSILON)
PROB_SOURCE = "lap_level"
LAMBDA_VALUE = 1.0
RUNTIME_N_ITERATIONS = int(os.getenv("STEP7_RUNTIME_N_ITERATIONS", "1000"))


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
    corrected_path = ROOT / "data" / "diagnostics" / "deg_rate_corrected_shrunk_v3_full_peryear_stats.csv"
    if corrected_path.exists():
        deg = pd.read_csv(corrected_path)
        event_name = str(state["laps"]["EventName"].dropna().iloc[0])
        state["deg_stats"] = deg[(deg["Year"] == int(race["year"])) & (deg["EventName"] == event_name)].copy()
    return state


def _compute_reference_cache_for_race(race: Mapping[str, Any], out_handle) -> Dict[str, Any]:
    state = _load_race_state_with_corrected_deg(race)
    laps = state["laps"]
    subject_driver = str(race["subject_driver"])

    actual_by_driver = build_actual_strategies(laps)
    if subject_driver not in actual_by_driver:
        raise ValueError(f"Subject driver {subject_driver} missing from actual strategies for {race}")

    actual_strategy = dict(actual_by_driver[subject_driver])
    race_length_laps = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())

    prob_sources = build_probability_sources(state)
    x_result = run_monte_carlo(
        race_state=state,
        subject_driver=subject_driver,
        subject_strategy=actual_strategy,
        prob_by_lap=prob_sources["merged"],
        lambda_=LAMBDA_VALUE,
        n_iterations=RUNTIME_N_ITERATIONS,
        rng=np.random.default_rng(sim_config.RNG_SEED),
    )

    b_result = argmax_strategy(
        race_state=state,
        subject_driver=subject_driver,
        prob_source="lap_level",
        information_set="unconditioned",
        lambda_=LAMBDA_VALUE,
        n_iterations=RUNTIME_N_ITERATIONS,
        rng=np.random.default_rng(sim_config.RNG_SEED),
        epsilon=0.01,
    )

    r_sc_result = argmax_strategy(
        race_state=state,
        subject_driver=subject_driver,
        prob_source="static_prior",
        information_set="unconditioned",
        lambda_=LAMBDA_VALUE,
        n_iterations=RUNTIME_N_ITERATIONS,
        rng=np.random.default_rng(sim_config.RNG_SEED),
        epsilon=0.01,
    )

    b_strategy = dict(b_result["best_strategy"])
    r_sc_strategy = dict(r_sc_result["best_strategy"])

    row = {
        "race": {
            "year": int(race["year"]),
            "round": int(race["round"]),
            "event_name": str(race["event_name"]),
            "archetype": str(race["archetype"]),
        },
        "subject_driver": subject_driver,
        "race_length_laps": race_length_laps,
        "X_actual_strategy": actual_strategy,
        "X_actual_first_pit_lap": _first_stop_lap(actual_strategy),
        "X_actual_mean_points": float(x_result["mean_points"]),
        "B_unconditioned_strategy": b_strategy,
        "B_unconditioned_first_pit_lap": _first_stop_lap(b_strategy),
        "B_unconditioned_mean_points": float(b_result["best_mean_points"]),
        "R_SC_static_prior_strategy": r_sc_strategy,
        "R_SC_static_prior_first_pit_lap": _first_stop_lap(r_sc_strategy),
        "R_SC_static_prior_mean_points": float(r_sc_result["best_mean_points"]),
    }

    emit(out_handle, "pilot_cache_row=" + json.dumps(row, sort_keys=True, default=str))
    return row


def _compute_trigger_rows(race: Mapping[str, Any], state: dict, cache_row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    subject_driver = str(cache_row["subject_driver"])
    trigger_data = detect_rival_trigger_events(
        race_state=state,
        subject_driver=subject_driver,
        proximity_margin_s=sim_config.PROXIMITY_MARGIN_S,
        include_big_three_as_rivals=sim_config.INCLUDE_BIG_THREE_AS_RIVALS,
    )
    events = trigger_data["trigger_events"]

    from src.simulation.optimiser import build_reactive_candidate_sets, score_subject_strategy_pool

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
        prob_source=PROB_SOURCE,
        lambda_=LAMBDA_VALUE,
        n_iterations=RUNTIME_N_ITERATIONS,
        rng=np.random.default_rng(sim_config.RNG_SEED),
    )

    mean_points = np.asarray(scored["mean_points"], dtype=np.float64)
    b_value = float(np.max(mean_points)) if mean_points.size else math.nan

    rows: List[Dict[str, Any]] = []
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
            unidentifiable = False
        else:
            r_value = float(np.max(mean_points[np.asarray(r_idx, dtype=np.int64)]))
            computable = True
            unidentifiable = bool(abs(b_value - r_value) < EPSILON)

        rows.append(
            {
                "race": {
                    "year": int(race["year"]),
                    "round": int(race["round"]),
                    "event_name": str(race["event_name"]),
                    "archetype": str(race["archetype"]),
                },
                "subject_driver": subject_driver,
                "trigger_lap": trigger_lap,
                "triggered_by": event["triggered_by"],
                "added_lap_for_r": added_lap,
                "b_mean_points": b_value,
                "r_anchoring_mean_points": r_value,
                "computable": computable,
                "step6b_unidentifiable_strict": unidentifiable,
                "r_pool_count": int(len(r_idx)),
            }
        )
    return rows


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        emit(out, "=" * 100)
        emit(out, "Step 7 - Bias decomposition recompute (fresh compute, no log parsing, no silent fallbacks)")
        emit(out, "=" * 100)
        emit(out, f"runtime_n_iterations={RUNTIME_N_ITERATIONS}")
        emit(out, f"epsilon={EPSILON}")

        emit(out, "")
        emit(out, "Step 1 - Build structured pilot cache (12 driver-races)")
        cache_rows: List[Dict[str, Any]] = []
        race_state_by_key: Dict[Tuple[int, int, str, str], dict] = {}

        for race in sim_config.LOCKED_ARCHETYPE_RACES:
            race_info = dict(race)
            state = _load_race_state_with_corrected_deg(race_info)
            subject_driver = race_info.get("subject_driver")
            if not subject_driver:
                from src.simulation.monte_carlo import select_midfield_subject_driver

                subject_driver = select_midfield_subject_driver(state["laps"])
            race_info["subject_driver"] = str(subject_driver)

            row = _compute_reference_cache_for_race(race_info, out)
            key = _race_key(row["race"], row["subject_driver"])
            race_state_by_key[key] = state
            cache_rows.append(row)

        CACHE_PATH.write_text(json.dumps(cache_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        emit(out, f"pilot_cache_written={CACHE_PATH}")
        emit(out, f"pilot_cache_row_count={len(cache_rows)}")

        emit(out, "")
        emit(out, "Step 2 - Sanity check B vs R_SC")
        equal_count = 0
        for row in cache_rows:
            b = float(row["B_unconditioned_mean_points"])
            r_sc = float(row["R_SC_static_prior_mean_points"])
            equal = bool(abs(b - r_sc) < EPSILON)
            if equal:
                equal_count += 1
            emit(
                out,
                "b_vs_r_sc="
                + json.dumps(
                    {
                        "race": row["race"],
                        "subject_driver": row["subject_driver"],
                        "B_mean_points": b,
                        "R_SC_mean_points": r_sc,
                        "equal_within_epsilon": equal,
                    },
                    sort_keys=True,
                    default=str,
                ),
            )
        emit(out, f"b_vs_r_sc_equal_count={equal_count}")

        emit(out, "")
        emit(out, "Step 3 - Conservatism recompute")
        emit(out, "Conservatism is computed in lap-space: X/B/R are first-pit-lap quantities.")
        emit(out, "conservatism_rows_begin")
        conservatism_rows: List[Dict[str, Any]] = []

        for row in cache_rows:
            race_state = {
                "race_length_laps": int(row["race_length_laps"]),
                "subject_final_compound": None,
            }
            ref = r_conservatism(race_state, str(row["subject_driver"]))
            X = row["X_actual_first_pit_lap"]
            B = row["B_unconditioned_first_pit_lap"]
            R = ref["reference_lap"]

            if X is None or B is None or R is None:
                out_row = {
                    "race": row["race"],
                    "subject_driver": row["subject_driver"],
                    "computable": False,
                    "reason": "missing first-pit-lap quantity",
                    "X_actual_first_pit_lap": X,
                    "B_first_pit_lap": B,
                    "R_conservatism_lap": R,
                }
            else:
                metrics = compute_r_b(float(X), float(B), float(R), EPSILON)
                out_row = {
                    "race": row["race"],
                    "subject_driver": row["subject_driver"],
                    "computable": True,
                    "X_actual_first_pit_lap": int(X),
                    "B_first_pit_lap": int(B),
                    "R_conservatism_lap": int(R),
                    "r_b": metrics["r_b"],
                    "magnitude": metrics["magnitude"],
                    "unidentifiable": metrics["unidentifiable"],
                }
            conservatism_rows.append(out_row)
            emit(out, json.dumps(out_row, sort_keys=True, default=str))

        emit(out, "conservatism_rows_end")
        emit(out, f"conservatism_total_rows={len(conservatism_rows)}")
        emit(out, f"conservatism_computable_rows={sum(1 for r in conservatism_rows if r.get('computable'))}")

        emit(out, "")
        emit(out, "Step 4 - Anchoring recompute using fresh X (no X=B fallback)")
        emit(out, "X is race-level actual strategy mean points from Step 1 cache.")

        all_trigger_rows: List[Dict[str, Any]] = []
        for row in cache_rows:
            key = _race_key(row["race"], row["subject_driver"])
            state = race_state_by_key[key]
            race = {
                "year": row["race"]["year"],
                "round": row["race"]["round"],
                "event_name": row["race"]["event_name"],
                "archetype": row["race"]["archetype"],
            }
            emit(
                out,
                "anchoring_race_start="
                + json.dumps(
                    {
                        "race": race,
                        "subject_driver": row["subject_driver"],
                    },
                    sort_keys=True,
                    default=str,
                ),
            )
            race_trigger_rows = _compute_trigger_rows(race, state, row)
            all_trigger_rows.extend(race_trigger_rows)
            emit(
                out,
                "anchoring_race_done="
                + json.dumps(
                    {
                        "race": race,
                        "subject_driver": row["subject_driver"],
                        "trigger_row_count": len(race_trigger_rows),
                    },
                    sort_keys=True,
                    default=str,
                ),
            )

        emit(out, "anchoring_rows_begin")
        anchoring_rows: List[Dict[str, Any]] = []
        r_b_not_one_count = 0
        recomputed_unidentifiable = 0
        step6b_computable = 0

        x_points_by_key = {
            _race_key(row["race"], row["subject_driver"]): float(row["X_actual_mean_points"])
            for row in cache_rows
        }

        for trow in all_trigger_rows:
            key = _race_key(trow["race"], trow["subject_driver"])
            X = x_points_by_key[key]
            B = trow["b_mean_points"]
            R = trow["r_anchoring_mean_points"]

            if not trow["computable"] or pd.isna(B) or pd.isna(R):
                out_row = {
                    "race": trow["race"],
                    "subject_driver": trow["subject_driver"],
                    "trigger_lap": trow["trigger_lap"],
                    "computable": False,
                    "reason": "no R_anchoring candidate for this trigger",
                }
            else:
                step6b_computable += 1
                metrics = compute_r_b(float(X), float(B), float(R), EPSILON)
                if metrics["unidentifiable"]:
                    recomputed_unidentifiable += 1
                if (not metrics["unidentifiable"]) and (not math.isclose(float(metrics["r_b"]), 1.0, abs_tol=1e-12)):
                    r_b_not_one_count += 1

                out_row = {
                    "race": trow["race"],
                    "subject_driver": trow["subject_driver"],
                    "trigger_lap": trow["trigger_lap"],
                    "computable": True,
                    "X_actual_mean_points": float(X),
                    "B_mean_points": float(B),
                    "R_anchoring_mean_points": float(R),
                    "r_b": metrics["r_b"],
                    "magnitude": metrics["magnitude"],
                    "unidentifiable": metrics["unidentifiable"],
                }
            anchoring_rows.append(out_row)
            emit(out, json.dumps(out_row, sort_keys=True, default=str))

        emit(out, "anchoring_rows_end")
        emit(out, f"anchoring_total_rows={len(anchoring_rows)}")
        emit(out, f"anchoring_step6b_computable_rows={step6b_computable}")
        emit(out, f"anchoring_recomputed_unidentifiable={recomputed_unidentifiable}")
        emit(out, f"anchoring_r_b_not_equal_1_count={r_b_not_one_count}")
        if r_b_not_one_count == 0:
            emit(out, "anchoring_anomaly=ALL_IDENTIFIABLE_ROWS_HAVE_r_b_EQ_1.0")

        emit(out, "")
        emit(out, "Step 5 - SC Underweighting recompute (race-level simplification)")
        emit(out, "Scoping note: race-level only, not per-caution-lap decision-point action changes.")
        emit(out, "sc_rows_begin")

        sc_rows: List[Dict[str, Any]] = []
        for row in cache_rows:
            X = float(row["X_actual_mean_points"])
            B = float(row["B_unconditioned_mean_points"])
            R = float(row["R_SC_static_prior_mean_points"])
            metrics = compute_r_b(X, B, R, EPSILON)
            out_row = {
                "race": row["race"],
                "subject_driver": row["subject_driver"],
                "computable": True,
                "X_actual_mean_points": X,
                "B_lap_level_mean_points": B,
                "R_sc_static_prior_mean_points": R,
                "r_b": metrics["r_b"],
                "magnitude": metrics["magnitude"],
                "unidentifiable": metrics["unidentifiable"],
            }
            sc_rows.append(out_row)
            emit(out, json.dumps(out_row, sort_keys=True, default=str))

        emit(out, "sc_rows_end")
        emit(out, f"sc_total_rows={len(sc_rows)}")
        emit(out, f"sc_computable_rows={sum(1 for r in sc_rows if r.get('computable'))}")

        emit(out, "")
        emit(out, "Step 6 - Cost partition sample")
        emit(out, "Anchoring contribution uses race-level average r_b across identifiable trigger rows.")
        emit(out, "cost_partition_rows_begin")

        cons_by_key = {
            _race_key(r["race"], r["subject_driver"]): r
            for r in conservatism_rows
            if r.get("computable")
        }
        sc_by_key = {
            _race_key(r["race"], r["subject_driver"]): r
            for r in sc_rows
            if r.get("computable")
        }

        anch_values_by_key: Dict[Tuple[int, int, str, str], List[float]] = defaultdict(list)
        for r in anchoring_rows:
            if not r.get("computable") or r.get("unidentifiable"):
                continue
            anch_values_by_key[_race_key(r["race"], r["subject_driver"])].append(float(r["r_b"]))

        sample_count = 0
        for row in cache_rows:
            if sample_count >= 5:
                break
            key = _race_key(row["race"], row["subject_driver"])

            candidates = {}
            c = cons_by_key.get(key)
            s = sc_by_key.get(key)
            a_vals = anch_values_by_key.get(key, [])
            if c is not None and not c.get("unidentifiable"):
                candidates["conservatism"] = {"r_b": c["r_b"], "unidentifiable": False}
            if s is not None and not s.get("unidentifiable"):
                candidates["sc_underweighting"] = {"r_b": s["r_b"], "unidentifiable": False}
            if a_vals:
                candidates["anchoring"] = {"r_b": float(np.mean(a_vals)), "unidentifiable": False}

            if len(candidates) < 2:
                continue

            total_cost = float(row["B_unconditioned_mean_points"]) - float(row["X_actual_mean_points"])
            split = partition_cost(total_cost, candidates)
            out_row = {
                "race": row["race"],
                "subject_driver": row["subject_driver"],
                "total_cost": total_cost,
                "r_b_inputs": {name: vals["r_b"] for name, vals in candidates.items()},
                "cost_split": {k: v for k, v in split.items() if not k.startswith("_")},
                "sum_check": split.get("_sum"),
                "sum_matches_total_cost": abs(float(split.get("_sum", 0.0)) - total_cost) < 1e-9,
            }
            emit(out, json.dumps(out_row, sort_keys=True, default=str))
            sample_count += 1

        emit(out, "cost_partition_rows_end")
        emit(out, f"cost_partition_sample_count={sample_count}")

    print(f"output_file={OUTPUT_PATH}")
    print(f"output_file_bytes={OUTPUT_PATH.stat().st_size if OUTPUT_PATH.exists() else 0}")


if __name__ == "__main__":
    main()
