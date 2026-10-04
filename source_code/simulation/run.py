"""Phase 3 full orchestrator.

Builds real subject rosters from base_df, projects runtime, executes one
archetype fully over the lambda grid, and computes bias/rollup outputs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation import config as sim_config  # noqa: E402
from source_code.simulation.monte_carlo import build_actual_strategies  # noqa: E402
from source_code.simulation.optimiser import (  # noqa: E402
    argmax_strategy,
    build_reactive_candidate_sets,
    detect_rival_trigger_events,
    score_subject_strategy_pool,
)
from source_code.simulation.race_state import load_race_state  # noqa: E402
from source_code.simulation.rollup import (  # noqa: E402
    compute_driver_race_bias_summary,
    compute_driver_race_cost_partition,
    rollup_team_archetype,
    rollup_team_race,
    sanity_checks,
)
from source_code.utils import canonical_constructor_group  # noqa: E402


DEFAULT_TECHNICAL_CACHE_PATH = ROOT / "data" / "diagnostics" / "phase3_technical_archetype_full_cache.json"

PILOT_REFERENCE_SECONDS = 1659.927
PILOT_REFERENCE_DRIVER_RACES = 12
TECHNICAL_ARCHETYPE = "Technical"


def emit(out_handle, text: str = "") -> None:
    print(text, flush=True)
    out_handle.write(text + "\n")
    out_handle.flush()


def _first_stop_lap(strategy: Mapping[str, Any]) -> int | None:
    stops = strategy.get("stops", [])
    if not stops:
        return None
    return int(stops[0][0])


def _race_key_from_spec(race_spec: Mapping[str, Any]) -> Tuple[int, int, str]:
    return (int(race_spec["year"]), int(race_spec["round"]), str(race_spec["event_name"]))


def _load_base_df() -> pd.DataFrame:
    base_df = pd.read_csv(sim_config.BASE_DF_PATH)
    required_cols = {"Year", "Round", "EventName", "Driver", "Team", "LapNumber"}
    missing = required_cols - set(base_df.columns)
    if missing:
        raise ValueError(f"base_df missing required columns: {sorted(missing)}")
    return base_df


def derive_real_subject_roster(locked_races: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Load each race entry list from base_df and filter to midfield-valid subjects."""

    base_df = _load_base_df()
    roster_rows: List[Dict[str, Any]] = []

    for race in locked_races:
        year = int(race["year"])
        rnd = int(race["round"])
        event_name = str(race["event_name"])
        archetype = str(race["archetype"])

        race_df = base_df[(base_df["Year"] == year) & (base_df["Round"] == rnd)].copy()
        if race_df.empty:
            raise ValueError(f"No rows found in base_df for Year={year}, Round={rnd}, Event={event_name}")

        race_length_laps = int(pd.to_numeric(race_df["LapNumber"], errors="coerce").max())

        driver_max_laps = (
            race_df.groupby("Driver", dropna=False)["LapNumber"]
            .max()
            .reset_index()
            .rename(columns={"LapNumber": "MaxLapNumber"})
        )

        # Driver-team list as actually present in race rows.
        entry_df = (
            race_df[["Driver", "Team"]]
            .dropna(subset=["Driver", "Team"])
            .drop_duplicates()
            .sort_values(["Team", "Driver"])
        )
        entry_df = entry_df.merge(driver_max_laps, on="Driver", how="left")
        entry_df["is_classified_finish"] = entry_df["MaxLapNumber"].apply(
            lambda x: sim_config.is_classified_finish(int(x), race_length_laps)
            if pd.notna(x)
            else False
        )

        valid_df = entry_df[
            entry_df["Team"].map(sim_config.is_valid_subject)
            & entry_df["is_classified_finish"].astype(bool)
        ].copy()
        subjects = [
            {"driver": str(r["Driver"]), "team": str(r["Team"])}
            for _, r in valid_df.iterrows()
        ]

        excluded_non_classified_df = entry_df[
            entry_df["Team"].map(sim_config.is_valid_subject)
            & ~entry_df["is_classified_finish"].astype(bool)
        ].copy()
        excluded_non_classified = [
            {
                "driver": str(r["Driver"]),
                "team": str(r["Team"]),
                "max_lap": int(r["MaxLapNumber"]),
                "race_length_laps": int(race_length_laps),
                "lap_fraction": float(int(r["MaxLapNumber"]) / race_length_laps),
            }
            for _, r in excluded_non_classified_df.iterrows()
        ]

        team_counts = (
            valid_df.groupby("Team", dropna=False)["Driver"]
            .nunique()
            .sort_index()
            .to_dict()
        )
        anomaly_teams = {str(team): int(count) for team, count in team_counts.items() if int(count) != 2}

        roster_rows.append(
            {
                "race": {
                    "year": year,
                    "round": rnd,
                    "event_name": event_name,
                    "archetype": archetype,
                },
                "race_length_laps": int(race_length_laps),
                "subjects": subjects,
                "valid_subject_driver_count": len(subjects),
                "team_valid_driver_counts": {str(k): int(v) for k, v in team_counts.items()},
                "excluded_non_classified_subjects": excluded_non_classified,
                "anomaly_teams": anomaly_teams,
                "anomaly_note": (
                    "Team has other-than-2 valid subject drivers; both/all listed drivers still count individually at driver-race level, and team-race sums whichever actually raced."
                    if anomaly_teams
                    else None
                ),
            }
        )

    return roster_rows


def compute_runtime_estimate(
    roster_rows: Iterable[Mapping[str, Any]],
    target_archetype: str,
    lambda_values: Iterable[float],
    parallel_workers: int,
    per_driver_race_seconds_override: float | None = None,
) -> Dict[str, Any]:
    per_driver_race_seconds = (
        float(per_driver_race_seconds_override)
        if per_driver_race_seconds_override is not None
        else float(PILOT_REFERENCE_SECONDS) / float(PILOT_REFERENCE_DRIVER_RACES)
    )
    lambda_count = len(list(lambda_values))

    roster_rows = list(roster_rows)
    one_arch_driver_races = int(
        sum(
            int(row["valid_subject_driver_count"])
            for row in roster_rows
            if str(row["race"]["archetype"]) == str(target_archetype)
        )
    )
    full_driver_races = int(sum(int(row["valid_subject_driver_count"]) for row in roster_rows))

    one_arch_driver_race_lambda_tasks = one_arch_driver_races * lambda_count
    full_driver_race_lambda_tasks = full_driver_races * lambda_count

    one_arch_serial_s = one_arch_driver_race_lambda_tasks * per_driver_race_seconds
    full_serial_s = full_driver_race_lambda_tasks * per_driver_race_seconds

    one_arch_wall_s = one_arch_serial_s / max(1, parallel_workers)
    full_wall_s = full_serial_s / max(1, parallel_workers)

    return {
        "pilot_reference": {
            "seconds": float(PILOT_REFERENCE_SECONDS),
            "driver_races": int(PILOT_REFERENCE_DRIVER_RACES),
            "per_driver_race_seconds": per_driver_race_seconds,
        },
        "parallel_workers": int(max(1, parallel_workers)),
        "lambda_count": int(lambda_count),
        "one_archetype": {
            "archetype": str(target_archetype),
            "driver_races": int(one_arch_driver_races),
            "driver_race_lambda_tasks": int(one_arch_driver_race_lambda_tasks),
            "serial_seconds": float(one_arch_serial_s),
            "projected_wall_seconds": float(one_arch_wall_s),
            "projected_wall_minutes": float(one_arch_wall_s / 60.0),
        },
        "full_sweep": {
            "driver_races": int(full_driver_races),
            "driver_race_lambda_tasks": int(full_driver_race_lambda_tasks),
            "serial_seconds": float(full_serial_s),
            "projected_wall_seconds": float(full_wall_s),
            "projected_wall_minutes": float(full_wall_s / 60.0),
        },
    }


def _compute_subject_lambda_row(task: Mapping[str, Any]) -> Dict[str, Any]:
    race = dict(task["race"])
    subject_driver = str(task["subject_driver"])
    subject_team = str(task["subject_team"])
    lambda_value = float(task["lambda"])

    state = load_race_state(int(race["year"]), int(race["round"]))
    laps = state["laps"]

    actual_by_driver = build_actual_strategies(laps)
    if subject_driver not in actual_by_driver:
        raise ValueError(f"Subject driver {subject_driver} missing in actual strategy for race={race}")
    actual_strategy = dict(actual_by_driver[subject_driver])

    race_length_laps = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())

    x_scored = score_subject_strategy_pool(
        race_state=state,
        subject_driver=subject_driver,
        strategies=[actual_strategy],
        prob_source="lap_level",
        lambda_=lambda_value,
        n_iterations=int(sim_config.N_ITERATIONS),
        rng=np.random.default_rng(sim_config.RNG_SEED),
    )
    x_mean_points = float(np.asarray(x_scored["mean_points"], dtype=np.float64)[0])

    b_result = argmax_strategy(
        race_state=state,
        subject_driver=subject_driver,
        prob_source="lap_level",
        information_set="unconditioned",
        lambda_=lambda_value,
        n_iterations=int(sim_config.N_ITERATIONS),
        rng=np.random.default_rng(sim_config.RNG_SEED),
        epsilon=0.01,
    )

    r_sc_result = argmax_strategy(
        race_state=state,
        subject_driver=subject_driver,
        prob_source="static_prior",
        information_set="unconditioned",
        lambda_=lambda_value,
        n_iterations=int(sim_config.N_ITERATIONS),
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
        lambda_=lambda_value,
        n_iterations=int(sim_config.N_ITERATIONS),
        rng=np.random.default_rng(sim_config.RNG_SEED),
    )
    mean_points = np.asarray(scored["mean_points"], dtype=np.float64)
    b_anchor = float(np.max(mean_points)) if mean_points.size else float("nan")

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
            r_value = float("nan")
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

    return {
        "race": {
            "year": int(race["year"]),
            "round": int(race["round"]),
            "event_name": str(race["event_name"]),
            "archetype": str(race["archetype"]),
        },
        "lambda": float(lambda_value),
        "subject_driver": subject_driver,
        "subject_team": subject_team,
        "race_length_laps": race_length_laps,
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


def _count_meaningful_shift(values: List[float], abs_threshold: float = 0.25, rel_threshold: float = 0.10) -> bool:
    if not values:
        return False
    vmin = min(values)
    vmax = max(values)
    spread = vmax - vmin
    baseline = max(1e-9, abs(sum(values) / len(values)))
    return bool(spread >= abs_threshold and (spread / baseline) >= rel_threshold)


def run_phase3_archetype_first_pass(
    target_archetype: str,
    output_path: Path,
    cache_path: Path,
    max_workers: int | None = None,
    runtime_estimate_per_driver_race_seconds: float | None = None,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    workers = int(max_workers or min(4, (os.cpu_count() or 1)))
    workers = max(1, workers)

    locked_races = [
        dict(r)
        for r in sim_config.LOCKED_ARCHETYPE_RACES
        if str(r["archetype"]) == str(target_archetype)
    ]

    archetype_label = str(target_archetype)
    archetype_slug = archetype_label.lower()

    with output_path.open("w", encoding="utf-8") as out:
        emit(out, "=" * 100)
        emit(out, f"Phase 3 run.py orchestrator - {archetype_label} archetype first full execution")
        emit(out, "=" * 100)
        emit(out, f"n_iterations={sim_config.N_ITERATIONS}")
        emit(out, f"lambda_grid={list(sim_config.LAMBDA_GRID)}")
        emit(out, f"parallel_workers={workers}")

        archetype_race_keys = {
            _race_key_from_spec(r)
            for r in locked_races
        }

        emit(out, "")
        emit(out, f"Step 1 - Real per-race subject roster from base_df.csv ({archetype_label} races only)")
        roster_rows = derive_real_subject_roster(locked_races)
        archetype_roster = [
            row for row in roster_rows if _race_key_from_spec(row["race"]) in archetype_race_keys
        ]
        for row in archetype_roster:
            emit(
                out,
                f"{archetype_slug}_race_subject_roster="
                + json.dumps(
                    {
                        "race": row["race"],
                        "race_length_laps": row.get("race_length_laps"),
                        "subjects": row["subjects"],
                        "valid_subject_driver_count": row["valid_subject_driver_count"],
                        "team_valid_driver_counts": row["team_valid_driver_counts"],
                        "excluded_non_classified_subjects": row.get("excluded_non_classified_subjects", []),
                        "anomaly_teams": row["anomaly_teams"],
                        "anomaly_note": row["anomaly_note"],
                    },
                    sort_keys=True,
                    default=str,
                ),
            )

        emit(out, "")
        emit(out, "Step 2 - Grounded time/resource estimate before scale")
        estimate = compute_runtime_estimate(
            roster_rows=roster_rows,
            target_archetype=archetype_label,
            lambda_values=sim_config.LAMBDA_GRID,
            parallel_workers=workers,
            per_driver_race_seconds_override=runtime_estimate_per_driver_race_seconds,
        )
        emit(out, "runtime_estimate=" + json.dumps(estimate, sort_keys=True, default=str))
        emit(out, "runtime_estimate_formula_note=Projected wall seconds = (driver_race_count x lambda_count x per_driver_race_seconds_from_pilot) / parallel_workers")
        one_arch = estimate["one_archetype"]
        per_driver_race_seconds = estimate["pilot_reference"]["per_driver_race_seconds"]
        emit(
            out,
            "runtime_estimate_arithmetic="
            f"({one_arch['driver_races']} driver-races x {estimate['lambda_count']} lambdas x {per_driver_race_seconds:.6f} s)"
            f" / {estimate['parallel_workers']} workers = {one_arch['projected_wall_seconds']:.3f} s"
            f" ({one_arch['projected_wall_minutes']:.3f} min)",
        )

        emit(out, "")
        emit(out, f"Step 3 - Execute {archetype_label} archetype fully over real subject roster and full lambda grid")
        tasks: List[Dict[str, Any]] = []
        for race_row in archetype_roster:
            race = dict(race_row["race"])
            for subject in race_row["subjects"]:
                for lambda_value in sim_config.LAMBDA_GRID:
                    tasks.append(
                        {
                            "race": race,
                            "subject_driver": str(subject["driver"]),
                            "subject_team": str(subject["team"]),
                            "lambda": float(lambda_value),
                        }
                    )

        emit(out, f"{archetype_slug}_task_count_driver_race_lambda={len(tasks)}")
        steps_3_to_5_start = time.perf_counter()

        cache_rows: List[Dict[str, Any]] = []

        def persist_cache_rows_partial() -> None:
            cache_rows.sort(
                key=lambda r: (
                    float(r["lambda"]),
                    int(r["race"]["year"]),
                    int(r["race"]["round"]),
                    str(r["subject_team"]),
                    str(r["subject_driver"]),
                )
            )
            cache_path.write_text(json.dumps(cache_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_compute_subject_lambda_row, task): task for task in tasks}
            for future in as_completed(future_map):
                row = future.result()
                cache_rows.append(row)
                # Persist each completed driver-race/lambda row so long runs remain observable/resumable.
                persist_cache_rows_partial()
                emit(
                    out,
                    f"{archetype_slug}_cache_row_summary="
                    + json.dumps(
                        {
                            "race": row["race"],
                            "lambda": row["lambda"],
                            "subject_driver": row["subject_driver"],
                            "subject_team": row["subject_team"],
                            "X_actual_mean_points": row["X_actual_mean_points"],
                            "B_unconditioned_mean_points": row["B_unconditioned_mean_points"],
                            "R_SC_static_prior_mean_points": row["R_SC_static_prior_mean_points"],
                            "anchoring_pair_count": len(row["anchoring_pairs"]),
                        },
                        sort_keys=True,
                        default=str,
                    ),
                )
                emit(out, f"{archetype_slug}_cache_rows_written_so_far={len(cache_rows)}")

        persist_cache_rows_partial()
        emit(out, f"{archetype_slug}_cache_written={cache_path}")
        emit(out, f"{archetype_slug}_cache_row_count={len(cache_rows)}")

        emit(out, "")
        emit(out, f"Step 4 - Compute bias and run full rollup on {archetype_slug} full cache")
        driver_bias_rows = compute_driver_race_bias_summary(
            cache_rows,
            epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
            materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
        )
        partitioned_rows = compute_driver_race_cost_partition(driver_bias_rows)
        team_race_rows = rollup_team_race(partitioned_rows)
        team_archetype_rows = rollup_team_archetype(team_race_rows)

        emit(out, f"driver_race_bias_row_count={len(driver_bias_rows)}")
        emit(out, f"driver_race_partition_row_count={len(partitioned_rows)}")
        emit(out, f"team_race_row_count={len(team_race_rows)}")
        emit(out, f"team_archetype_row_count={len(team_archetype_rows)}")

        team_race_double_driver_rows = [row for row in team_race_rows if int(row.get("driver_race_count", 0)) == 2]
        emit(out, f"team_race_rows_with_driver_race_count_eq_2={len(team_race_double_driver_rows)}")

        pair_sum_checks: List[Dict[str, Any]] = []
        for team_row in team_race_double_driver_rows:
            race = team_row["race"]
            lambda_value = float(team_row.get("lambda", 1.0))
            contributors = [
                r
                for r in partitioned_rows
                if int(r["race"]["year"]) == int(race["year"])
                and int(r["race"]["round"]) == int(race["round"])
                and str(r["race"]["event_name"]) == str(race["event_name"])
                and canonical_constructor_group(r.get("subject_team")) == str(team_row["team"])
                and abs(float(r.get("lambda", 1.0)) - lambda_value) < 1e-12
            ]
            summed = float(sum(float(r["total_cost"]) for r in contributors))
            row_total = float(team_row["total_cost"])
            pair_sum_checks.append(
                {
                    "race": race,
                    "team": team_row["team"],
                    "lambda": lambda_value,
                    "driver_race_count": len(contributors),
                    "summed_driver_race_total_cost": summed,
                    "team_race_total_cost": row_total,
                    "sum_matches": abs(summed - row_total) < 1e-9,
                }
            )

        for item in pair_sum_checks:
            emit(out, "team_race_two_driver_sum_check=" + json.dumps(item, sort_keys=True, default=str))

        rb_source_rows = [
            row
            for row in team_race_rows
            if str(row.get("team", "")) == "RB"
            and any(source in {"AlphaTauri", "RB"} for source in row.get("source_teams", []))
        ]
        for row in rb_source_rows:
            emit(
                out,
                f"{archetype_slug}_rb_canonical_source_trace="
                + json.dumps(
                    {
                        "race": row["race"],
                        "team": row["team"],
                        "lambda": row["lambda"],
                        "driver_race_count": row["driver_race_count"],
                        "source_teams": row.get("source_teams", []),
                    },
                    sort_keys=True,
                    default=str,
                ),
            )

        emit(out, "")
        emit(out, f"Step 5 - Lambda sensitivity on {archetype_label} team x archetype cells")
        by_team_arch: Dict[Tuple[str, str], Dict[float, Dict[str, Any]]] = defaultdict(dict)
        for row in team_archetype_rows:
            by_team_arch[(str(row["team"]), str(row["archetype"]))][float(row.get("lambda", 1.0))] = row

        sensitivity_rows: List[Dict[str, Any]] = []
        for (team, archetype), lambda_map in sorted(by_team_arch.items()):
            series = {}
            for lambda_value in sorted(lambda_map.keys()):
                entry = lambda_map[lambda_value]
                avg_total_cost = float(entry["avg_total_cost"])
                avg_cost_conservatism = float(entry["avg_cost_conservatism"])
                avg_cost_anchoring = float(entry["avg_cost_anchoring"])
                avg_cost_sc_underweighting = float(entry["avg_cost_sc_underweighting"])
                avg_cost_unattributed = float(entry["avg_cost_unattributed"])
                series[str(lambda_value)] = {
                    "avg_total_cost": avg_total_cost,
                    "avg_cost_conservatism": avg_cost_conservatism,
                    "avg_cost_anchoring": avg_cost_anchoring,
                    "avg_cost_sc_underweighting": avg_cost_sc_underweighting,
                    "avg_cost_unattributed": avg_cost_unattributed,
                    "avg_cost_unattributed_fraction_of_avg_total_cost": (
                        avg_cost_unattributed / avg_total_cost if avg_total_cost else 0.0
                    ),
                    "race_support_count": int(entry["race_support_count"]),
                    "low_confidence_support": bool(entry["low_confidence_support"]),
                }

            totals = [float(v["avg_total_cost"]) for v in series.values()]
            cons_vals = [float(v["avg_cost_conservatism"]) for v in series.values()]
            anch_vals = [float(v["avg_cost_anchoring"]) for v in series.values()]
            sc_vals = [float(v["avg_cost_sc_underweighting"]) for v in series.values()]
            unattributed_vals = [float(v["avg_cost_unattributed"]) for v in series.values()]

            dominant_by_lambda = {}
            for lambda_text, values in series.items():
                dominant = max(
                    [
                        ("conservatism", values["avg_cost_conservatism"]),
                        ("anchoring", values["avg_cost_anchoring"]),
                        ("sc_underweighting", values["avg_cost_sc_underweighting"]),
                        ("unattributed", values["avg_cost_unattributed"]),
                    ],
                    key=lambda x: x[1],
                )[0]
                dominant_by_lambda[lambda_text] = dominant

            dominant_shift = len(set(dominant_by_lambda.values())) > 1
            magnitude_shift = any(
                [
                    _count_meaningful_shift(totals),
                    _count_meaningful_shift(cons_vals),
                    _count_meaningful_shift(anch_vals),
                    _count_meaningful_shift(sc_vals),
                    _count_meaningful_shift(unattributed_vals),
                ]
            )

            interpretation = (
                "Meaningful lambda sensitivity detected in this cell."
                if (dominant_shift or magnitude_shift)
                else "Little practical lambda sensitivity in this cell; consistent with earlier limited lambda impact finding."
            )

            row_out = {
                "team": team,
                "archetype": archetype,
                "series_by_lambda": series,
                "dominant_bias_by_lambda": dominant_by_lambda,
                "dominant_bias_shift": bool(dominant_shift),
                "magnitude_shift_meaningful": bool(magnitude_shift),
                "interpretation": interpretation,
            }
            sensitivity_rows.append(row_out)
            emit(out, f"{archetype_slug}_lambda_sensitivity_row=" + json.dumps(row_out, sort_keys=True, default=str))

        checks = sanity_checks(partitioned_rows, team_race_rows, team_archetype_rows)
        emit(out, "sanity_checks=" + json.dumps(checks, sort_keys=True, default=str))

        steps_3_to_5_seconds = time.perf_counter() - steps_3_to_5_start

        emit(out, "")
        emit(out, "Step 6 - Actual runtime and refined projection")
        emit(out, f"steps_3_to_5_wall_clock_seconds={steps_3_to_5_seconds:.3f}")

        archetype_task_count = max(1, len(tasks))
        realized_wall_seconds_per_driver_race_lambda_task = steps_3_to_5_seconds / float(archetype_task_count)
        full_driver_races = int(sum(int(r["valid_subject_driver_count"]) for r in roster_rows))
        full_task_count = full_driver_races * len(sim_config.LAMBDA_GRID)
        refined_full_wall_seconds = realized_wall_seconds_per_driver_race_lambda_task * float(full_task_count)

        emit(
            out,
            "runtime_refinement="
            + json.dumps(
                {
                    "technical_tasks_completed": int(archetype_task_count),
                    "realized_wall_seconds_per_driver_race_lambda_task": realized_wall_seconds_per_driver_race_lambda_task,
                    "full_driver_races": int(full_driver_races),
                    "full_task_count_driver_race_lambda": int(full_task_count),
                    "refined_full_projection_wall_seconds": refined_full_wall_seconds,
                    "refined_full_projection_wall_minutes": refined_full_wall_seconds / 60.0,
                },
                sort_keys=True,
                default=str,
            ),
        )

        emit(out, "")
        emit(out, f"Step 7 - Completed {archetype_slug}-only execution (no other archetypes run)")
        emit(out, f"execution_scope_note=Only {archetype_label} archetype executed in this step by design.")
        emit(out, f"{archetype_slug}_cache_path={cache_path}")
        emit(out, f"output_path={output_path}")

    return {
        "output_path": output_path,
        "cache_path": cache_path,
    }


def run_phase3_technical_first_pass(
    output_path: Path,
    cache_path: Path = DEFAULT_TECHNICAL_CACHE_PATH,
    max_workers: int | None = None,
) -> Dict[str, Any]:
    return run_phase3_archetype_first_pass(
        TECHNICAL_ARCHETYPE,
        output_path=output_path,
        cache_path=cache_path,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    output_dir = ROOT / "data" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase3_simulation_output.txt"
    result = run_phase3_technical_first_pass(
        output_path=output_path,
        cache_path=DEFAULT_TECHNICAL_CACHE_PATH,
    )
    print(f"Phase 3 simulation complete.")
    print(f"Output: {result['output_path']}")
    print(f"Cache: {result['cache_path']}")
