from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from source_code.simulation import config as sim_config
from source_code.simulation.monte_carlo import build_actual_strategies
from source_code.simulation.optimiser import argmax_strategy
from source_code.simulation.race_state import load_race_state
from source_code.simulation.rollup import (
    compute_driver_race_bias_summary,
    compute_driver_race_cost_partition,
    rollup_team_archetype,
    rollup_team_race,
    sanity_checks,
)


POWER_CACHE_PATH = Path("data/diagnostics/phase3_power_archetype_full_cache.json")
ROLLUP_OUTPUT_PATH = Path("data/diagnostics/phase3_power_archetype_full_cache_recomputed_rollup_output.txt")

TARGET_RACE_KEY = (2023, 10, "British Grand Prix", "OCO")


def _row_key(row: Dict[str, Any]) -> Tuple[int, int, str, str, float]:
    race = row["race"]
    return (
        int(race["year"]),
        int(race["round"]),
        str(race["event_name"]),
        str(row["subject_driver"]),
        float(row["lambda"]),
    )


def _first_stop_lap(strategy: Dict[str, Any]) -> int | None:
    stops = strategy.get("stops", [])
    if not stops:
        return None
    return int(stops[0][0])


def _scan_locked_zero_stop_cases() -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for race in sim_config.LOCKED_ARCHETYPE_RACES:
        year = int(race["year"])
        rnd = int(race["round"])
        event_name = str(race["event_name"])
        archetype = str(race["archetype"])

        state = load_race_state(year, rnd)
        actual = build_actual_strategies(state["laps"])
        team_map = {
            str(row.get("Driver")): str(row.get("Team"))
            for row in state.get("drivers_with_teams", [])
            if row.get("Driver") is not None and row.get("Team") is not None
        }

        for driver, strategy in actual.items():
            team = team_map.get(str(driver), "")
            if not sim_config.is_valid_subject(team):
                continue
            stops = strategy.get("stops", [])
            if len(stops) == 0:
                hits.append(
                    {
                        "race": {
                            "year": year,
                            "round": rnd,
                            "event_name": event_name,
                            "archetype": archetype,
                        },
                        "subject_driver": str(driver),
                        "subject_team": str(team),
                        "starting_compound": str(strategy.get("starting_compound")),
                        "stops": [],
                    }
                )

    hits.sort(
        key=lambda r: (
            str(r["race"]["archetype"]),
            int(r["race"]["year"]),
            int(r["race"]["round"]),
            str(r["subject_team"]),
            str(r["subject_driver"]),
        )
    )
    return hits


def _recompute_power_targets(cache_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    updates: List[Dict[str, Any]] = []

    for row in cache_rows:
        race = row["race"]
        key_wo_lambda = (
            int(race["year"]),
            int(race["round"]),
            str(race["event_name"]),
            str(row["subject_driver"]),
        )
        if key_wo_lambda != TARGET_RACE_KEY:
            continue

        lam = float(row["lambda"])
        state = load_race_state(int(race["year"]), int(race["round"]))
        result = argmax_strategy(
            race_state=state,
            subject_driver=str(row["subject_driver"]),
            prob_source="lap_level",
            information_set="unconditioned",
            lambda_=lam,
            n_iterations=int(sim_config.N_ITERATIONS),
            rng=np.random.default_rng(sim_config.RNG_SEED),
            epsilon=0.01,
        )

        old_b = float(row["B_unconditioned_mean_points"])
        new_b = float(result["best_mean_points"])

        row["B_unconditioned_strategy"] = dict(result["best_strategy"])
        row["B_unconditioned_first_pit_lap"] = _first_stop_lap(dict(result["best_strategy"]))
        row["B_unconditioned_mean_points"] = new_b

        updates.append(
            {
                "lambda": lam,
                "old_B_unconditioned_mean_points": old_b,
                "new_B_unconditioned_mean_points": new_b,
                "X_actual_mean_points": float(row["X_actual_mean_points"]),
                "new_B_minus_X": float(new_b - float(row["X_actual_mean_points"])),
                "new_B_ge_X": bool(new_b >= float(row["X_actual_mean_points"])),
                "new_B_strategy": row["B_unconditioned_strategy"],
            }
        )

    updates.sort(key=lambda r: float(r["lambda"]))
    return cache_rows, updates


def _compute_rollups(cache_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    driver_bias = compute_driver_race_bias_summary(
        cache_rows,
        epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
        materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
    )
    driver_cost = compute_driver_race_cost_partition(driver_bias)
    team_race = rollup_team_race(driver_cost)
    team_archetype = rollup_team_archetype(team_race)
    checks = sanity_checks(driver_cost, team_race, team_archetype)
    return {
        "driver_cost": driver_cost,
        "team_race": team_race,
        "team_archetype": team_archetype,
        "checks": checks,
    }


def _extract_alpine_power_cells(team_arch_rows: List[Dict[str, Any]]) -> Dict[float, Dict[str, Any]]:
    out: Dict[float, Dict[str, Any]] = {}
    for row in team_arch_rows:
        if str(row["team"]) == "Alpine" and str(row["archetype"]) == "Power":
            out[float(row["lambda"])] = dict(row)
    return out


def main() -> None:
    cache_rows = json.loads(POWER_CACHE_PATH.read_text(encoding="utf-8"))

    print("=" * 100)
    print("Step 9g - Zero-stop wet fix validation + targeted Power cache patch")
    print("=" * 100)
    print(f"power_cache_path={POWER_CACHE_PATH.resolve()}")
    print(f"power_cache_row_count={len(cache_rows)}")

    print("\nStep 2 - Locked-race scan for real zero-stop midfield drivers")
    zero_stop_hits = _scan_locked_zero_stop_cases()
    print(f"locked_zero_stop_count={len(zero_stop_hits)}")
    print("locked_zero_stop_rows_begin")
    for row in zero_stop_hits:
        print("locked_zero_stop_row=" + json.dumps(row, sort_keys=True))
    print("locked_zero_stop_rows_end")

    monaco_singapore = [
        row
        for row in zero_stop_hits
        if (
            (str(row["race"]["event_name"]) == "Monaco Grand Prix" and int(row["race"]["year"]) == 2023)
            or (str(row["race"]["event_name"]) == "Singapore Grand Prix" and int(row["race"]["year"]) == 2022)
        )
    ]
    print(f"monaco_2023_or_singapore_2022_zero_stop_count={len(monaco_singapore)}")

    print("\nStep 3 - Recompute only the 3 affected Power rows")
    before_rollups = _compute_rollups(cache_rows)
    before_alpine = _extract_alpine_power_cells(before_rollups["team_archetype"])

    cache_rows, updates = _recompute_power_targets(cache_rows)
    print(f"targeted_rows_recomputed={len(updates)}")
    print("targeted_updates_begin")
    for row in updates:
        print("targeted_update=" + json.dumps(row, sort_keys=True))
    print("targeted_updates_end")

    print("\nStep 4 - Patch cache + rerollup")
    POWER_CACHE_PATH.write_text(json.dumps(cache_rows, indent=2), encoding="utf-8")
    after_rollups = _compute_rollups(cache_rows)
    after_alpine = _extract_alpine_power_cells(after_rollups["team_archetype"])

    alpine_delta_rows: List[Dict[str, Any]] = []
    for lam in sorted(set(before_alpine.keys()) | set(after_alpine.keys())):
        b = before_alpine.get(lam)
        a = after_alpine.get(lam)
        if b is None or a is None:
            continue
        alpine_delta_rows.append(
            {
                "lambda": float(lam),
                "before_avg_total_cost": float(b["avg_total_cost"]),
                "after_avg_total_cost": float(a["avg_total_cost"]),
                "delta_avg_total_cost": float(a["avg_total_cost"] - b["avg_total_cost"]),
                "before_avg_cost_anchoring": float(b["avg_cost_anchoring"]),
                "after_avg_cost_anchoring": float(a["avg_cost_anchoring"]),
                "delta_avg_cost_anchoring": float(a["avg_cost_anchoring"] - b["avg_cost_anchoring"]),
                "before_avg_cost_sc_underweighting": float(b["avg_cost_sc_underweighting"]),
                "after_avg_cost_sc_underweighting": float(a["avg_cost_sc_underweighting"]),
                "delta_avg_cost_sc_underweighting": float(a["avg_cost_sc_underweighting"] - b["avg_cost_sc_underweighting"]),
                "before_avg_cost_conservatism": float(b["avg_cost_conservatism"]),
                "after_avg_cost_conservatism": float(a["avg_cost_conservatism"]),
                "delta_avg_cost_conservatism": float(a["avg_cost_conservatism"] - b["avg_cost_conservatism"]),
                "before_avg_cost_unattributed": float(b["avg_cost_unattributed"]),
                "after_avg_cost_unattributed": float(a["avg_cost_unattributed"]),
                "delta_avg_cost_unattributed": float(a["avg_cost_unattributed"] - b["avg_cost_unattributed"]),
            }
        )

    checks = after_rollups["checks"]
    print("sanity_checks=" + json.dumps(checks, sort_keys=True))
    print("alpine_power_delta_begin")
    for row in alpine_delta_rows:
        print("alpine_power_delta_row=" + json.dumps(row, sort_keys=True))
    print("alpine_power_delta_end")

    rollup_output_lines: List[str] = []
    rollup_output_lines.append("=" * 100)
    rollup_output_lines.append("Power cache recomputed rollup after targeted zero-stop fix rows")
    rollup_output_lines.append("=" * 100)
    rollup_output_lines.append("sanity_checks=" + json.dumps(checks, sort_keys=True))
    rollup_output_lines.append("alpine_power_delta_begin")
    for row in alpine_delta_rows:
        rollup_output_lines.append("alpine_power_delta_row=" + json.dumps(row, sort_keys=True))
    rollup_output_lines.append("alpine_power_delta_end")
    ROLLUP_OUTPUT_PATH.write_text("\n".join(rollup_output_lines) + "\n", encoding="utf-8")

    print("\nStep 5 - Street implications")
    if monaco_singapore:
        print("street_implication=true")
        print("street_implication_detail=Locked Street sample includes zero-stop real strategies; wet-aware zero-stop inclusion is required before Street run.")
    else:
        print("street_implication=false")

    print("\nStep 6 - Completion")
    print(f"updated_cache_written={POWER_CACHE_PATH}")
    print(f"rerollup_output_written={ROLLUP_OUTPUT_PATH}")
    print("diagnostic_complete=true")


if __name__ == "__main__":
    main()
