from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.simulation.config import GRID_SPACING
from src.simulation.strategy import enumerate_feasible_strategies


POWER_CACHE_PATH = Path("data/diagnostics/phase3_power_archetype_full_cache.json")
TECHNICAL_CACHE_PATH = Path("data/diagnostics/phase3_technical_archetype_full_cache.json")

TARGET_DRIVER = "OCO"
TARGET_EVENT = "British Grand Prix"
TARGET_YEAR = 2023


def _load_cache(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_stops(strategy: Dict[str, object]) -> List[Tuple[int, str]]:
    stops = strategy.get("stops", [])
    out: List[Tuple[int, str]] = []
    for stop in stops:
        lap, compound = stop
        out.append((int(lap), str(compound)))
    return out


def _nearest_grid_points(lap: int, race_length: int, grid_spacing: int) -> Dict[str, object]:
    grid = list(range(1, race_length, grid_spacing))
    nearest = min(grid, key=lambda x: abs(x - lap))
    best_dist = abs(nearest - lap)
    equidistant = [g for g in grid if abs(g - lap) == best_dist]
    return {
        "lap": int(lap),
        "nearest_distance": int(best_dist),
        "nearest_grid_points": [int(x) for x in equidistant],
    }


def _summarize_b_less_than_x(rows: List[dict], cache_name: str) -> Dict[str, object]:
    findings: List[Dict[str, object]] = []
    for row in rows:
        b = float(row["B_unconditioned_mean_points"])
        x = float(row["X_actual_mean_points"])
        gap = b - x
        if gap < 0:
            findings.append(
                {
                    "cache": cache_name,
                    "lambda": float(row["lambda"]),
                    "gap_b_minus_x": float(gap),
                    "abs_gap": float(abs(gap)),
                    "race": row["race"],
                    "subject_driver": str(row["subject_driver"]),
                    "subject_team": str(row["subject_team"]),
                    "x_stops": _strategy_stops(row["X_actual_strategy"]),
                    "b_stops": _strategy_stops(row["B_unconditioned_strategy"]),
                }
            )

    findings.sort(key=lambda r: r["gap_b_minus_x"])  # most negative first
    return {
        "cache": cache_name,
        "row_count": int(len(rows)),
        "b_less_than_x_count": int(len(findings)),
        "worst_gap": float(findings[0]["gap_b_minus_x"]) if findings else 0.0,
        "worst_rows": findings[:10],
        "all_findings": findings,
    }


def main() -> None:
    power_rows = _load_cache(POWER_CACHE_PATH)
    technical_rows = _load_cache(TECHNICAL_CACHE_PATH)

    print("=" * 100)
    print("Step 9f - B < X integrity diagnosis (cache-only)")
    print("=" * 100)
    print(f"power_cache_path={POWER_CACHE_PATH.resolve()}")
    print(f"technical_cache_path={TECHNICAL_CACHE_PATH.resolve()}")
    print(f"grid_spacing={int(GRID_SPACING)}")
    print(f"power_row_count={len(power_rows)}")
    print(f"technical_row_count={len(technical_rows)}")

    print("\nStep 1 - Check whether OCO British 2023 actual strategy is on the grid")
    target_rows = [
        r
        for r in power_rows
        if str(r.get("subject_driver")) == TARGET_DRIVER
        and str(r.get("race", {}).get("event_name")) == TARGET_EVENT
        and int(r.get("race", {}).get("year")) == TARGET_YEAR
    ]
    target_rows.sort(key=lambda r: float(r["lambda"]))
    print(f"target_rows_found={len(target_rows)}")

    if not target_rows:
        print("target_row_missing=true")
        return

    race_length = int(target_rows[0]["race_length_laps"])
    grid_points = list(range(1, race_length, int(GRID_SPACING)))
    x_stops = _strategy_stops(target_rows[0]["X_actual_strategy"])
    x_stop_laps = [lap for lap, _ in x_stops]

    print(f"target_race_length_laps={race_length}")
    print(f"target_grid_points_count={len(grid_points)}")
    print(f"target_grid_points_sample_start={grid_points[:10]}")
    print(f"target_grid_points_sample_end={grid_points[-10:]}")
    print(f"x_actual_stops={x_stops}")

    lap_distances = [_nearest_grid_points(lap, race_length, int(GRID_SPACING)) for lap in x_stop_laps]
    off_grid = [d for d in lap_distances if d["nearest_distance"] != 0]

    if lap_distances:
        for row in lap_distances:
            print("actual_lap_grid_distance=" + json.dumps(row, sort_keys=True))
    else:
        print("actual_lap_grid_distance=none (no pit stops in actual strategy)")

    print(f"actual_strategy_on_grid_via_pit_laps={len(off_grid) == 0}")

    print("\nStep 2 - Off-grid quantification (conditional)")
    if off_grid:
        print("off_grid_detected=true")
        print("note=Off-grid found, but cache does not persist full candidate-level mean_points for nearest-neighbor strategies.")
        print("note=Would require targeted re-score for those candidate strategies; not done in this cache-only run.")
    else:
        print("off_grid_detected=false")
        print("step2_result=not_applicable")

    print("\nStep 3 - On-grid inclusion and bug trace")
    # Key integrity check: the unconditioned enumeration only includes stop_count >= 1.
    # If X has zero stops, it cannot be represented in B's candidate pool.
    sample_starting_compound = str(target_rows[0]["X_actual_strategy"]["starting_compound"])
    enum_space = enumerate_feasible_strategies(
        race_length=race_length,
        starting_compound=sample_starting_compound,
        is_wet_race=False,
        max_stops=3,
    )
    if hasattr(enum_space, "materialize"):
        materialized = enum_space.materialize(limit=2_000_000)
    else:
        materialized = list(enum_space)

    no_stop_in_full_space = any(len(_strategy_stops(s)) == 0 for s in materialized)
    b_rows = []
    for row in target_rows:
        b = float(row["B_unconditioned_mean_points"])
        x = float(row["X_actual_mean_points"])
        b_rows.append(
            {
                "lambda": float(row["lambda"]),
                "B_unconditioned_mean_points": b,
                "X_actual_mean_points": x,
                "B_minus_X": float(b - x),
                "B_strategy_stops": _strategy_stops(row["B_unconditioned_strategy"]),
            }
        )
    print(f"enumerated_strategy_count={len(materialized)}")
    print(f"no_stop_strategy_present_in_enumerated_space={no_stop_in_full_space}")
    print("target_lambda_rows_begin")
    for row in b_rows:
        print("target_lambda_row=" + json.dumps(row, sort_keys=True))
    print("target_lambda_rows_end")
    if x_stop_laps == [] and not no_stop_in_full_space:
        print("step3_finding=BUG_SCOPE_CANDIDATE_GENERATION")
        print("step3_reason=X actual has zero stops, but B search space excludes zero-stop strategies by construction.")
    else:
        print("step3_finding=NO_CANDIDATE_GENERATION_BUG_DETECTED_FOR_TARGET")

    print("\nStep 4 - Broader integrity scan across Power + Technical")
    power_scan = _summarize_b_less_than_x(power_rows, "Power")
    technical_scan = _summarize_b_less_than_x(technical_rows, "Technical")

    combined_findings = power_scan["all_findings"] + technical_scan["all_findings"]
    combined_findings.sort(key=lambda r: r["gap_b_minus_x"])

    print("power_summary=" + json.dumps({
        "row_count": power_scan["row_count"],
        "b_less_than_x_count": power_scan["b_less_than_x_count"],
        "worst_gap": power_scan["worst_gap"],
    }, sort_keys=True))
    print("technical_summary=" + json.dumps({
        "row_count": technical_scan["row_count"],
        "b_less_than_x_count": technical_scan["b_less_than_x_count"],
        "worst_gap": technical_scan["worst_gap"],
    }, sort_keys=True))
    print(f"combined_b_less_than_x_count={len(combined_findings)}")

    print("combined_worst_rows_begin")
    for row in combined_findings[:15]:
        compact = {
            "cache": row["cache"],
            "lambda": row["lambda"],
            "gap_b_minus_x": row["gap_b_minus_x"],
            "race": row["race"],
            "subject_driver": row["subject_driver"],
            "subject_team": row["subject_team"],
            "x_stops": row["x_stops"],
            "b_stops": row["b_stops"],
        }
        print("combined_worst_row=" + json.dumps(compact, sort_keys=True))
    print("combined_worst_rows_end")

    print("\nStep 5 - Report recommendation (no code fix in this step)")
    if x_stop_laps == [] and not no_stop_in_full_space:
        print("recommendation=GENUINE_BUG")
        print("recommendation_detail=Fix candidate generation to include zero-stop strategies before proceeding to Street.")
    elif off_grid:
        print("recommendation=GRID_RESOLUTION_LIMITATION")
        print("recommendation_detail=Document Ch5 limitation: grid-search optimum is resolution-bound.")
    else:
        print("recommendation=NO_TARGET_BUG_CONFIRMED")

    print("\nStep 6 - Completion")
    print("diagnostic_complete=true")


if __name__ == "__main__":
    main()
