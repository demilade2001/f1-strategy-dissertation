from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation import config as sim_config  # noqa: E402
from source_code.simulation.race_state import load_race_state  # noqa: E402
from source_code.simulation.rollup import (  # noqa: E402
    compute_driver_race_bias_summary,
    compute_driver_race_cost_partition,
    rollup_team_archetype,
    rollup_team_race,
    sanity_checks,
)


DIAG_DIR = ROOT / "data" / "diagnostics"
CACHE_PATH = DIAG_DIR / "phase3_pilot_reference_cache.json"
OUTPUT_PATH = DIAG_DIR / "simulation_step8_rollup_mechanism_check_output.txt"


def emit(out_handle, text: str = "") -> None:
    print(text, flush=True)
    out_handle.write(text + "\n")
    out_handle.flush()


def _race_key(row: Dict[str, Any]) -> tuple[int, int, str]:
    race = row["race"]
    return (int(race["year"]), int(race["round"]), str(race["event_name"]))


def _subject_team_for_row(row: Dict[str, Any], race_team_cache: Dict[tuple[int, int, str], Dict[str, str]]) -> str:
    race = row["race"]
    subject_driver = str(row["subject_driver"])
    key = _race_key(row)

    if key not in race_team_cache:
        state = load_race_state(int(race["year"]), int(race["round"]))
        laps = state["laps"]
        driver_team: Dict[str, str] = {}
        for _, sub in laps[["Driver", "Team"]].dropna(subset=["Driver", "Team"]).drop_duplicates().iterrows():
            driver_team[str(sub["Driver"])] = str(sub["Team"])
        race_team_cache[key] = driver_team

    if subject_driver not in race_team_cache[key]:
        raise ValueError(f"Could not resolve team for driver={subject_driver} race={race}")
    return race_team_cache[key][subject_driver]


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        emit(out, "=" * 100)
        emit(out, "Step 8 - Rollup mechanism check (driver-race -> team-race -> team x archetype)")
        emit(out, "=" * 100)

        if not CACHE_PATH.exists():
            raise FileNotFoundError(f"Missing pilot cache: {CACHE_PATH}")

        cache_rows: List[Dict[str, Any]] = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        emit(out, f"pilot_cache_path={CACHE_PATH}")
        emit(out, f"pilot_cache_row_count={len(cache_rows)}")

        emit(out, "")
        emit(out, "Step 1 - Per-driver-race bias summary")
        driver_rows = compute_driver_race_bias_summary(
            cache_rows,
            epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
            materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
        )

        race_team_cache: Dict[tuple[int, int, str], Dict[str, str]] = {}
        for row in driver_rows:
            row["subject_team"] = _subject_team_for_row(row, race_team_cache)

            emit(
                out,
                "driver_race_bias_row="
                + json.dumps(
                    {
                        "race": row["race"],
                        "subject_driver": row["subject_driver"],
                        "subject_team": row["subject_team"],
                        "total_cost": row["total_cost"],
                        "conservatism_r_b": row["conservatism_r_b"],
                        "conservatism_identifiable_count": row["conservatism_identifiable_count"],
                        "anchoring_r_b": row["anchoring_r_b"],
                        "anchoring_identifiable_trigger_count": row["anchoring_identifiable_trigger_count"],
                        "anchoring_computable_trigger_count": row["anchoring_computable_trigger_count"],
                        "sc_underweighting_r_b": row["sc_underweighting_r_b"],
                        "sc_underweighting_identifiable_count": row["sc_underweighting_identifiable_count"],
                    },
                    sort_keys=True,
                    default=str,
                ),
            )

        emit(out, f"driver_race_bias_summary_count={len(driver_rows)}")

        emit(out, "")
        emit(out, "Step 2 - Per-driver-race cost partition")
        partitioned_rows = compute_driver_race_cost_partition(driver_rows)
        for row in partitioned_rows:
            emit(
                out,
                "driver_race_cost_row="
                + json.dumps(
                    {
                        "race": row["race"],
                        "subject_driver": row["subject_driver"],
                        "subject_team": row["subject_team"],
                        "total_cost": row["total_cost"],
                        "r_b_inputs": row["r_b_inputs"],
                        "eligible_biases": row["eligible_biases"],
                        "cost_conservatism": row["cost_conservatism"],
                        "cost_anchoring": row["cost_anchoring"],
                        "cost_sc_underweighting": row["cost_sc_underweighting"],
                        "cost_sum": row["cost_sum"],
                        "cost_sum_matches_total_cost": row["cost_sum_matches_total_cost"],
                    },
                    sort_keys=True,
                    default=str,
                ),
            )

        emit(out, f"driver_race_cost_rows_count={len(partitioned_rows)}")

        emit(out, "")
        emit(out, "Step 3 - Team-race rollup")
        emit(
            out,
            "team_race_rollup_note=Mechanism-only validation: this 12-race pilot has exactly one subject driver per team per race, so team-race equals driver-race here. Full two-driver team-race aggregation is deferred to run.py.",
        )
        team_race_rows = rollup_team_race(partitioned_rows)
        for row in team_race_rows:
            emit(out, "team_race_row=" + json.dumps(row, sort_keys=True, default=str))
        emit(out, f"team_race_row_count={len(team_race_rows)}")

        max_driver_count_per_team_race = max(int(r["driver_race_count"]) for r in team_race_rows) if team_race_rows else 0
        emit(out, f"team_race_max_driver_count_in_pilot={max_driver_count_per_team_race}")

        emit(out, "")
        emit(out, "Step 4 - Team x archetype rollup")
        team_arch_rows = rollup_team_archetype(team_race_rows)
        for row in team_arch_rows:
            emit(out, "team_archetype_row=" + json.dumps(row, sort_keys=True, default=str))
        emit(out, f"team_archetype_row_count={len(team_arch_rows)}")

        low_conf_rows = [row for row in team_arch_rows if bool(row["low_confidence_support"])]
        emit(out, f"team_archetype_low_confidence_cell_count={len(low_conf_rows)}")
        for row in low_conf_rows:
            emit(
                out,
                "team_archetype_low_confidence_cell="
                + json.dumps(
                    {
                        "team": row["team"],
                        "archetype": row["archetype"],
                        "race_support_count": row["race_support_count"],
                    },
                    sort_keys=True,
                    default=str,
                ),
            )

        emit(out, "")
        emit(out, "Step 5 - Sanity checks")
        checks = sanity_checks(partitioned_rows, team_race_rows, team_arch_rows)
        emit(out, "sanity_checks=" + json.dumps(checks, sort_keys=True, default=str))

        emit(out, "")
        emit(out, "Step 6 - Mechanism-check completion")
        emit(out, f"output_file={OUTPUT_PATH}")

    print(f"output_file={OUTPUT_PATH}")
    print(f"output_file_bytes={OUTPUT_PATH.stat().st_size if OUTPUT_PATH.exists() else 0}")


if __name__ == "__main__":
    main()
