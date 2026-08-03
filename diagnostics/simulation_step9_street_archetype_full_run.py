from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation import config as sim_config  # noqa: E402
from source_code.simulation.run import derive_real_subject_roster, run_phase3_archetype_first_pass  # noqa: E402


DIAG_DIR = ROOT / "data" / "diagnostics"
OUTPUT_PATH = DIAG_DIR / "simulation_step9_street_archetype_full_run_output.txt"
CACHE_PATH = DIAG_DIR / "phase3_street_archetype_full_cache.json"
PREFLIGHT_PATH = DIAG_DIR / "simulation_step9_street_preflight_output.txt"

STREET_ARCHETYPE = "Street"
# User-requested realized per-task bracket from completed archetypes.
LOWER_PER_TASK_SECONDS = 68.5
UPPER_PER_TASK_SECONDS = 150.6


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    locked_street = [
        dict(r)
        for r in sim_config.LOCKED_ARCHETYPE_RACES
        if str(r["archetype"]) == STREET_ARCHETYPE
    ]
    roster_rows = derive_real_subject_roster(locked_street)

    task_count = int(
        sum(int(row["valid_subject_driver_count"]) for row in roster_rows) * len(sim_config.LAMBDA_GRID)
    )
    workers = int(max(1, min(4, (os.cpu_count() or 1))))

    lower_wall_s = (task_count * LOWER_PER_TASK_SECONDS) / workers
    upper_wall_s = (task_count * UPPER_PER_TASK_SECONDS) / workers

    with PREFLIGHT_PATH.open("w", encoding="utf-8") as out:
        def emit(text: str = "") -> None:
            print(text)
            out.write(text + "\n")

        emit("=" * 100)
        emit("Street Preflight - Classification-gated roster and bracketed runtime estimate")
        emit("=" * 100)
        emit(f"street_locked_race_count={len(locked_street)}")
        emit(f"lambda_grid={list(sim_config.LAMBDA_GRID)}")
        emit(f"street_driver_race_count_after_classification={sum(int(r['valid_subject_driver_count']) for r in roster_rows)}")
        emit(f"street_task_count_driver_race_lambda={task_count}")
        emit(f"parallel_workers_assumed={workers}")
        emit(
            "runtime_bracket="
            + json.dumps(
                {
                    "lower_per_task_seconds_from_power": LOWER_PER_TASK_SECONDS,
                    "upper_per_task_seconds_from_technical": UPPER_PER_TASK_SECONDS,
                    "lower_projected_wall_seconds": lower_wall_s,
                    "upper_projected_wall_seconds": upper_wall_s,
                    "lower_projected_wall_minutes": lower_wall_s / 60.0,
                    "upper_projected_wall_minutes": upper_wall_s / 60.0,
                },
                sort_keys=True,
            )
        )
        for row in roster_rows:
            emit("street_preflight_roster_row=" + json.dumps(row, sort_keys=True, default=str))

    result = run_phase3_archetype_first_pass(
        target_archetype=STREET_ARCHETYPE,
        output_path=OUTPUT_PATH,
        cache_path=CACHE_PATH,
    )
    print("street_output_path=" + str(result["output_path"]))
    print("street_cache_path=" + str(result["cache_path"]))


if __name__ == "__main__":
    main()
