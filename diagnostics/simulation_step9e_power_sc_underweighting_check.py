from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import config as sim_config  # noqa: E402
from src.simulation.bias import compute_r_b  # noqa: E402
from src.simulation.rollup import (  # noqa: E402
    compute_driver_race_bias_summary,
    compute_driver_race_cost_partition,
    rollup_team_archetype,
    rollup_team_race,
)


DIAG_DIR = ROOT / "data" / "diagnostics"
CACHE_PATH = DIAG_DIR / "phase3_power_archetype_full_cache.json"
OUTPUT_PATH = DIAG_DIR / "simulation_step9e_power_sc_underweighting_check_output.txt"


def emit(out_handle, text: str = "") -> None:
    print(text, flush=True)
    out_handle.write(text + "\n")
    out_handle.flush()


def _is_power_row(row: Dict[str, Any]) -> bool:
    race = row.get("race", {})
    return str(race.get("archetype", "")) == "Power"


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        emit(out, "=" * 100)
        emit(out, "Step 9e - Power SC Underweighting axis investigation")
        emit(out, "=" * 100)

        if not CACHE_PATH.exists():
            raise FileNotFoundError(f"Missing cache file: {CACHE_PATH}")

        cache_rows: List[Dict[str, Any]] = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        power_rows = [row for row in cache_rows if _is_power_row(row)]

        emit(out, f"cache_path={CACHE_PATH}")
        emit(out, f"cache_row_count={len(cache_rows)}")
        emit(out, f"power_row_count={len(power_rows)}")

        epsilon = float(sim_config.UNIDENTIFIABLE_EPSILON)
        materiality = float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R)
        emit(out, f"epsilon={epsilon}")
        emit(out, f"materiality_threshold_abs_B_minus_R={materiality}")

        # Step 1: recompute SC-axis metrics directly from cache values.
        sc_rows: List[Dict[str, Any]] = []
        for row in power_rows:
            x = float(row["X_actual_mean_points"])
            b = float(row["B_unconditioned_mean_points"])
            r_sc = float(row["R_SC_static_prior_mean_points"])
            metrics = compute_r_b(
                X=x,
                B=b,
                R_b=r_sc,
                epsilon=epsilon,
                materiality_threshold=materiality,
            )
            sc_rows.append(
                {
                    "race": row["race"],
                    "subject_driver": row["subject_driver"],
                    "subject_team": row.get("subject_team"),
                    "lambda": float(row.get("lambda", 1.0)),
                    "X": x,
                    "B": b,
                    "R_SC": r_sc,
                    "abs_B_minus_R_SC": abs(b - r_sc),
                    "r_b": metrics.get("r_b"),
                    "unidentifiable_strict": bool(metrics.get("unidentifiable_strict", False)),
                    "unidentifiable_materiality": bool(metrics.get("unidentifiable_materiality", False)),
                }
            )

        sc_rows.sort(
            key=lambda r: (
                float(r["lambda"]),
                int(r["race"]["year"]),
                int(r["race"]["round"]),
                str(r["subject_team"]),
                str(r["subject_driver"]),
            )
        )

        material_rows = [r for r in sc_rows if float(r["abs_B_minus_R_SC"]) >= materiality]
        emit(out, "")
        emit(out, "Step 1 - Rows clearing |B-R_SC| >= materiality threshold")
        emit(out, f"material_rows_count={len(material_rows)}")
        emit(out, "material_rows_begin")
        for row in material_rows:
            emit(out, "material_row=" + json.dumps(row, sort_keys=True, default=str))
        emit(out, "material_rows_end")

        # Step 2: classify material rows by effective SC-axis bias status.
        def classify(row: Dict[str, Any]) -> str:
            if row["unidentifiable_strict"] or row["unidentifiable_materiality"]:
                return "unidentifiable"
            r_b = row.get("r_b")
            if r_b is None:
                return "unidentifiable"
            return "r_b_lt_1" if float(r_b) < 1.0 else "r_b_ge_1"

        class_counts = {"r_b_lt_1": 0, "r_b_ge_1": 0, "unidentifiable": 0}
        for row in material_rows:
            class_counts[classify(row)] += 1

        emit(out, "")
        emit(out, "Step 2 - Classification of material rows")
        emit(out, "material_row_classification=" + json.dumps(class_counts, sort_keys=True))

        # Step 3: trace any potential break from per-row to rollup.
        driver_bias_rows = compute_driver_race_bias_summary(
            power_rows,
            epsilon=epsilon,
            materiality_threshold=materiality,
        )
        partitioned_rows = compute_driver_race_cost_partition(driver_bias_rows)
        team_race_rows = rollup_team_race(partitioned_rows)
        team_arch_rows = rollup_team_archetype(team_race_rows)

        sc_positive_driver_rows = [
            r for r in partitioned_rows if float(r.get("cost_sc_underweighting", 0.0)) > 0.0
        ]
        sc_positive_team_race_rows = [
            r for r in team_race_rows if float(r.get("cost_sc_underweighting", 0.0)) > 0.0
        ]
        sc_positive_team_arch_rows = [
            r for r in team_arch_rows if float(r.get("avg_cost_sc_underweighting", 0.0)) > 0.0
        ]

        emit(out, "")
        emit(out, "Step 3 - SC cost propagation checks")
        emit(out, f"driver_rows_with_positive_cost_sc_underweighting={len(sc_positive_driver_rows)}")
        emit(out, f"team_race_rows_with_positive_cost_sc_underweighting={len(sc_positive_team_race_rows)}")
        emit(out, f"team_archetype_rows_with_positive_avg_cost_sc_underweighting={len(sc_positive_team_arch_rows)}")

        if sc_positive_driver_rows:
            emit(out, "driver_rows_positive_sc_cost_begin")
            for row in sc_positive_driver_rows:
                emit(
                    out,
                    "driver_row_positive_sc_cost="
                    + json.dumps(
                        {
                            "race": row["race"],
                            "subject_driver": row["subject_driver"],
                            "subject_team": row.get("subject_team"),
                            "lambda": float(row.get("lambda", 1.0)),
                            "total_cost": float(row["total_cost"]),
                            "cost_sc_underweighting": float(row.get("cost_sc_underweighting", 0.0)),
                            "sc_underweighting_r_b": row.get("sc_underweighting_r_b"),
                            "r_b_inputs": row.get("r_b_inputs", {}),
                            "eligible_biases": row.get("eligible_biases", []),
                        },
                        sort_keys=True,
                        default=str,
                    ),
                )
            emit(out, "driver_rows_positive_sc_cost_end")

        emit(out, "")
        emit(out, "Step 4 - Team x archetype SC averages")
        emit(out, "team_archetype_rows_begin")
        for row in sorted(
            team_arch_rows,
            key=lambda r: (
                float(r.get("lambda", 1.0)),
                str(r.get("archetype", "")),
                str(r.get("team", "")),
            ),
        ):
            emit(
                out,
                "team_archetype_row="
                + json.dumps(
                    {
                        "team": row["team"],
                        "archetype": row["archetype"],
                        "lambda": float(row.get("lambda", 1.0)),
                        "avg_total_cost": float(row["avg_total_cost"]),
                        "avg_cost_sc_underweighting": float(row.get("avg_cost_sc_underweighting", 0.0)),
                        "avg_cost_conservatism": float(row.get("avg_cost_conservatism", 0.0)),
                        "avg_cost_anchoring": float(row.get("avg_cost_anchoring", 0.0)),
                        "avg_cost_unattributed": float(row.get("avg_cost_unattributed", 0.0)),
                    },
                    sort_keys=True,
                    default=str,
                ),
            )
        emit(out, "team_archetype_rows_end")

        emit(out, "")
        finding = {
            "material_rows_count": len(material_rows),
            "classification": class_counts,
            "positive_sc_cost_driver_rows": len(sc_positive_driver_rows),
            "positive_sc_cost_team_race_rows": len(sc_positive_team_race_rows),
            "positive_sc_cost_team_archetype_rows": len(sc_positive_team_arch_rows),
        }
        emit(out, "finding_summary=" + json.dumps(finding, sort_keys=True, default=str))
        emit(out, "diagnostic_complete=true")


if __name__ == "__main__":
    main()