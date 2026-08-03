from __future__ import annotations

import json
from pathlib import Path

from source_code.simulation.bias import partition_cost
from source_code.simulation.rollup import compute_driver_race_bias_summary, compute_driver_race_cost_partition


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "diagnostics" / "phase3_technical_archetype_full_cache.json"
OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step9d_cost_partition_mismatch_diagnosis_output.txt"
EPSILON = 1e-12


def emit(handle, text: str = "") -> None:
    print(text, flush=True)
    handle.write(text + "\n")
    handle.flush()


def main() -> None:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Cache file not found: {CACHE_PATH}")

    with CACHE_PATH.open("r", encoding="utf-8") as fh:
        cache_rows = json.load(fh)

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        emit(out, "=" * 100)
        emit(out, "Step 9d - Technical driver-race cost partition mismatch diagnosis")
        emit(out, "=" * 100)
        emit(out, f"cache_path={CACHE_PATH}")
        emit(out, f"cache_row_count={len(cache_rows)}")

        driver_bias_rows = compute_driver_race_bias_summary(
            cache_rows,
            epsilon=EPSILON,
            materiality_threshold=0.88,
        )
        partitioned_rows = compute_driver_race_cost_partition(driver_bias_rows)

        failing_rows = []
        sar_rows = []
        for row in partitioned_rows:
            total = float(row["total_cost"])
            summed = float(row["cost_conservatism"] + row["cost_anchoring"] + row["cost_sc_underweighting"])
            diff = summed - total
            if abs(diff) > EPSILON:
                failing_rows.append((row, summed, diff))

            race = row["race"]
            if str(row.get("subject_driver")) == "SAR" or (
                str(race.get("event_name")) == "Spanish Grand Prix"
                and int(race.get("year", -1)) == 2023
                and int(race.get("round", -1)) == 7
                and str(row.get("subject_team")) == "Williams"
            ):
                sar_rows.append((row, summed, diff))

        emit(out, f"failing_row_count={len(failing_rows)}")
        emit(out, "failing_rows_begin")
        for row, summed, diff in failing_rows:
            payload = {
                "race": row["race"],
                "subject_driver": row["subject_driver"],
                "subject_team": row.get("subject_team"),
                "lambda": row.get("lambda"),
                "total_cost": float(row["total_cost"]),
                "cost_sum": float(summed),
                "diff": float(diff),
                "conservatism": {
                    "cost": float(row["cost_conservatism"]),
                    "r_b": row.get("conservatism_r_b"),
                    "unidentifiable_strict": row.get("conservatism_unidentifiable_strict"),
                    "unidentifiable_materiality": row.get("conservatism_unidentifiable_materiality"),
                },
                "anchoring": {
                    "cost": float(row["cost_anchoring"]),
                    "r_b": row.get("anchoring_r_b"),
                    "unidentifiable_strict": row.get("anchoring_unidentifiable_strict_count"),
                    "unidentifiable_materiality": row.get("anchoring_unidentifiable_materiality_count"),
                },
                "sc_underweighting": {
                    "cost": float(row["cost_sc_underweighting"]),
                    "r_b": row.get("sc_underweighting_r_b"),
                    "unidentifiable_strict": row.get("sc_underweighting_unidentifiable_strict"),
                    "unidentifiable_materiality": row.get("sc_underweighting_unidentifiable_materiality"),
                },
            }
            emit(out, "failing_row=" + json.dumps(payload, sort_keys=True, default=str))
        emit(out, "failing_rows_end")

        emit(out, f"sar_related_row_count={len(sar_rows)}")
        emit(out, "sar_related_rows_begin")
        for row, summed, diff in sar_rows:
            payload = {
                "race": row["race"],
                "subject_driver": row["subject_driver"],
                "subject_team": row.get("subject_team"),
                "lambda": row.get("lambda"),
                "total_cost": float(row["total_cost"]),
                "cost_sum": float(summed),
                "diff": float(diff),
                "conservatism_r_b": row.get("conservatism_r_b"),
                "anchoring_r_b": row.get("anchoring_r_b"),
                "sc_underweighting_r_b": row.get("sc_underweighting_r_b"),
                "conservatism_unidentifiable_strict": row.get("conservatism_unidentifiable_strict"),
                "conservatism_unidentifiable_materiality": row.get("conservatism_unidentifiable_materiality"),
                "anchoring_unidentifiable_strict_count": row.get("anchoring_unidentifiable_strict_count"),
                "anchoring_unidentifiable_materiality_count": row.get("anchoring_unidentifiable_materiality_count"),
                "sc_underweighting_unidentifiable_strict": row.get("sc_underweighting_unidentifiable_strict"),
                "sc_underweighting_unidentifiable_materiality": row.get("sc_underweighting_unidentifiable_materiality"),
            }
            emit(out, "sar_related_row=" + json.dumps(payload, sort_keys=True, default=str))
        emit(out, "sar_related_rows_end")

        zero_total_rows = [row for row in partitioned_rows if abs(float(row["total_cost"])) <= EPSILON]
        emit(out, f"zero_total_cost_row_count={len(zero_total_rows)}")
        emit(out, "zero_total_cost_rows_begin")
        for row in zero_total_rows:
            payload = {
                "race": row["race"],
                "subject_driver": row["subject_driver"],
                "subject_team": row.get("subject_team"),
                "lambda": row.get("lambda"),
                "total_cost": float(row["total_cost"]),
                "cost_conservatism": float(row["cost_conservatism"]),
                "cost_anchoring": float(row["cost_anchoring"]),
                "cost_sc_underweighting": float(row["cost_sc_underweighting"]),
                "r_b_inputs": row.get("r_b_inputs"),
                "eligible_biases": row.get("eligible_biases"),
            }
            emit(out, "zero_total_cost_row=" + json.dumps(payload, sort_keys=True, default=str))
        emit(out, "zero_total_cost_rows_end")

        emit(out, "diagnostic_complete=true")


if __name__ == "__main__":
    main()