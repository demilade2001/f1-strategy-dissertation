from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from source_code.simulation import config as sim_config
from source_code.simulation.bias import compute_r_b
from source_code.simulation.references import r_conservatism
from source_code.simulation.rollup import (
    compute_driver_race_bias_summary,
    compute_driver_race_cost_partition,
    rollup_team_archetype,
    rollup_team_race,
    sanity_checks,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "diagnostics"


CACHE_SPECS = [
    {
        "archetype": "Technical",
        "path": OUT_DIR / "phase3_technical_archetype_full_cache.json",
        "expected_rows": 162,
    },
    {
        "archetype": "Power",
        "path": OUT_DIR / "phase3_power_archetype_full_cache.json",
        "expected_rows": 153,
    },
    {
        "archetype": "Street",
        "path": OUT_DIR / "phase3_street_archetype_full_cache.json",
        "expected_rows": 135,
    },
]

MASTER_TABLE_PATH = OUT_DIR / "phase3_cross_archetype_synthesis.csv"
SIGNED_TABLE_PATH = OUT_DIR / "phase3_signed_bias_distributions.csv"
REPORT_PATH = OUT_DIR / "phase3_synthesis_output.txt"

APPROX_ONE_TOL = 1e-6


def _dominant_named_bias(row: pd.Series) -> str:
    candidates = {
        "conservatism": float(row["avg_cost_conservatism"]),
        "anchoring": float(row["avg_cost_anchoring"]),
        "sc_underweighting": float(row["avg_cost_sc_underweighting"]),
    }
    return max(candidates.items(), key=lambda kv: kv[1])[0]


def _signed_bucket(r_b: float, tol: float = APPROX_ONE_TOL) -> str:
    if r_b < (1.0 - tol):
        return "lt_1"
    if r_b > (1.0 + tol):
        return "gt_1"
    return "approx_1"


def _append_distribution_record(
    records: List[Dict[str, Any]],
    bias_name: str,
    archetype: str,
    scope: str,
    stats: Dict[str, Any],
) -> None:
    row = {
        "bias": bias_name,
        "archetype": archetype,
        "scope": scope,
    }
    row.update(stats)
    records.append(row)


def _distribution_stats(metrics: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = list(metrics)
    total = len(metrics)
    strict_count = sum(1 for m in metrics if bool(m["unidentifiable_strict"]))
    materiality_count = sum(1 for m in metrics if bool(m["unidentifiable_materiality"]))

    finite_identifiable = [
        m for m in metrics if not bool(m["unidentifiable_strict"]) and pd.notna(m.get("r_b"))
    ]

    bucket_counter = Counter(_signed_bucket(float(m["r_b"])) for m in finite_identifiable)

    materially_identifiable = [
        m
        for m in finite_identifiable
        if not bool(m["unidentifiable_materiality"])
    ]
    direction_counter = Counter(_signed_bucket(float(m["r_b"])) for m in materially_identifiable)

    if direction_counter["lt_1"] > direction_counter["gt_1"]:
        directional_skew = "toward_lt_1"
    elif direction_counter["gt_1"] > direction_counter["lt_1"]:
        directional_skew = "toward_gt_1"
    else:
        directional_skew = "balanced_or_tied"

    return {
        "total_rows": int(total),
        "rb_lt_1_count": int(bucket_counter["lt_1"]),
        "rb_gt_1_count": int(bucket_counter["gt_1"]),
        "rb_approx_1_count": int(bucket_counter["approx_1"]),
        "unidentifiable_strict_count": int(strict_count),
        "unidentifiable_materiality_count": int(materiality_count),
        "finite_identifiable_count": int(len(finite_identifiable)),
        "materially_identifiable_count": int(len(materially_identifiable)),
        "directional_skew_materially_identifiable": directional_skew,
        "materially_identifiable_lt_1_count": int(direction_counter["lt_1"]),
        "materially_identifiable_gt_1_count": int(direction_counter["gt_1"]),
        "materially_identifiable_approx_1_count": int(direction_counter["approx_1"]),
    }


def _sc_metrics(cache_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in cache_rows:
        b = row.get("B_unconditioned_mean_points")
        r_sc = row.get("R_SC_static_prior_mean_points")
        x = row.get("X_actual_mean_points")
        if b is None or r_sc is None or x is None:
            continue
        m = compute_r_b(
            float(x),
            float(b),
            float(r_sc),
            epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
            materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
        )
        out.append(m)
    return out


def _conservatism_metrics(cache_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in cache_rows:
        x_lap = row.get("X_actual_first_pit_lap")
        b_lap = row.get("B_unconditioned_first_pit_lap")
        race_length = row.get("race_length_laps")
        if x_lap is None or b_lap is None or race_length is None:
            continue

        cons_ref = r_conservatism(
            {
                "race_length_laps": int(race_length),
                "subject_final_compound": None,
            },
            str(row["subject_driver"]),
        )
        r_lap = cons_ref.get("reference_lap")
        if r_lap is None:
            continue

        m = compute_r_b(
            float(x_lap),
            float(b_lap),
            float(r_lap),
            epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
            materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
        )
        out.append(m)
    return out


def _anchoring_metrics(cache_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in cache_rows:
        x = row.get("X_actual_mean_points")
        if x is None:
            continue
        for pair in row.get("anchoring_pairs", []):
            if not bool(pair.get("computable", False)):
                continue
            b_anchor = pair.get("B_anchoring_mean_points")
            r_anchor = pair.get("R_anchoring_mean_points")
            if b_anchor is None or r_anchor is None:
                continue
            m = compute_r_b(
                float(x),
                float(b_anchor),
                float(r_anchor),
                epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
                materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
            )
            out.append(m)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reports: List[str] = []
    reports.append("=" * 100)
    reports.append("Phase 3 cross-archetype synthesis and signed bias distributions")
    reports.append("=" * 100)

    all_team_arch_rows: List[Dict[str, Any]] = []
    signed_distribution_rows: List[Dict[str, Any]] = []

    cache_rows_by_archetype: Dict[str, List[Dict[str, Any]]] = {}

    reports.append("")
    reports.append("Step 1 - Fresh load and recompute for Technical, Power, Street")
    for spec in CACHE_SPECS:
        archetype = str(spec["archetype"])
        cache_path = Path(spec["path"])
        expected = int(spec["expected_rows"])

        cache_rows = json.loads(cache_path.read_text(encoding="utf-8"))
        actual = len(cache_rows)
        reports.append(
            f"cache_count_check={{\"archetype\": \"{archetype}\", \"path\": \"{cache_path}\", \"expected\": {expected}, \"actual\": {actual}, \"matches\": {str(actual == expected).lower()}}}"
        )
        if actual != expected:
            raise ValueError(
                f"{archetype} cache row count mismatch: expected {expected}, got {actual}"
            )

        cache_rows_by_archetype[archetype] = cache_rows

        bias_rows = compute_driver_race_bias_summary(
            cache_rows,
            epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
            materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
        )
        partitioned = compute_driver_race_cost_partition(bias_rows)
        team_race = rollup_team_race(partitioned)
        team_arch = rollup_team_archetype(team_race)
        checks = sanity_checks(partitioned, team_race, team_arch)

        for row in team_arch:
            row_copy = dict(row)
            row_copy["dominant_bias"] = _dominant_named_bias(pd.Series(row_copy))
            total = float(row_copy["avg_total_cost"])
            unattributed = float(row_copy["avg_cost_unattributed"])
            row_copy[
                "avg_cost_unattributed_fraction_of_avg_total_cost"
            ] = (unattributed / total) if total else 0.0
            all_team_arch_rows.append(row_copy)

        reports.append(
            "fresh_recompute_summary="
            + json.dumps(
                {
                    "archetype": archetype,
                    "driver_race_rows": len(partitioned),
                    "team_race_rows": len(team_race),
                    "team_archetype_rows": len(team_arch),
                    "sanity_checks": checks,
                },
                sort_keys=True,
                default=str,
            )
        )

        # Signed distributions per archetype.
        sc_stats = _distribution_stats(_sc_metrics(cache_rows))
        cons_stats = _distribution_stats(_conservatism_metrics(cache_rows))
        anch_stats = _distribution_stats(_anchoring_metrics(cache_rows))

        _append_distribution_record(
            signed_distribution_rows,
            "sc_underweighting",
            archetype,
            "per_archetype",
            sc_stats,
        )
        _append_distribution_record(
            signed_distribution_rows,
            "conservatism",
            archetype,
            "per_archetype",
            cons_stats,
        )
        _append_distribution_record(
            signed_distribution_rows,
            "anchoring",
            archetype,
            "per_archetype",
            anch_stats,
        )

    # Pooled signed distributions across all archetypes.
    pooled_rows = [r for rows in cache_rows_by_archetype.values() for r in rows]
    _append_distribution_record(
        signed_distribution_rows,
        "sc_underweighting",
        "Pooled",
        "pooled",
        _distribution_stats(_sc_metrics(pooled_rows)),
    )
    _append_distribution_record(
        signed_distribution_rows,
        "conservatism",
        "Pooled",
        "pooled",
        _distribution_stats(_conservatism_metrics(pooled_rows)),
    )
    _append_distribution_record(
        signed_distribution_rows,
        "anchoring",
        "Pooled",
        "pooled",
        _distribution_stats(_anchoring_metrics(pooled_rows)),
    )

    # Step 2 master table.
    master_df = pd.DataFrame(all_team_arch_rows)
    master_df = master_df[
        [
            "team",
            "archetype",
            "lambda",
            "avg_total_cost",
            "avg_cost_conservatism",
            "avg_cost_anchoring",
            "avg_cost_sc_underweighting",
            "avg_cost_unattributed",
            "avg_cost_unattributed_fraction_of_avg_total_cost",
            "race_support_count",
            "low_confidence_support",
            "dominant_bias",
        ]
    ].sort_values(["archetype", "team", "lambda"], ascending=[True, True, False])

    reports.append("")
    reports.append("Step 2 - Master team x archetype x lambda table")
    reports.append(master_df.to_csv(index=False).strip())

    # Step 3 descriptive comparison (effect sizes only).
    reports.append("")
    reports.append("Step 3 - Cross-archetype descriptive comparison (no significance testing)")
    for archetype, group in master_df.groupby("archetype", sort=True):
        mean_total = float(group["avg_total_cost"].mean())
        min_total = float(group["avg_total_cost"].min())
        max_total = float(group["avg_total_cost"].max())
        dominant_mode = str(group["dominant_bias"].value_counts().idxmax())
        reports.append(
            "descriptive_archetype_summary="
            + json.dumps(
                {
                    "archetype": archetype,
                    "mean_avg_total_cost_across_teams": mean_total,
                    "dominant_bias_most_often": dominant_mode,
                    "team_cost_range_min": min_total,
                    "team_cost_range_max": max_total,
                    "team_cost_range_span": max_total - min_total,
                    "note": "Descriptive effect sizes only; no p-values, CIs, or significance tests computed.",
                },
                sort_keys=True,
                default=str,
            )
        )

    signed_df = pd.DataFrame(signed_distribution_rows).sort_values(
        ["bias", "scope", "archetype"],
        ascending=[True, True, True],
    )

    reports.append("")
    reports.append("Step 4/5 - Signed bias distributions (SC, Conservatism, Anchoring)")
    reports.append(signed_df.to_csv(index=False).strip())
    reports.append(
        "directional_hypothesis_note=Directional skew is evaluated from materially identifiable rows (strict=false and materiality=false) by comparing counts with r_b<1 vs r_b>1."
    )

    master_df.to_csv(MASTER_TABLE_PATH, index=False)
    signed_df.to_csv(SIGNED_TABLE_PATH, index=False)
    REPORT_PATH.write_text("\n".join(reports) + "\n", encoding="utf-8")

    print(f"Wrote: {MASTER_TABLE_PATH}")
    print(f"Wrote: {SIGNED_TABLE_PATH}")
    print(f"Wrote: {REPORT_PATH}")


if __name__ == "__main__":
    main()
