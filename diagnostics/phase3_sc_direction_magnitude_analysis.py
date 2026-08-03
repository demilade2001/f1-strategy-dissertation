from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from source_code.simulation import config as sim_config
from source_code.simulation.bias import compute_r_b


ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = ROOT / "data" / "diagnostics"

CACHE_SPECS = [
    ("Technical", DIAG_DIR / "phase3_technical_archetype_full_cache.json", 162),
    ("Power", DIAG_DIR / "phase3_power_archetype_full_cache.json", 153),
    ("Street", DIAG_DIR / "phase3_street_archetype_full_cache.json", 135),
]

OUT_CSV = DIAG_DIR / "phase3_sc_direction_magnitude_analysis.csv"

EXPECTED_POOLED_MATERIAL = 81
EXPECTED_LT1 = 13
EXPECTED_GT1 = 68
RB_ONE_TOL = 1e-6


def _direction_from_rb(r_b: float) -> str:
    if r_b < (1.0 - RB_ONE_TOL):
        return "underweighting"
    if r_b > (1.0 + RB_ONE_TOL):
        return "overweighting"
    return "approx_equal"


def _safe_ratio(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return float("inf")
    return a / b


def _format_num(x: float) -> str:
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return f"{x:.6f}"


def _summary_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["count", "mean", "median", "min", "max"])
    out = (
        df.groupby(group_cols, dropna=False)["magnitude"]
        .agg(count="count", mean="mean", median="median", min="min", max="max")
        .reset_index()
    )
    return out


def _build_rows() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for archetype, cache_path, expected_rows in CACHE_SPECS:
        cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
        actual_rows = len(cache_data)
        if actual_rows != expected_rows:
            raise ValueError(
                f"Cache row count mismatch for {archetype}: expected {expected_rows}, got {actual_rows}"
            )

        for r in cache_data:
            x = r.get("X_actual_mean_points")
            b = r.get("B_unconditioned_mean_points")
            r_sc = r.get("R_SC_static_prior_mean_points")
            if x is None or b is None or r_sc is None:
                continue

            metrics = compute_r_b(
                float(x),
                float(b),
                float(r_sc),
                epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
                materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
            )

            rb = metrics.get("r_b")
            rb_finite = isinstance(rb, (int, float)) and not math.isnan(float(rb))
            direction = _direction_from_rb(float(rb)) if rb_finite else "nan"

            row = {
                "archetype": str(archetype),
                "year": int(r["race"]["year"]),
                "round": int(r["race"]["round"]),
                "event_name": str(r["race"]["event_name"]),
                "subject_driver": str(r["subject_driver"]),
                "subject_team": str(r.get("subject_team", "")),
                "lambda": float(r.get("lambda", 1.0)),
                "X_actual_mean_points": float(x),
                "B_unconditioned_mean_points": float(b),
                "R_SC_static_prior_mean_points": float(r_sc),
                "magnitude": abs(float(b) - float(x)),
                "r_b": float(rb) if rb_finite else float("nan"),
                "direction": direction,
                "unidentifiable_strict": bool(metrics.get("unidentifiable_strict", False)),
                "unidentifiable_materiality": bool(metrics.get("unidentifiable_materiality", False)),
            }
            row["materially_identifiable"] = (
                (not row["unidentifiable_strict"])
                and (not row["unidentifiable_materiality"])
                and rb_finite
            )
            rows.append(row)

    return pd.DataFrame(rows)


def _add_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Thresholds for archetype-direction groups (Step 3 means).
    med_arch = (
        out.groupby(["archetype", "direction"], dropna=False)["magnitude"].median().rename("median_arch")
    )
    out = out.merge(med_arch, on=["archetype", "direction"], how="left")
    out["outlier_vs_archetype_direction"] = out["magnitude"] > (3.0 * out["median_arch"])

    # Thresholds for pooled-direction groups (Step 3 pooled).
    med_pool = out.groupby(["direction"], dropna=False)["magnitude"].median().rename("median_pool")
    out = out.merge(med_pool, on=["direction"], how="left")
    out["outlier_vs_pooled_direction"] = out["magnitude"] > (3.0 * out["median_pool"])

    # Thresholds for team-direction groups (Step 4 means).
    med_team = (
        out.groupby(["subject_team", "direction"], dropna=False)["magnitude"].median().rename("median_team")
    )
    out = out.merge(med_team, on=["subject_team", "direction"], how="left")
    out["outlier_vs_team_direction"] = out["magnitude"] > (3.0 * out["median_team"])

    out["outlier_any"] = (
        out["outlier_vs_archetype_direction"]
        | out["outlier_vs_pooled_direction"]
        | out["outlier_vs_team_direction"]
    )

    return out


def _print_step_3(step3_df: pd.DataFrame) -> None:
    print("\nStep 3 - Magnitude by direction, per archetype and pooled")
    print("scope,group,direction,count,mean,median,min,max")
    for _, r in step3_df.iterrows():
        print(
            f"{r['scope']},{r['group']},{r['direction']},{int(r['count'])},"
            f"{_format_num(float(r['mean']))},{_format_num(float(r['median']))},"
            f"{_format_num(float(r['min']))},{_format_num(float(r['max']))}"
        )

    print("\nStep 3 - Direction comparison summaries")
    comp = []

    def _dir_value(frame: pd.DataFrame, direction: str, column: str) -> float:
        sub = frame[frame["direction"] == direction]
        if sub.empty:
            return float("nan")
        return float(sub[column].iloc[0])

    for scope, group in step3_df[["scope", "group"]].drop_duplicates().itertuples(index=False):
        g = step3_df[(step3_df["scope"] == scope) & (step3_df["group"] == group)]
        mean_under = _dir_value(g, "underweighting", "mean")
        mean_over = _dir_value(g, "overweighting", "mean")
        med_under = _dir_value(g, "underweighting", "median")
        med_over = _dir_value(g, "overweighting", "median")

        if math.isnan(mean_under) and not math.isnan(mean_over):
            mean_winner = "overweighting_only_present"
            mean_abs_diff = float("nan")
            mean_ratio = float("nan")
        elif math.isnan(mean_over) and not math.isnan(mean_under):
            mean_winner = "underweighting_only_present"
            mean_abs_diff = float("nan")
            mean_ratio = float("nan")
        elif mean_over > mean_under:
            mean_winner = "overweighting"
            mean_abs_diff = mean_over - mean_under
            mean_ratio = _safe_ratio(mean_over, mean_under)
        else:
            mean_winner = "underweighting"
            mean_abs_diff = mean_under - mean_over
            mean_ratio = _safe_ratio(mean_under, mean_over)

        if math.isnan(med_under) and not math.isnan(med_over):
            med_winner = "overweighting_only_present"
            med_abs_diff = float("nan")
            med_ratio = float("nan")
        elif math.isnan(med_over) and not math.isnan(med_under):
            med_winner = "underweighting_only_present"
            med_abs_diff = float("nan")
            med_ratio = float("nan")
        elif med_over > med_under:
            med_winner = "overweighting"
            med_abs_diff = med_over - med_under
            med_ratio = _safe_ratio(med_over, med_under)
        else:
            med_winner = "underweighting"
            med_abs_diff = med_under - med_over
            med_ratio = _safe_ratio(med_under, med_over)

        comp.append(
            {
                "scope": scope,
                "group": group,
                "mean_larger_direction": mean_winner,
                "mean_abs_diff": mean_abs_diff,
                "mean_ratio": mean_ratio,
                "median_larger_direction": med_winner,
                "median_abs_diff": med_abs_diff,
                "median_ratio": med_ratio,
            }
        )

    for r in comp:
        print(
            "direction_compare="
            + json.dumps(
                {
                    "scope": r["scope"],
                    "group": r["group"],
                    "mean_larger_direction": r["mean_larger_direction"],
                    "mean_abs_diff": round(float(r["mean_abs_diff"]), 6),
                    "mean_ratio": round(float(r["mean_ratio"]), 6),
                    "median_larger_direction": r["median_larger_direction"],
                    "median_abs_diff": round(float(r["median_abs_diff"]), 6),
                    "median_ratio": round(float(r["median_ratio"]), 6),
                },
                sort_keys=True,
            )
        )


def _print_step_4(step4_df: pd.DataFrame) -> None:
    print("\nStep 4 - Team-level pooled breakdown by direction")
    print("team,direction,count,mean_magnitude,low_sample_flag_lt3")
    for _, r in step4_df.sort_values(["subject_team", "direction"]).iterrows():
        print(
            f"{r['subject_team']},{r['direction']},{int(r['count'])},"
            f"{_format_num(float(r['mean']))},{str(bool(r['low_sample_flag_lt3'])).lower()}"
        )


def _print_step_6(
    step3_all: pd.DataFrame,
    step3_no_outliers: pd.DataFrame,
) -> None:
    def winner_info(df: pd.DataFrame, group: str) -> Tuple[str, float]:
        g = df[(df["scope"] == "archetype") & (df["group"] == group)]

        under_rows = g[g["direction"] == "underweighting"]
        over_rows = g[g["direction"] == "overweighting"]
        under = float(under_rows["mean"].iloc[0]) if not under_rows.empty else float("nan")
        over = float(over_rows["mean"].iloc[0]) if not over_rows.empty else float("nan")

        if math.isnan(under) and not math.isnan(over):
            return ("overweighting_only_present", float("nan"))
        if math.isnan(over) and not math.isnan(under):
            return ("underweighting_only_present", float("nan"))
        return ("overweighting" if over > under else "underweighting", abs(over - under))

    pooled_all = step3_all[(step3_all["scope"] == "pooled") & (step3_all["group"] == "Pooled")]
    pu = float(pooled_all[pooled_all["direction"] == "underweighting"]["mean"].iloc[0])
    po = float(pooled_all[pooled_all["direction"] == "overweighting"]["mean"].iloc[0])

    pooled_no = step3_no_outliers[
        (step3_no_outliers["scope"] == "pooled") & (step3_no_outliers["group"] == "Pooled")
    ]
    pu_no = float(pooled_no[pooled_no["direction"] == "underweighting"]["mean"].iloc[0])
    po_no = float(pooled_no[pooled_no["direction"] == "overweighting"]["mean"].iloc[0])

    print("\nStep 6 - Plain-language summary")
    print(
        "pooled_with_outliers="
        + json.dumps(
            {
                "larger_mean_direction": "overweighting" if po > pu else "underweighting",
                "underweighting_mean": round(pu, 6),
                "overweighting_mean": round(po, 6),
                "abs_diff": round(abs(po - pu), 6),
                "ratio_larger_to_smaller": round(_safe_ratio(max(po, pu), min(po, pu)), 6),
            },
            sort_keys=True,
        )
    )
    print(
        "pooled_without_outliers="
        + json.dumps(
            {
                "larger_mean_direction": "overweighting" if po_no > pu_no else "underweighting",
                "underweighting_mean": round(pu_no, 6),
                "overweighting_mean": round(po_no, 6),
                "abs_diff": round(abs(po_no - pu_no), 6),
                "ratio_larger_to_smaller": round(_safe_ratio(max(po_no, pu_no), min(po_no, pu_no)), 6),
            },
            sort_keys=True,
        )
    )

    for archetype in ["Power", "Street", "Technical"]:
        winner_all, diff_all = winner_info(step3_all, archetype)
        winner_no, diff_no = winner_info(step3_no_outliers, archetype)
        print(
            "archetype_consistency="
            + json.dumps(
                {
                    "archetype": archetype,
                    "larger_mean_direction_with_outliers": winner_all,
                    "larger_mean_direction_without_outliers": winner_no,
                    "abs_diff_with_outliers": float(diff_all) if not math.isnan(diff_all) else None,
                    "abs_diff_without_outliers": float(diff_no) if not math.isnan(diff_no) else None,
                },
                sort_keys=True,
            )
        )


def main() -> None:
    full_df = _build_rows()

    material = full_df[full_df["materially_identifiable"]].copy()
    material = material[material["direction"].isin(["underweighting", "overweighting"])].copy()

    pooled_count = len(material)
    lt1_count = int((material["direction"] == "underweighting").sum())
    gt1_count = int((material["direction"] == "overweighting").sum())

    print("Step 1 - Materially-identifiable SC rows check")
    print(
        "material_counts="
        + json.dumps(
            {
                "pooled_materially_identifiable_count": pooled_count,
                "underweighting_lt_1_count": lt1_count,
                "overweighting_gt_1_count": gt1_count,
                "expected": {
                    "pooled": EXPECTED_POOLED_MATERIAL,
                    "lt_1": EXPECTED_LT1,
                    "gt_1": EXPECTED_GT1,
                },
            },
            sort_keys=True,
        )
    )

    if (
        pooled_count != EXPECTED_POOLED_MATERIAL
        or lt1_count != EXPECTED_LT1
        or gt1_count != EXPECTED_GT1
    ):
        raise ValueError(
            "Step 1 verification failed: materially-identifiable SC counts do not match expected 81/13/68."
        )

    material = _add_outlier_flags(material)

    # Step 5 outlier report before mean interpretation.
    outliers = material[material["outlier_any"]].copy()
    print("\nStep 5 - Outlier check (magnitude > 3x median of direction/group)")
    print(f"outlier_row_count={len(outliers)}")
    if not outliers.empty:
        keep_cols = [
            "archetype",
            "year",
            "round",
            "event_name",
            "subject_driver",
            "subject_team",
            "direction",
            "magnitude",
            "r_b",
            "outlier_vs_pooled_direction",
            "outlier_vs_archetype_direction",
            "outlier_vs_team_direction",
        ]
        for _, r in outliers.sort_values(["archetype", "year", "round", "subject_driver"]).iterrows():
            payload = {k: r[k] for k in keep_cols}
            payload["magnitude"] = round(float(payload["magnitude"]), 6)
            payload["r_b"] = round(float(payload["r_b"]), 6)
            print("outlier_row=" + json.dumps(payload, sort_keys=True, default=str))

    # Step 3 stats: pooled + per archetype, by direction.
    pooled_tag = material.copy()
    pooled_tag["scope"] = "pooled"
    pooled_tag["group"] = "Pooled"
    arch_tag = material.copy()
    arch_tag["scope"] = "archetype"
    arch_tag["group"] = arch_tag["archetype"]
    step3_input = pd.concat([pooled_tag, arch_tag], ignore_index=True)
    step3_stats = _summary_stats(step3_input, ["scope", "group", "direction"])

    # Step 4 stats: pooled across archetypes by team + direction.
    step4_stats = _summary_stats(material, ["subject_team", "direction"])
    step4_stats["low_sample_flag_lt3"] = step4_stats["count"] < 3

    # Recompute Step 3 with outliers removed.
    material_no_outliers = material[~material["outlier_any"]].copy()
    pooled_tag_no = material_no_outliers.copy()
    pooled_tag_no["scope"] = "pooled"
    pooled_tag_no["group"] = "Pooled"
    arch_tag_no = material_no_outliers.copy()
    arch_tag_no["scope"] = "archetype"
    arch_tag_no["group"] = arch_tag_no["archetype"]
    step3_no_outliers = _summary_stats(
        pd.concat([pooled_tag_no, arch_tag_no], ignore_index=True),
        ["scope", "group", "direction"],
    )

    # Row-level output artifact.
    out_df = material.copy()
    out_df = out_df.sort_values(["archetype", "year", "round", "subject_team", "subject_driver", "lambda"])
    out_df.to_csv(OUT_CSV, index=False)

    _print_step_3(step3_stats)
    _print_step_4(step4_stats)
    _print_step_6(step3_stats, step3_no_outliers)

    print("\nSaved row-level artifact")
    print(str(OUT_CSV))


if __name__ == "__main__":
    main()
