"""Rollup utilities for Phase 3 bias decomposition outputs.

This module intentionally operates on cached pilot outputs so aggregation checks
can run quickly without re-running Monte Carlo simulation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .bias import compute_r_b, partition_cost
from .references import r_conservatism
from src.utils import canonical_constructor_group


BIAS_KEYS = ("conservatism", "anchoring", "sc_underweighting")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def _is_materially_identifiable(metrics: Mapping[str, Any]) -> bool:
    return (
        not bool(metrics.get("unidentifiable_strict", False))
        and not bool(metrics.get("unidentifiable_materiality", False))
        and _is_finite_number(metrics.get("r_b"))
    )


def _race_identity(row: Mapping[str, Any]) -> Dict[str, Any]:
    race = row["race"]
    return {
        "year": int(race["year"]),
        "round": int(race["round"]),
        "event_name": str(race["event_name"]),
        "archetype": str(race["archetype"]),
    }


def compute_driver_race_bias_summary(
    cache_rows: Iterable[Mapping[str, Any]],
    epsilon: float,
    materiality_threshold: float,
) -> List[Dict[str, Any]]:
    """Build one bias-summary row per driver-race from cached pilot references.

    Anchoring is averaged over materially-identifiable trigger rows only.
    """

    summary_rows: List[Dict[str, Any]] = []

    for row in cache_rows:
        race = _race_identity(row)
        subject_driver = str(row["subject_driver"])

        x_points = float(row["X_actual_mean_points"])
        b_points = float(row["B_unconditioned_mean_points"])
        r_sc_points = float(row["R_SC_static_prior_mean_points"])
        total_cost_raw = b_points - x_points
        total_cost = max(0.0, total_cost_raw)

        # Conservatism (race-level, lap-based)
        cons_r_b = None
        cons_identifiable_count = 0
        cons_metrics = None
        x_lap = row.get("X_actual_first_pit_lap")
        b_lap = row.get("B_unconditioned_first_pit_lap")
        if x_lap is not None and b_lap is not None:
            cons_ref = r_conservatism(
                {
                    "race_length_laps": int(row["race_length_laps"]),
                    "subject_final_compound": None,
                },
                subject_driver,
            )
            r_lap = cons_ref.get("reference_lap")
            if r_lap is not None:
                cons_metrics = compute_r_b(
                    float(x_lap),
                    float(b_lap),
                    float(r_lap),
                    float(epsilon),
                    float(materiality_threshold),
                )
                if _is_materially_identifiable(cons_metrics):
                    cons_r_b = float(cons_metrics["r_b"])
                    cons_identifiable_count = 1

        # Anchoring (trigger-level; average over materially-identifiable rows)
        anchoring_identifiable_values: List[float] = []
        anchoring_computable_count = 0
        anchoring_strict_unidentifiable_count = 0
        anchoring_materiality_unidentifiable_count = 0

        for pair in row.get("anchoring_pairs", []):
            if not bool(pair.get("computable", False)):
                continue
            anchoring_computable_count += 1
            a_metrics = compute_r_b(
                x_points,
                float(pair["B_anchoring_mean_points"]),
                float(pair["R_anchoring_mean_points"]),
                float(epsilon),
                float(materiality_threshold),
            )
            if bool(a_metrics.get("unidentifiable_strict", False)):
                anchoring_strict_unidentifiable_count += 1
            if bool(a_metrics.get("unidentifiable_materiality", False)):
                anchoring_materiality_unidentifiable_count += 1
            if _is_materially_identifiable(a_metrics):
                anchoring_identifiable_values.append(float(a_metrics["r_b"]))

        anchoring_identifiable_count = len(anchoring_identifiable_values)
        anchoring_r_b = (
            float(mean(anchoring_identifiable_values))
            if anchoring_identifiable_values
            else None
        )

        # SC underweighting (race-level points-based)
        sc_metrics = compute_r_b(
            x_points,
            b_points,
            r_sc_points,
            float(epsilon),
            float(materiality_threshold),
        )
        sc_r_b = float(sc_metrics["r_b"]) if _is_materially_identifiable(sc_metrics) else None
        sc_identifiable_count = 1 if sc_r_b is not None else 0

        summary_rows.append(
            {
                "race": race,
                "subject_driver": subject_driver,
                "subject_team": row.get("subject_team"),
                "lambda": float(row.get("lambda", 1.0)),
                "total_cost_raw": total_cost_raw,
                "total_cost": total_cost,
                "X_actual_mean_points": x_points,
                "B_unconditioned_mean_points": b_points,
                "R_SC_static_prior_mean_points": r_sc_points,
                "conservatism_r_b": cons_r_b,
                "conservatism_identifiable_count": cons_identifiable_count,
                "conservatism_unidentifiable_strict": (
                    bool(cons_metrics.get("unidentifiable_strict", False)) if cons_metrics else None
                ),
                "conservatism_unidentifiable_materiality": (
                    bool(cons_metrics.get("unidentifiable_materiality", False)) if cons_metrics else None
                ),
                "anchoring_r_b": anchoring_r_b,
                "anchoring_identifiable_trigger_count": anchoring_identifiable_count,
                "anchoring_computable_trigger_count": anchoring_computable_count,
                "anchoring_unidentifiable_strict_count": anchoring_strict_unidentifiable_count,
                "anchoring_unidentifiable_materiality_count": anchoring_materiality_unidentifiable_count,
                "sc_underweighting_r_b": sc_r_b,
                "sc_underweighting_identifiable_count": sc_identifiable_count,
                "sc_underweighting_unidentifiable_strict": bool(
                    sc_metrics.get("unidentifiable_strict", False)
                ),
                "sc_underweighting_unidentifiable_materiality": bool(
                    sc_metrics.get("unidentifiable_materiality", False)
                ),
            }
        )

    summary_rows.sort(
        key=lambda r: (
            int(r["race"]["year"]),
            int(r["race"]["round"]),
            str(r["race"]["event_name"]),
            str(r["subject_driver"]),
        )
    )
    return summary_rows


def compute_driver_race_cost_partition(
    driver_race_rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Allocate each driver-race total cost across eligible bias components."""

    out_rows: List[Dict[str, Any]] = []

    for row in driver_race_rows:
        total_cost = float(row["total_cost"])
        r_b_by_bias = {
            "conservatism": row.get("conservatism_r_b"),
            "anchoring": row.get("anchoring_r_b"),
            "sc_underweighting": row.get("sc_underweighting_r_b"),
        }

        eligible_inputs: Dict[str, Dict[str, Any]] = {}
        for bias_name, r_b_value in r_b_by_bias.items():
            if _is_finite_number(r_b_value) and float(r_b_value) < 1.0:
                eligible_inputs[bias_name] = {
                    "r_b": float(r_b_value),
                    "unidentifiable": False,
                }

        split = partition_cost(total_cost, eligible_inputs)
        costs = {bias_name: float(split.get(bias_name, 0.0)) for bias_name in BIAS_KEYS}
        unattributed_cost = float(split.get("cost_unattributed", 0.0))
        split_sum = sum(costs.values()) + unattributed_cost

        row_copy = dict(row)
        row_copy["r_b_inputs"] = r_b_by_bias
        row_copy["eligible_biases"] = sorted(eligible_inputs.keys())
        row_copy["cost_conservatism"] = costs["conservatism"]
        row_copy["cost_anchoring"] = costs["anchoring"]
        row_copy["cost_sc_underweighting"] = costs["sc_underweighting"]
        row_copy["cost_unattributed"] = unattributed_cost
        row_copy["cost_sum"] = split_sum
        row_copy["cost_sum_matches_total_cost"] = abs(split_sum - total_cost) < 1e-9
        out_rows.append(row_copy)

    return out_rows


def rollup_team_race(driver_race_rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate driver-race costs to team-race level via summation."""

    agg: Dict[Tuple[int, int, str, str, str, float], Dict[str, Any]] = {}

    for row in driver_race_rows:
        race = row["race"]
        team = row.get("subject_team")
        if team is None:
            raise ValueError("subject_team must be populated before team-race rollup")

        canonical_team = canonical_constructor_group(team)

        key = (
            int(race["year"]),
            int(race["round"]),
            str(race["event_name"]),
            str(race["archetype"]),
            str(canonical_team),
            float(row.get("lambda", 1.0)),
        )
        if key not in agg:
            agg[key] = {
                "race": {
                    "year": int(race["year"]),
                    "round": int(race["round"]),
                    "event_name": str(race["event_name"]),
                    "archetype": str(race["archetype"]),
                },
                "team": str(canonical_team),
                "lambda": float(row.get("lambda", 1.0)),
                "driver_race_count": 0,
                "total_cost": 0.0,
                "cost_conservatism": 0.0,
                "cost_anchoring": 0.0,
                "cost_sc_underweighting": 0.0,
                "cost_unattributed": 0.0,
                "source_teams": set(),
            }

        agg_row = agg[key]
        agg_row["source_teams"].add(str(team))
        agg_row["driver_race_count"] += 1
        agg_row["total_cost"] += float(row["total_cost"])
        agg_row["cost_conservatism"] += float(row["cost_conservatism"])
        agg_row["cost_anchoring"] += float(row["cost_anchoring"])
        agg_row["cost_sc_underweighting"] += float(row["cost_sc_underweighting"])
        agg_row["cost_unattributed"] += float(row.get("cost_unattributed", 0.0))

    rows = []
    for row in agg.values():
        row_copy = dict(row)
        row_copy["source_teams"] = sorted(row_copy["source_teams"])
        rows.append(row_copy)
    rows.sort(
        key=lambda r: (
            float(r.get("lambda", 1.0)),
            int(r["race"]["year"]),
            int(r["race"]["round"]),
            str(r["team"]),
        )
    )
    return rows


def rollup_team_archetype(team_race_rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Average team-race costs to team-by-archetype cells."""

    groups: Dict[Tuple[str, str, float], List[Mapping[str, Any]]] = defaultdict(list)
    for row in team_race_rows:
        key = (
            str(row["team"]),
            str(row["race"]["archetype"]),
            float(row.get("lambda", 1.0)),
        )
        groups[key].append(row)

    out: List[Dict[str, Any]] = []
    for (team, archetype, lambda_value), rows in groups.items():
        support = len(rows)
        out.append(
            {
                "team": team,
                "archetype": archetype,
                "lambda": float(lambda_value),
                "race_support_count": support,
                "low_confidence_support": bool(support == 1),
                "avg_total_cost": float(mean(float(r["total_cost"]) for r in rows)),
                "avg_cost_conservatism": float(mean(float(r["cost_conservatism"]) for r in rows)),
                "avg_cost_anchoring": float(mean(float(r["cost_anchoring"]) for r in rows)),
                "avg_cost_sc_underweighting": float(mean(float(r["cost_sc_underweighting"]) for r in rows)),
                "avg_cost_unattributed": float(mean(float(r["cost_unattributed"]) for r in rows)),
            }
        )

    out.sort(key=lambda r: (float(r.get("lambda", 1.0)), str(r["archetype"]), str(r["team"])))
    return out


def sanity_checks(
    driver_race_rows: Iterable[Mapping[str, Any]],
    team_race_rows: Iterable[Mapping[str, Any]],
    team_archetype_rows: Iterable[Mapping[str, Any]],
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    """Run accounting and non-negativity checks across rollup levels."""

    driver_rows = list(driver_race_rows)
    team_rows = list(team_race_rows)
    archetype_rows = list(team_archetype_rows)

    driver_non_negative = all(
        float(row["total_cost"]) >= -tolerance
        and float(row["cost_conservatism"]) >= -tolerance
        and float(row["cost_anchoring"]) >= -tolerance
        and float(row["cost_sc_underweighting"]) >= -tolerance
        and float(row.get("cost_unattributed", 0.0)) >= -tolerance
        for row in driver_rows
    )
    team_non_negative = all(
        float(row["total_cost"]) >= -tolerance
        and float(row["cost_conservatism"]) >= -tolerance
        and float(row["cost_anchoring"]) >= -tolerance
        and float(row["cost_sc_underweighting"]) >= -tolerance
        and float(row.get("cost_unattributed", 0.0)) >= -tolerance
        for row in team_rows
    )
    archetype_non_negative = all(
        float(row["avg_total_cost"]) >= -tolerance
        and float(row["avg_cost_conservatism"]) >= -tolerance
        and float(row["avg_cost_anchoring"]) >= -tolerance
        and float(row["avg_cost_sc_underweighting"]) >= -tolerance
        and float(row["avg_cost_unattributed"]) >= -tolerance
        for row in archetype_rows
    )

    per_driver_sum_ok = all(
        abs(
            (
                float(row["cost_conservatism"])
                + float(row["cost_anchoring"])
                + float(row["cost_sc_underweighting"])
                + float(row.get("cost_unattributed", 0.0))
            )
            - float(row["total_cost"])
        )
        <= tolerance
        for row in driver_rows
    )

    driver_count_in = len(driver_rows)
    driver_count_into_team = int(sum(int(row["driver_race_count"]) for row in team_rows))
    team_row_count = len(team_rows)
    team_count_into_archetype = int(sum(int(row["race_support_count"]) for row in archetype_rows))

    return {
        "costs_non_negative": bool(driver_non_negative and team_non_negative and archetype_non_negative),
        "driver_race_cost_sum_matches_total_cost": bool(per_driver_sum_ok),
        "row_count_accounting": {
            "driver_race_rows_in": driver_count_in,
            "driver_race_support_summed_at_team_race": driver_count_into_team,
            "team_race_rows_in": team_row_count,
            "team_race_support_summed_at_team_archetype": team_count_into_archetype,
            "driver_to_team_accounting_ok": bool(driver_count_in == driver_count_into_team),
            "team_to_archetype_accounting_ok": bool(team_row_count == team_count_into_archetype),
        },
    }
