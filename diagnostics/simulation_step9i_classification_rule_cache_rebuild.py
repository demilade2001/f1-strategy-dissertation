from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import pandas as pd

from src.simulation import config as sim_config
from src.simulation.monte_carlo import build_actual_strategies
from src.simulation.rollup import (
    compute_driver_race_bias_summary,
    compute_driver_race_cost_partition,
    rollup_team_archetype,
    rollup_team_race,
    sanity_checks,
)
from src.simulation.run import derive_real_subject_roster


BASE_DF_PATH = Path("data/processed/base_df.csv")
POWER_CACHE_PATH = Path("data/diagnostics/phase3_power_archetype_full_cache.json")
TECHNICAL_CACHE_PATH = Path("data/diagnostics/phase3_technical_archetype_full_cache.json")

POWER_ROLLUP_OUTPUT_PATH = Path("data/diagnostics/phase3_power_archetype_full_cache_recomputed_rollup_output.txt")
TECHNICAL_ROLLUP_OUTPUT_PATH = Path("data/diagnostics/phase3_technical_archetype_full_cache_recomputed_rollup_output.txt")

STEP9I_OUTPUT_PATH = Path("data/diagnostics/simulation_step9i_classification_rule_cache_rebuild_output.txt")

LEGACY_FLAG_THRESHOLD = 0.85


def _race_key(year: int, rnd: int, event_name: str) -> Tuple[int, int, str]:
    return int(year), int(rnd), str(event_name)


def _driver_race_key(year: int, rnd: int, event_name: str, driver: str) -> Tuple[int, int, str, str]:
    return int(year), int(rnd), str(event_name), str(driver)


def _cache_driver_race_key(row: Mapping[str, Any]) -> Tuple[int, int, str, str]:
    race = row["race"]
    return _driver_race_key(int(race["year"]), int(race["round"]), str(race["event_name"]), str(row["subject_driver"]))


def _load_json(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _collect_locked_subject_records(base_df: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for race in sim_config.LOCKED_ARCHETYPE_RACES:
        year = int(race["year"])
        rnd = int(race["round"])
        event_name = str(race["event_name"])
        archetype = str(race["archetype"])

        race_df = base_df[(base_df["Year"] == year) & (base_df["Round"] == rnd)].copy()
        if race_df.empty:
            continue

        race_length = int(pd.to_numeric(race_df["LapNumber"], errors="coerce").max())
        team_map = {
            str(r.Driver): str(r.Team)
            for r in race_df[["Driver", "Team"]].dropna().drop_duplicates().itertuples(index=False)
        }
        actual = build_actual_strategies(race_df)

        for driver, strategy in actual.items():
            team = team_map.get(str(driver), "")
            if not sim_config.is_valid_subject(team):
                continue
            driver_df = race_df[race_df["Driver"].astype(str) == str(driver)]
            max_lap = int(pd.to_numeric(driver_df["LapNumber"], errors="coerce").max())
            frac = float(max_lap / race_length) if race_length else 0.0
            records.append(
                {
                    "race": {
                        "year": year,
                        "round": rnd,
                        "event_name": event_name,
                        "archetype": archetype,
                    },
                    "subject_driver": str(driver),
                    "subject_team": str(team),
                    "max_lap": int(max_lap),
                    "race_length_laps": int(race_length),
                    "lap_fraction": float(frac),
                    "passes_90pct_rule": bool(sim_config.is_classified_finish(max_lap, race_length)),
                    "stop_count": int(len(strategy.get("stops", []))),
                }
            )

    records.sort(
        key=lambda r: (
            str(r["race"]["archetype"]),
            int(r["race"]["year"]),
            int(r["race"]["round"]),
            str(r["subject_team"]),
            str(r["subject_driver"]),
        )
    )
    return records


def _filter_cache_rows_by_classification(
    cache_rows: List[Dict[str, Any]],
    classified_keys: set[Tuple[int, int, str, str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for row in cache_rows:
        if _cache_driver_race_key(row) in classified_keys:
            kept.append(row)
        else:
            removed.append(row)
    return kept, removed


def _compute_rollup_bundle(cache_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    driver_bias = compute_driver_race_bias_summary(
        cache_rows,
        epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
        materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
    )
    driver_cost = compute_driver_race_cost_partition(driver_bias)
    team_race = rollup_team_race(driver_cost)
    team_arch = rollup_team_archetype(team_race)
    checks = sanity_checks(driver_cost, team_race, team_arch)
    return {
        "driver_cost": driver_cost,
        "team_race": team_race,
        "team_arch": team_arch,
        "checks": checks,
    }


def _team_arch_map(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, float], Dict[str, Any]]:
    return {
        (str(r["team"]), str(r["archetype"]), float(r["lambda"])): r
        for r in rows
    }


def _extract_single_driver_team_rows(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    driver_cost = bundle["driver_cost"]
    team_race = bundle["team_race"]
    by_team_race_lambda: Dict[Tuple[int, int, str, str, float], List[Dict[str, Any]]] = {}
    for row in driver_cost:
        race = row["race"]
        key = (
            int(race["year"]),
            int(race["round"]),
            str(race["event_name"]),
            str(row["subject_team"]),
            float(row.get("lambda", 1.0)),
        )
        by_team_race_lambda.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for tr in team_race:
        race = tr["race"]
        key = (
            int(race["year"]),
            int(race["round"]),
            str(race["event_name"]),
            str(tr["team"]),
            float(tr.get("lambda", 1.0)),
        )
        drivers = by_team_race_lambda.get(key, [])
        if int(tr.get("driver_race_count", 0)) == 1 and len(drivers) == 1:
            d = drivers[0]
            out.append(
                {
                    "race": race,
                    "team": str(tr["team"]),
                    "lambda": float(tr.get("lambda", 1.0)),
                    "driver": str(d["subject_driver"]),
                    "team_race_total_cost": float(tr["total_cost"]),
                    "driver_total_cost": float(d["total_cost"]),
                    "exact_match": bool(abs(float(tr["total_cost"]) - float(d["total_cost"])) < 1e-9),
                }
            )
    out.sort(key=lambda r: (float(r["lambda"]), int(r["race"]["year"]), int(r["race"]["round"]), str(r["team"])))
    return out


def _emit_rollup_summary(path: Path, archetype: str, bundle: Dict[str, Any], team_arch_changes: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append(f"{archetype} cache rollup after FIA 90% classified-subject exclusion")
    lines.append("=" * 100)
    lines.append("sanity_checks=" + json.dumps(bundle["checks"], sort_keys=True))
    lines.append("team_archetype_changes_begin")
    for row in team_arch_changes:
        lines.append("team_archetype_change=" + json.dumps(row, sort_keys=True))
    lines.append("team_archetype_changes_end")
    singles = _extract_single_driver_team_rows(bundle)
    lines.append("single_driver_team_race_rows_begin")
    for row in singles:
        lines.append("single_driver_team_race_row=" + json.dumps(row, sort_keys=True))
    lines.append("single_driver_team_race_rows_end")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_df = pd.read_csv(BASE_DF_PATH)
    power_old = _load_json(POWER_CACHE_PATH)
    technical_old = _load_json(TECHNICAL_CACHE_PATH)

    records = _collect_locked_subject_records(base_df)

    flagged15 = [r for r in records if float(r["lap_fraction"]) < LEGACY_FLAG_THRESHOLD]
    fails90 = [r for r in records if not bool(r["passes_90pct_rule"])]
    classified_keys = {
        _driver_race_key(
            int(r["race"]["year"]),
            int(r["race"]["round"]),
            str(r["race"]["event_name"]),
            str(r["subject_driver"]),
        )
        for r in records
        if bool(r["passes_90pct_rule"])
    }

    power_before_bundle = _compute_rollup_bundle(power_old)
    technical_before_bundle = _compute_rollup_bundle(technical_old)

    power_new, power_removed = _filter_cache_rows_by_classification(power_old, classified_keys)
    technical_new, technical_removed = _filter_cache_rows_by_classification(technical_old, classified_keys)

    _dump_json(POWER_CACHE_PATH, power_new)
    _dump_json(TECHNICAL_CACHE_PATH, technical_new)

    power_after_bundle = _compute_rollup_bundle(power_new)
    technical_after_bundle = _compute_rollup_bundle(technical_new)

    power_before_map = _team_arch_map(power_before_bundle["team_arch"])
    power_after_map = _team_arch_map(power_after_bundle["team_arch"])
    technical_before_map = _team_arch_map(technical_before_bundle["team_arch"])
    technical_after_map = _team_arch_map(technical_after_bundle["team_arch"])

    def _changes(before_map: Dict[Tuple[str, str, float], Dict[str, Any]], after_map: Dict[Tuple[str, str, float], Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        all_keys = sorted(set(before_map.keys()) | set(after_map.keys()), key=lambda x: (x[2], x[1], x[0]))
        for key in all_keys:
            b = before_map.get(key)
            a = after_map.get(key)
            if b is None or a is None:
                continue
            delta_total = float(a["avg_total_cost"] - b["avg_total_cost"])
            delta_cons = float(a["avg_cost_conservatism"] - b["avg_cost_conservatism"])
            delta_anchor = float(a["avg_cost_anchoring"] - b["avg_cost_anchoring"])
            delta_sc = float(a["avg_cost_sc_underweighting"] - b["avg_cost_sc_underweighting"])
            delta_unatt = float(a["avg_cost_unattributed"] - b["avg_cost_unattributed"])
            if any(abs(x) > 1e-12 for x in [delta_total, delta_cons, delta_anchor, delta_sc, delta_unatt]):
                out.append(
                    {
                        "team": str(a["team"]),
                        "archetype": str(a["archetype"]),
                        "lambda": float(a["lambda"]),
                        "before_avg_total_cost": float(b["avg_total_cost"]),
                        "after_avg_total_cost": float(a["avg_total_cost"]),
                        "delta_avg_total_cost": delta_total,
                        "delta_avg_cost_conservatism": delta_cons,
                        "delta_avg_cost_anchoring": delta_anchor,
                        "delta_avg_cost_sc_underweighting": delta_sc,
                        "delta_avg_cost_unattributed": delta_unatt,
                    }
                )
        return out

    power_changes = _changes(power_before_map, power_after_map)
    technical_changes = _changes(technical_before_map, technical_after_map)

    _emit_rollup_summary(POWER_ROLLUP_OUTPUT_PATH, "Power", power_after_bundle, power_changes)
    _emit_rollup_summary(TECHNICAL_ROLLUP_OUTPUT_PATH, "Technical", technical_after_bundle, technical_changes)

    roster_rows = derive_real_subject_roster(sim_config.LOCKED_ARCHETYPE_RACES)
    power_roster = [r for r in roster_rows if str(r["race"]["archetype"]) == "Power"]
    technical_roster = [r for r in roster_rows if str(r["race"]["archetype"]) == "Technical"]

    singapore_zero_stop = [
        r for r in records
        if str(r["race"]["event_name"]) == "Singapore Grand Prix"
        and int(r["race"]["year"]) == 2022
        and int(r["stop_count"]) == 0
    ]
    japan_zero_stop = [
        r for r in records
        if str(r["race"]["event_name"]) == "Japanese Grand Prix"
        and int(r["race"]["year"]) == 2022
        and int(r["stop_count"]) == 0
    ]

    with STEP9I_OUTPUT_PATH.open("w", encoding="utf-8") as out:
        def emit(s: str = "") -> None:
            print(s)
            out.write(s + "\n")

        emit("=" * 100)
        emit("Step 9i - FIA 90% classification exclusion rebuild")
        emit("=" * 100)
        emit(f"min_classified_distance_fraction={sim_config.MIN_CLASSIFIED_DISTANCE_FRACTION}")

        emit("\nStep 3 - Full list of 15 legacy-flagged driver-races and 90% pass/fail")
        emit(f"legacy_flag_threshold_fraction={LEGACY_FLAG_THRESHOLD}")
        emit(f"legacy_flagged_count={len(flagged15)}")
        emit("legacy_flagged_rows_begin")
        for row in flagged15:
            emit("legacy_flagged_row=" + json.dumps(row, sort_keys=True))
        emit("legacy_flagged_rows_end")
        emit(f"fails_90pct_count_across_locked_midfield={len(fails90)}")

        emit("\nStep 4 - Corrected rosters for Power and Technical")
        emit("power_roster_begin")
        for row in power_roster:
            emit("power_roster_row=" + json.dumps(row, sort_keys=True))
        emit("power_roster_end")
        emit("technical_roster_begin")
        for row in technical_roster:
            emit("technical_roster_row=" + json.dumps(row, sort_keys=True))
        emit("technical_roster_end")

        emit("\nStep 5 - Cache rebuild by exclusion")
        emit("power_cache_counts=" + json.dumps({"before": len(power_old), "after": len(power_new), "removed": len(power_removed)}, sort_keys=True))
        emit("technical_cache_counts=" + json.dumps({"before": len(technical_old), "after": len(technical_new), "removed": len(technical_removed)}, sort_keys=True))
        emit("power_removed_rows_begin")
        for row in power_removed:
            emit("power_removed_row=" + json.dumps({
                "race": row["race"],
                "subject_driver": row["subject_driver"],
                "subject_team": row["subject_team"],
                "lambda": row["lambda"],
            }, sort_keys=True))
        emit("power_removed_rows_end")
        emit("technical_removed_rows_begin")
        for row in technical_removed:
            emit("technical_removed_row=" + json.dumps({
                "race": row["race"],
                "subject_driver": row["subject_driver"],
                "subject_team": row["subject_team"],
                "lambda": row["lambda"],
            }, sort_keys=True))
        emit("technical_removed_rows_end")

        emit("\nStep 6 - Sanity checks on corrected caches")
        emit("power_sanity_checks=" + json.dumps(power_after_bundle["checks"], sort_keys=True))
        emit("technical_sanity_checks=" + json.dumps(technical_after_bundle["checks"], sort_keys=True))
        emit("power_single_driver_team_race_rows_begin")
        for row in _extract_single_driver_team_rows(power_after_bundle):
            emit("power_single_driver_team_race_row=" + json.dumps(row, sort_keys=True))
        emit("power_single_driver_team_race_rows_end")
        emit("technical_single_driver_team_race_rows_begin")
        for row in _extract_single_driver_team_rows(technical_after_bundle):
            emit("technical_single_driver_team_race_row=" + json.dumps(row, sort_keys=True))
        emit("technical_single_driver_team_race_rows_end")

        emit("\nStep 7 - Team x archetype cell changes")
        emit("power_team_archetype_changes_begin")
        for row in power_changes:
            emit("power_team_archetype_change=" + json.dumps(row, sort_keys=True))
        emit("power_team_archetype_changes_end")
        emit("technical_team_archetype_changes_begin")
        for row in technical_changes:
            emit("technical_team_archetype_change=" + json.dumps(row, sort_keys=True))
        emit("technical_team_archetype_changes_end")

        emit("\nStep 9 - Street/Japan implications under 90% rule")
        emit(f"singapore_2022_zero_stop_total={len(singapore_zero_stop)}")
        emit(f"singapore_2022_zero_stop_survive_90pct={sum(1 for r in singapore_zero_stop if r['passes_90pct_rule'])}")
        emit(f"japan_2022_zero_stop_total={len(japan_zero_stop)}")
        emit(f"japan_2022_zero_stop_survive_90pct={sum(1 for r in japan_zero_stop if r['passes_90pct_rule'])}")

        emit("\nArtifacts")
        emit(f"power_cache_path={POWER_CACHE_PATH.resolve()}")
        emit(f"technical_cache_path={TECHNICAL_CACHE_PATH.resolve()}")
        emit(f"power_rollup_output_path={POWER_ROLLUP_OUTPUT_PATH.resolve()}")
        emit(f"technical_rollup_output_path={TECHNICAL_ROLLUP_OUTPUT_PATH.resolve()}")
        emit("diagnostic_complete=true")


if __name__ == "__main__":
    main()
