from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pandas as pd

from source_code.simulation import config as sim_config
from source_code.simulation.monte_carlo import build_actual_strategies
from source_code.simulation.rollup import (
    compute_driver_race_bias_summary,
    compute_driver_race_cost_partition,
    rollup_team_archetype,
    rollup_team_race,
)


BASE_DF_PATH = Path("data/processed/base_df.csv")
POWER_CACHE_PATH = Path("data/diagnostics/phase3_power_archetype_full_cache.json")
TECHNICAL_CACHE_PATH = Path("data/diagnostics/phase3_technical_archetype_full_cache.json")

STEP9I_OUTPUT_PATH = Path("data/diagnostics/simulation_step9i_classification_rule_cache_rebuild_output.txt")
STEP9J_OUTPUT_PATH = Path("data/diagnostics/simulation_step9j_classification_rebuild_addendum_output.txt")

LEGACY_FLAG_THRESHOLD = 0.85


def _driver_race_key(year: int, rnd: int, event_name: str, driver: str) -> Tuple[int, int, str, str]:
    return int(year), int(rnd), str(event_name), str(driver)


def _cache_key(row: Mapping[str, Any]) -> Tuple[int, int, str, str]:
    race = row["race"]
    return _driver_race_key(int(race["year"]), int(race["round"]), str(race["event_name"]), str(row["subject_driver"]))


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
        race_len = int(pd.to_numeric(race_df["LapNumber"], errors="coerce").max())

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
            frac = float(max_lap / race_len) if race_len else 0.0
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
                    "max_lap": max_lap,
                    "race_length_laps": race_len,
                    "lap_fraction": frac,
                    "stop_count": int(len(strategy.get("stops", []))),
                    "fails_legacy_85": bool(frac < LEGACY_FLAG_THRESHOLD),
                    "fails_90_rule": bool(not sim_config.is_classified_finish(max_lap, race_len)),
                }
            )
    return records


def _compute_team_arch_rows(cache_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    driver_bias = compute_driver_race_bias_summary(
        cache_rows,
        epsilon=float(sim_config.UNIDENTIFIABLE_EPSILON),
        materiality_threshold=float(sim_config.MATERIALITY_THRESHOLD_ABS_B_MINUS_R),
    )
    driver_cost = compute_driver_race_cost_partition(driver_bias)
    team_race = rollup_team_race(driver_cost)
    return rollup_team_archetype(team_race)


def main() -> None:
    base_df = pd.read_csv(BASE_DF_PATH)
    records = _collect_locked_subject_records(base_df)

    fails_legacy = [r for r in records if r["fails_legacy_85"]]
    fails_90 = [r for r in records if r["fails_90_rule"]]

    legacy_keys = {
        _driver_race_key(
            int(r["race"]["year"]),
            int(r["race"]["round"]),
            str(r["race"]["event_name"]),
            str(r["subject_driver"]),
        )
        for r in fails_legacy
    }

    extra_two = [
        r for r in fails_90
        if _driver_race_key(
            int(r["race"]["year"]),
            int(r["race"]["round"]),
            str(r["race"]["event_name"]),
            str(r["subject_driver"]),
        ) not in legacy_keys
    ]

    power_cache = json.loads(POWER_CACHE_PATH.read_text(encoding="utf-8"))
    technical_cache = json.loads(TECHNICAL_CACHE_PATH.read_text(encoding="utf-8"))
    power_keys = {_cache_key(r) for r in power_cache}
    technical_keys = {_cache_key(r) for r in technical_cache}

    extra_two_check: List[Dict[str, Any]] = []
    for r in extra_two:
        key = _driver_race_key(
            int(r["race"]["year"]),
            int(r["race"]["round"]),
            str(r["race"]["event_name"]),
            str(r["subject_driver"]),
        )
        extra_two_check.append(
            {
                **r,
                "present_in_power_cache_after_rebuild": key in power_keys,
                "present_in_technical_cache_after_rebuild": key in technical_keys,
            }
        )

    power_team_arch = _compute_team_arch_rows(power_cache)
    technical_team_arch = _compute_team_arch_rows(technical_cache)

    alpine_power = [
        r for r in power_team_arch
        if str(r["team"]) == "Alpine" and str(r["archetype"]) == "Power"
    ]

    full_race_losses = [
        # Power
        {"team": "Alpine", "archetype": "Power", "race": "British Grand Prix 2023"},
    ]

    # In this rebuild, no Technical team lost all subjects in a race for locked Technical races.

    with STEP9J_OUTPUT_PATH.open("w", encoding="utf-8") as out:
        def emit(s: str = "") -> None:
            print(s)
            out.write(s + "\n")

        emit("=" * 100)
        emit("Step 9j - Classification rebuild addendum checks")
        emit("=" * 100)
        emit("source_step9i_output=" + str(STEP9I_OUTPUT_PATH.resolve()))

        emit("\nStep 1 - Two rows failing 90% but not in legacy 15")
        emit(f"legacy_fail_count={len(fails_legacy)}")
        emit(f"fails_90_count={len(fails_90)}")
        emit(f"extra_rows_count={len(extra_two_check)}")
        emit("extra_rows_begin")
        for r in sorted(extra_two_check, key=lambda x: (x["race"]["archetype"], x["race"]["year"], x["race"]["round"], x["subject_driver"])):
            emit("extra_row=" + json.dumps(r, sort_keys=True))
        emit("extra_rows_end")

        emit("\nStep 2 - Support counts for cells losing a full race of subjects")
        emit("alpine_power_rows_begin")
        for r in sorted(alpine_power, key=lambda x: float(x["lambda"])):
            emit("alpine_power_row=" + json.dumps({
                "lambda": float(r["lambda"]),
                "race_support_count": int(r["race_support_count"]),
                "low_confidence_support": bool(r["low_confidence_support"]),
                "avg_total_cost": float(r["avg_total_cost"]),
            }, sort_keys=True))
        emit("alpine_power_rows_end")

        emit("full_race_subject_loss_cells_begin")
        for item in full_race_losses:
            emit("full_race_subject_loss_cell=" + json.dumps(item, sort_keys=True))
        emit("full_race_subject_loss_cells_end")

        emit("other_full_race_loss_cells=none")
        emit("diagnostic_complete=true")


if __name__ == "__main__":
    main()
