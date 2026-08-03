from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from source_code.simulation import config as sim_config
from source_code.simulation.monte_carlo import build_actual_strategies


BASE_DF_PATH = Path("data/processed/base_df.csv")
POWER_CACHE_PATH = Path("data/diagnostics/phase3_power_archetype_full_cache.json")
TECHNICAL_CACHE_PATH = Path("data/diagnostics/phase3_technical_archetype_full_cache.json")

RETIREMENT_THRESHOLD_FRACTION = 0.85
TARGET_OCO = (2023, 10, "British Grand Prix", "OCO")


def _load_base_df() -> pd.DataFrame:
    return pd.read_csv(BASE_DF_PATH)


def _race_key(race: Dict[str, Any]) -> Tuple[int, int, str]:
    return (int(race["year"]), int(race["round"]), str(race["event_name"]))


def _cache_driver_keys(rows: List[Dict[str, Any]]) -> set[Tuple[int, int, str, str]]:
    out = set()
    for row in rows:
        race = row["race"]
        out.add((int(race["year"]), int(race["round"]), str(race["event_name"]), str(row["subject_driver"])))
    return out


def _midfield_team_map(race_df: pd.DataFrame) -> Dict[str, str]:
    dedup = race_df[["Driver", "Team"]].dropna().drop_duplicates()
    return {str(r.Driver): str(r.Team) for r in dedup.itertuples(index=False)}


def _compute_locked_race_scan(base_df: pd.DataFrame) -> Dict[str, Any]:
    zero_stop_rows: List[Dict[str, Any]] = []
    truncated_rows: List[Dict[str, Any]] = []

    power_cache = json.loads(POWER_CACHE_PATH.read_text(encoding="utf-8")) if POWER_CACHE_PATH.exists() else []
    technical_cache = json.loads(TECHNICAL_CACHE_PATH.read_text(encoding="utf-8")) if TECHNICAL_CACHE_PATH.exists() else []
    power_keys = _cache_driver_keys(power_cache)
    technical_keys = _cache_driver_keys(technical_cache)

    for race in sim_config.LOCKED_ARCHETYPE_RACES:
        year = int(race["year"])
        rnd = int(race["round"])
        event_name = str(race["event_name"])
        archetype = str(race["archetype"])

        race_df = base_df[(base_df["Year"] == year) & (base_df["Round"] == rnd)].copy()
        if race_df.empty:
            continue
        race_distance = int(pd.to_numeric(race_df["LapNumber"], errors="coerce").max())
        actual = build_actual_strategies(race_df)
        team_map = _midfield_team_map(race_df)

        for driver, strategy in actual.items():
            team = team_map.get(str(driver), "")
            if not sim_config.is_valid_subject(team):
                continue
            driver_df = race_df[race_df["Driver"].astype(str) == str(driver)]
            max_lap = int(pd.to_numeric(driver_df["LapNumber"], errors="coerce").max())
            lap_fraction = float(max_lap / race_distance) if race_distance else float("nan")
            cache_key = (year, rnd, event_name, str(driver))
            record = {
                "race": {
                    "year": year,
                    "round": rnd,
                    "event_name": event_name,
                    "archetype": archetype,
                },
                "subject_driver": str(driver),
                "subject_team": str(team),
                "starting_compound": str(strategy.get("starting_compound")),
                "stop_count": int(len(strategy.get("stops", []))),
                "max_lap": max_lap,
                "race_distance": race_distance,
                "lap_fraction_of_distance": lap_fraction,
                "shortfall_laps": int(race_distance - max_lap),
                "position_non_null_count": int(driver_df["Position"].notna().sum()) if "Position" in driver_df.columns else None,
                "position_last_non_null_lap": (
                    int(pd.to_numeric(driver_df.loc[driver_df["Position"].notna(), "LapNumber"], errors="coerce").max())
                    if "Position" in driver_df.columns and driver_df["Position"].notna().any()
                    else None
                ),
                "present_in_power_cache": cache_key in power_keys,
                "present_in_technical_cache": cache_key in technical_keys,
            }
            if int(len(strategy.get("stops", []))) == 0:
                zero_stop_rows.append(record)
            if race_distance and lap_fraction < RETIREMENT_THRESHOLD_FRACTION:
                truncated_rows.append(record)

    zero_stop_rows.sort(key=lambda r: (str(r["race"]["archetype"]), int(r["race"]["year"]), int(r["race"]["round"]), str(r["subject_driver"])))
    truncated_rows.sort(key=lambda r: (str(r["race"]["archetype"]), int(r["race"]["year"]), int(r["race"]["round"]), str(r["subject_driver"])))
    return {
        "zero_stop_rows": zero_stop_rows,
        "truncated_rows": truncated_rows,
    }


def main() -> None:
    base_df = _load_base_df()
    scan = _compute_locked_race_scan(base_df)

    print("=" * 100)
    print("Step 9h - DNF / retirement scope check")
    print("=" * 100)
    print(f"base_df_path={BASE_DF_PATH.resolve()}")
    print(f"retirement_threshold_fraction={RETIREMENT_THRESHOLD_FRACTION}")
    print(f"base_df_columns={json.dumps(list(base_df.columns))}")

    print("\nStep 1 - OCO British GP 2023 lap count vs race distance")
    year, rnd, event_name, driver = TARGET_OCO
    race_df = base_df[(base_df["Year"] == year) & (base_df["Round"] == rnd)].copy()
    driver_df = race_df[race_df["Driver"].astype(str) == driver].copy()
    race_distance = int(pd.to_numeric(race_df["LapNumber"], errors="coerce").max())
    oco_max_lap = int(pd.to_numeric(driver_df["LapNumber"], errors="coerce").max())
    print(f"oco_british_2023_max_lap={oco_max_lap}")
    print(f"oco_british_2023_race_distance={race_distance}")
    print(f"oco_british_2023_shortfall_laps={race_distance - oco_max_lap}")
    print(f"oco_british_2023_lap_fraction={oco_max_lap / race_distance:.6f}")
    print(f"oco_british_2023_position_non_null_count={int(driver_df['Position'].notna().sum())}")
    print(
        "oco_british_2023_position_last_non_null_lap="
        + str(int(pd.to_numeric(driver_df.loc[driver_df['Position'].notna(), 'LapNumber'], errors='coerce').max()))
    )

    print("\nStep 2 - Retirement/classification signal availability")
    status_like_columns = [
        col for col in base_df.columns
        if any(token in col.lower() for token in ["status", "retire", "classif", "dnf", "finish"])
    ]
    print("status_like_columns=" + json.dumps(status_like_columns))
    print("position_signal_available=true")
    print("position_signal_note=Processed data retains per-lap Position and truncates each driver to their recorded laps; final classified position in features.py is derived from last non-null Position, not from an explicit finish/DNF field.")

    print("\nStep 3 - Other zero-stop cases: max lap vs race distance")
    print(f"zero_stop_case_count={len(scan['zero_stop_rows'])}")
    print("zero_stop_rows_begin")
    for row in scan["zero_stop_rows"]:
        print("zero_stop_row=" + json.dumps(row, sort_keys=True))
    print("zero_stop_rows_end")

    print("\nStep 4 - Broader truncated-driver scope across all locked-race midfield subjects")
    print(f"truncated_midfield_driver_race_count={len(scan['truncated_rows'])}")
    print("truncated_rows_begin")
    for row in scan["truncated_rows"]:
        print("truncated_row=" + json.dumps(row, sort_keys=True))
    print("truncated_rows_end")
    power_count = sum(1 for row in scan["truncated_rows"] if row["present_in_power_cache"])
    technical_count = sum(1 for row in scan["truncated_rows"] if row["present_in_technical_cache"])
    print(f"truncated_rows_present_in_power_cache={power_count}")
    print(f"truncated_rows_present_in_technical_cache={technical_count}")

    print("\nStep 5 - Recommendation (report only)")
    print("recommended_fix=exclude_retired_or_heavily_truncated_subject_driver_races_from_phase3_subject_roster")
    print("recommended_fix_reason=For a mechanical retirement or early DNF, the observed lap record does not define a meaningful completed-race strategy baseline, so comparing X/B/R on race-finish points is not well-posed.")
    print("alternative_note=If the dissertation specifically wants retirement-counterfactual analysis, that requires a distinct framing and explicit censoring logic, not reuse of the current completed-race strategy extractor.")

    print("\nStep 6 - Completion")
    print("diagnostic_complete=true")


if __name__ == "__main__":
    main()