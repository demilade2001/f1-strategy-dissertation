"""Step 9b diagnostic: verify team mapping and constructor-label merging.

Report-only script. It does not modify source code or datasets.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation import config as sim_config
from source_code.simulation.run import derive_real_subject_roster
from source_code.utils import CONSTRUCTOR_MAPPINGS_BY_YEAR, canonical_constructor_group, get_constructor


BASE_DF_PATH = ROOT / "data" / "processed" / "base_df.csv"


TARGET_RACES = [
    (2022, 22, "Abu Dhabi Grand Prix"),
    (2022, 13, "Hungarian Grand Prix"),
    (2022, 18, "Japanese Grand Prix"),
]


CHANGED_DRIVERS_2022_2024 = [
    {
        "driver": "ALO",
        "actual_change": "Alpine -> Aston Martin (from 2023)",
        "note": "Expected in 2022: Alpine",
    },
    {
        "driver": "GAS",
        "actual_change": "AlphaTauri -> Alpine (from 2023)",
        "note": "Expected in 2022: AlphaTauri",
    },
    {
        "driver": "RIC",
        "actual_change": "McLaren -> AlphaTauri (2023 R12+) -> RB (2024, rebrand)",
        "note": "Handled with round-aware special case in get_constructor()",
    },
    {
        "driver": "LAW",
        "actual_change": "AlphaTauri (2023 cameo) -> RB (2024 R19+)",
        "note": "Handled with round-aware special case in get_constructor()",
    },
    {
        "driver": "DEV",
        "actual_change": "Williams (2022 cameo) -> AlphaTauri (2023)",
        "note": "Not represented as year-specific split in current static 2022 map",
    },
]


def _print_header(title: str) -> None:
    line = "=" * 110
    print(line)
    print(title)
    print(line)


def _load_base_df() -> pd.DataFrame:
    df = pd.read_csv(BASE_DF_PATH)
    required = {"Year", "Round", "EventName", "Driver", "Team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"base_df missing required columns: {sorted(missing)}")
    return df


def _step1_confirm_assignments(df: pd.DataFrame) -> dict:
    _print_header("Step 1 - Confirm Alonso/Gasly 2022 assignment from base_df.csv")
    results: dict = {}

    for year, rnd, event_name in TARGET_RACES:
        race_mask = (df["Year"] == year) & (df["Round"] == rnd)
        race_rows = df.loc[race_mask, ["Driver", "Team", "EventName"]].copy()
        event_names = sorted(set(race_rows["EventName"].dropna().astype(str).tolist()))

        print(f"race={year}-R{rnd} expected_event='{event_name}' observed_events={event_names}")
        race_result = {}
        for drv in ("ALO", "GAS"):
            drv_rows = race_rows[race_rows["Driver"].astype(str).str.upper() == drv]
            unique_teams = sorted(set(drv_rows["Team"].dropna().astype(str).tolist()))
            row_count = int(len(drv_rows))
            race_result[drv] = {
                "row_count": row_count,
                "teams": unique_teams,
            }
            print(
                "raw_team"
                f" year={year} round={rnd} driver={drv} rows={row_count} teams={unique_teams}"
            )

        alo_am = race_result["ALO"]["teams"] == ["Aston Martin"]
        gas_alpine = race_result["GAS"]["teams"] == ["Alpine"]
        plain = "YES" if (alo_am and gas_alpine) else "NO"
        print(
            "plain_check"
            f" race={year}-R{rnd} ALO_is_Aston_Martin={alo_am} GAS_is_Alpine={gas_alpine}"
            f" historical_2022_mismatch_pattern={plain}"
        )
        results[f"{year}_{rnd}"] = race_result

    return results


def _step2_mapping_source() -> dict:
    _print_header("Step 2 - Locate mapping source and print get_constructor() verbatim")
    src = inspect.getsource(get_constructor)
    print(src)

    signature = inspect.signature(get_constructor)
    takes_year = "year" in signature.parameters
    takes_round = "round_number" in signature.parameters
    uses_year_lookup = "CONSTRUCTOR_MAPPINGS_BY_YEAR" in src

    print(f"mapping_signature={signature}")
    print(f"takes_year_input={takes_year}")
    print(f"takes_round_input={takes_round}")
    print(f"uses_year_keyed_lookup_table={uses_year_lookup}")

    year_unaware = not takes_year
    print(f"is_year_unaware_static_lookup={year_unaware}")

    return {
        "source": src,
        "takes_year": takes_year,
        "takes_round": takes_round,
        "uses_year_lookup": uses_year_lookup,
        "is_year_unaware_static_lookup": year_unaware,
    }


def _step3_scope_damage(mapping_info: dict) -> dict:
    _print_header("Step 3 - Scope likely impact across locked races and archetypes")

    locked = sim_config.LOCKED_ARCHETYPE_RACES
    locked_2022 = [r for r in locked if int(r["year"]) == 2022]

    print(f"locked_race_count_total={len(locked)}")
    print(f"locked_race_count_2022={len(locked_2022)}")
    print("locked_2022_races_begin")
    for r in locked_2022:
        print(
            json.dumps(
                {
                    "year": int(r["year"]),
                    "round": int(r["round"]),
                    "event_name": str(r["event_name"]),
                    "archetype": str(r["archetype"]),
                },
                sort_keys=True,
            )
        )
    print("locked_2022_races_end")

    print("known_team_switches_2022_2024_begin")
    for row in CHANGED_DRIVERS_2022_2024:
        print(json.dumps(row, sort_keys=True))
    print("known_team_switches_2022_2024_end")

    year_unaware = bool(mapping_info.get("is_year_unaware_static_lookup", False))
    if year_unaware:
        print("scope_interpretation=If year-unaware, all years in locked races could be systematically mislabeled for switchers.")
    else:
        print("scope_interpretation=Mapping is year-aware; impact depends on incorrect entries inside year-specific map.")
        print("observed_primary_risk=2022 labels for ALO and GAS are inconsistent with history and affect all 2022 locked races.")

    affected_locked_races_for_alo_gas = [
        {
            "year": int(r["year"]),
            "round": int(r["round"]),
            "event_name": str(r["event_name"]),
            "archetype": str(r["archetype"]),
        }
        for r in locked_2022
    ]
    print("affected_locked_races_alo_gas_begin")
    for r in affected_locked_races_for_alo_gas:
        print(json.dumps(r, sort_keys=True))
    print("affected_locked_races_alo_gas_end")

    return {
        "locked_total": len(locked),
        "locked_2022": affected_locked_races_for_alo_gas,
    }


def _step4_merge_checks() -> dict:
    _print_header("Step 4 - Check AlphaTauri/RB and Alfa Romeo/Sauber merge behavior")

    roster_rows = derive_real_subject_roster(sim_config.LOCKED_ARCHETYPE_RACES)

    teams_by_year: dict[int, set[str]] = {}
    for row in roster_rows:
        year = int(row["race"]["year"])
        team_set = teams_by_year.setdefault(year, set())
        for subject in row["subjects"]:
            team_set.add(str(subject["team"]))

    print("roster_teams_by_year_begin")
    for year in sorted(teams_by_year):
        print(f"year={year} teams={sorted(teams_by_year[year])}")
    print("roster_teams_by_year_end")

    map_2022 = CONSTRUCTOR_MAPPINGS_BY_YEAR.get(2022, {})
    map_2023 = CONSTRUCTOR_MAPPINGS_BY_YEAR.get(2023, {})
    map_2024 = CONSTRUCTOR_MAPPINGS_BY_YEAR.get(2024, {})
    print(
        "alfa_sauber_source_logic="
        f"2022(BOT={map_2022.get('BOT')},ZHO={map_2022.get('ZHO')}) "
        f"2023(BOT={map_2023.get('BOT')},ZHO={map_2023.get('ZHO')}) "
        f"2024(BOT={map_2024.get('BOT')},ZHO={map_2024.get('ZHO')})"
    )

    alpha_rb_unified_anywhere = (
        canonical_constructor_group("AlphaTauri")
        == canonical_constructor_group("RB")
    )
    alfa_sauber_unified_anywhere = (
        canonical_constructor_group("Alfa Romeo")
        == canonical_constructor_group("Sauber")
    )

    print(
        "alpha_rb_grouping_logic="
        f"canonical('AlphaTauri')={canonical_constructor_group('AlphaTauri')} "
        f"canonical('RB')={canonical_constructor_group('RB')}"
    )
    print(
        "alfa_sauber_grouping_logic="
        f"canonical('Alfa Romeo')={canonical_constructor_group('Alfa Romeo')} "
        f"canonical('Sauber')={canonical_constructor_group('Sauber')}"
    )
    print(f"alpha_rb_unification_found_at_rollup_grouping={str(alpha_rb_unified_anywhere).lower()}")
    print(f"alfa_sauber_unification_found_at_rollup_grouping={str(alfa_sauber_unified_anywhere).lower()}")

    return {
        "teams_by_year": {k: sorted(v) for k, v in teams_by_year.items()},
        "alpha_rb_unified": alpha_rb_unified_anywhere,
        "alfa_sauber_unified": alfa_sauber_unified_anywhere,
    }


def _step5_recommend_fix_path(step1_results: dict, mapping_info: dict) -> None:
    _print_header("Step 5 - Recommend fix path (no implementation)")

    mismatch_pattern_all_three = True
    for key in ("2022_22", "2022_13", "2022_18"):
        rr = step1_results.get(key, {})
        alo_teams = rr.get("ALO", {}).get("teams", [])
        gas_teams = rr.get("GAS", {}).get("teams", [])
        if not (alo_teams == ["Aston Martin"] and gas_teams == ["Alpine"]):
            mismatch_pattern_all_three = False

    year_unaware = bool(mapping_info.get("is_year_unaware_static_lookup", False))

    if mismatch_pattern_all_three and not year_unaware:
        print("bug_assessment=REAL_MAPPING_BUG_CONFIRMED")
        print("bug_type=year-aware function with incorrect year-specific assignments in 2022 mapping")
        print("recommended_fix_path=FULL_BASE_DF_REBUILD_AFTER_MAPPING_FIX")
        print("reason=Team label is persisted in base_df.csv and is used to derive Step 9 real race roster and team rollups.")
        print("posthoc_cache_relabel_only=NOT_RECOMMENDED for primary correction; can be temporary triage but leaves source dataset inconsistent.")
    elif year_unaware:
        print("bug_assessment=YEAR_UNAWARE_MAPPING_BUG")
        print("recommended_fix_path=FULL_BASE_DF_REBUILD_AFTER_YEAR_AWARE_MAPPING_INTRODUCTION")
    else:
        print("bug_assessment=NO_DIRECT_MAPPING_BUG_CONFIRMED_FROM_REQUESTED_ROWS")
        print("recommended_fix_path=No rebuild required based on checked rows.")


def main() -> None:
    _print_header("Step 9b - Team mapping verification gate before Step 4")
    print(f"base_df_path={BASE_DF_PATH.as_posix()}")
    print(f"locked_race_count={len(sim_config.LOCKED_ARCHETYPE_RACES)}")

    df = _load_base_df()
    step1_results = _step1_confirm_assignments(df)
    mapping_info = _step2_mapping_source()
    _step3_scope_damage(mapping_info)
    _step4_merge_checks()
    _step5_recommend_fix_path(step1_results, mapping_info)


if __name__ == "__main__":
    main()
