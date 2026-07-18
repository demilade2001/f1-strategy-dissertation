from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.race_state import load_race_state
from src.simulation.config import is_valid_subject
from src.simulation.strategy import enumerate_feasible_strategies


WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
RACES = [
    ("Monaco 2024", 2024, 8),
    ("Monza 2024", 2024, 16),
    ("Hungary 2024", 2024, 13),
]


def pick_reference_driver(laps: pd.DataFrame) -> tuple[str, str]:
    final_rows = (
        laps.sort_values(["Driver", "LapNumber"])
        .groupby("Driver", as_index=False)
        .tail(1)
        .copy()
    )
    final_rows = final_rows[final_rows["Team"].apply(is_valid_subject)].copy()
    if final_rows.empty:
        raise ValueError("No midfield driver found for validation race")
    final_rows = final_rows.sort_values(["Position", "LapNumber", "Driver"])
    row = final_rows.iloc[0]
    return str(row["Driver"]), str(row["Team"])


def determine_wet_race(race_laps: pd.DataFrame):
    wet_cols = [c for c in race_laps.columns if "wet" in c.lower() and "compound" not in c.lower()]
    if wet_cols:
        col = wet_cols[0]
        non_null = race_laps[col].dropna()
        value = non_null.iloc[0] if not non_null.empty else None
        return col, value, bool(value)

    wet_seen = race_laps["TyreCompound"].isin(WET_COMPOUNDS).any()
    return None, None, bool(wet_seen)


def extract_driver_laps(race_laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    sub = race_laps[race_laps["Driver"] == driver].copy()
    if sub.empty:
        raise ValueError(f"No laps found for driver {driver}")
    return sub.sort_values("LapNumber").reset_index(drop=True)


def extract_actual_strategy(driver_laps: pd.DataFrame):
    driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)
    lap1 = driver_laps[driver_laps["LapNumber"] == 1]
    if lap1.empty:
        raise ValueError("Driver has no LapNumber == 1 row; cannot determine starting compound")
    starting_compound = str(lap1.iloc[0]["TyreCompound"])

    stops = []
    prev_compound = None
    for _, row in driver_laps.iterrows():
        compound = row["TyreCompound"]
        if pd.isna(compound):
            continue
        compound = str(compound)
        if prev_compound is None:
            prev_compound = compound
            continue
        if compound != prev_compound:
            stops.append((int(row["LapNumber"]), compound))
        prev_compound = compound

    return {
        "starting_compound": starting_compound,
        "stops": stops,
    }


def print_driver_race_summary(label: str, race_laps: pd.DataFrame, driver: str, team: str):
    driver_laps = extract_driver_laps(race_laps, driver)
    race_length = int(driver_laps["LapNumber"].max())
    starting_compound = str(driver_laps.loc[driver_laps["LapNumber"] == 1, "TyreCompound"].iloc[0])
    wet_col, wet_val, is_wet_race = determine_wet_race(race_laps)

    print(f"selected_driver={driver}")
    print(f"selected_driver_team={team}")
    print(f"selected_driver_is_valid_subject={is_valid_subject(team)}")
    print(f"driver_race_length={race_length}")
    print(f"starting_compound={starting_compound}")
    if wet_col is not None:
        print(f"wet_indicator_column={wet_col}")
        print(f"wet_indicator_value={wet_val}")
    else:
        print("wet_indicator_column=None")
        print("wet_indicator_value=None")
        print("wet_encoding_note=No explicit wet_race_flag column exists in base_df.csv; wet conditions are encoded via TyreCompound containing INTERMEDIATE/WET.")
    print(f"is_wet_race={is_wet_race}")

    actual_strategy = extract_actual_strategy(driver_laps)
    print(f"actual_strategy={actual_strategy['stops']}")
    return driver_laps, actual_strategy, race_length, starting_compound, is_wet_race


def main():
    size_rows = []
    feasibility_rows = []

    for label, year, round_num in RACES:
        state = load_race_state(year=year, round_num=round_num)
        race_laps = state["laps"]
        driver, team = pick_reference_driver(race_laps)

        print("=" * 100)
        print(f"{label} | Year={year}, Round={round_num}")
        print("=" * 100)

        driver_laps, actual_strategy, race_length, starting_compound, is_wet_race = print_driver_race_summary(
            label, race_laps, driver, team
        )

        for max_stops in [2, 3, 4]:
            feasible = enumerate_feasible_strategies(
                race_length=race_length,
                starting_compound=starting_compound,
                is_wet_race=is_wet_race,
                max_stops=max_stops,
            )
            size_rows.append(
                {
                    "race": label,
                    "driver": driver,
                    "team": team,
                    "max_stops": max_stops,
                    "feasible_set_size": len(feasible),
                }
            )

        actual_stop_count = len(actual_strategy["stops"])
        feasible_for_actual = enumerate_feasible_strategies(
            race_length=race_length,
            starting_compound=starting_compound,
            is_wet_race=is_wet_race,
            max_stops=max(actual_stop_count, 1),
        )
        is_feasible = actual_strategy in feasible_for_actual

        violation = None
        if not is_feasible:
            if actual_stop_count < 1:
                violation = "stop count below dry-race minimum"
            elif actual_stop_count > max(actual_stop_count, 1):
                violation = "stop count exceeds max_stops"
            elif any(lap < 1 or lap > race_length - 1 for lap, _ in actual_strategy["stops"]):
                violation = "pit lap outside [1, race_length - 1]"
            elif any(
                actual_strategy["stops"][idx][0] >= actual_strategy["stops"][idx + 1][0]
                for idx in range(len(actual_strategy["stops"]) - 1)
            ):
                violation = "pit laps not strictly increasing"
            else:
                violation = "compound rule"

        feasibility_rows.append(
            {
                "race": label,
                "driver": driver,
                "team": team,
                "actual_stop_count": actual_stop_count,
                "checked_with_max_stops": max(actual_stop_count, 1),
                "actual_strategy_feasible": is_feasible,
                "violation": violation,
                "actual_strategy": actual_strategy["stops"],
            }
        )

    print("\nStep 4 - Feasible set sizes")
    size_df = pd.DataFrame(size_rows)
    print(size_df.to_string(index=False))

    print("\nStep 5 - Actual strategy feasibility")
    feas_df = pd.DataFrame(feasibility_rows)
    print(feas_df.to_string(index=False))


if __name__ == "__main__":
    main()