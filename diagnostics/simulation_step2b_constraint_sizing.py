from collections import Counter
from math import comb
from pathlib import Path
import sys
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.config import BASE_DF_PATH, is_valid_subject
from src.simulation.race_state import load_race_state
from src.simulation.strategy import enumerate_feasible_strategies, is_strategy_feasible


RACES = [
    ("Monaco 2024", 2024, 8),
    ("Monza 2024", 2024, 16),
    ("Hungary 2024", 2024, 13),
]
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"})


def detect_stop_laps(driver_laps: pd.DataFrame) -> list[int]:
    stop_laps = []
    prev_compound = None
    for _, row in driver_laps.sort_values("LapNumber").iterrows():
        compound = row["TyreCompound"]
        if pd.isna(compound):
            continue
        compound = str(compound)
        if prev_compound is None:
            prev_compound = compound
            continue
        if compound != prev_compound:
            stop_laps.append(int(row["LapNumber"]))
        prev_compound = compound
    return stop_laps


def compute_stint_lengths(race_length: int, stop_laps: list[int]) -> list[int]:
    boundaries = [1] + stop_laps + [race_length + 1]
    stint_lengths = []
    for idx in range(len(boundaries) - 1):
        start_lap = boundaries[idx]
        next_boundary = boundaries[idx + 1]
        stint_lengths.append(int(next_boundary - start_lap))
    return stint_lengths


def determine_wet_race(race_laps: pd.DataFrame) -> bool:
    wet_cols = [c for c in race_laps.columns if "wet" in c.lower() and "compound" not in c.lower()]
    if wet_cols:
        col = wet_cols[0]
        non_null = race_laps[col].dropna()
        if not non_null.empty:
            return bool(non_null.iloc[0])
    return bool(race_laps["TyreCompound"].isin(WET_COMPOUNDS).any())


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


def extract_driver_laps(race_laps: pd.DataFrame, driver: str) -> pd.DataFrame:
    sub = race_laps[race_laps["Driver"] == driver].copy()
    if sub.empty:
        raise ValueError(f"No laps found for driver {driver}")
    return sub.sort_values("LapNumber").reset_index(drop=True)


def extract_actual_strategy(driver_laps: pd.DataFrame) -> dict:
    lap1 = driver_laps[driver_laps["LapNumber"] == 1]
    if lap1.empty:
        raise ValueError("Driver has no LapNumber == 1 row; cannot determine starting compound")
    starting_compound = str(lap1.iloc[0]["TyreCompound"])
    return {
        "starting_compound": starting_compound,
        "stops": [(lap, str(compound)) for lap, compound in zip(detect_stop_laps(driver_laps), [
            driver_laps.loc[driver_laps["LapNumber"] == lap, "TyreCompound"].iloc[0]
            for lap in detect_stop_laps(driver_laps)
        ])],
    }


def compound_sequence_count(starting_compound: str, stop_count: int, is_wet_race: bool) -> int:
    base = len(DRY_COMPOUNDS) ** stop_count
    if is_wet_race:
        return base
    if starting_compound in set(DRY_COMPOUNDS):
        return base - 1
    return base


def constrained_pit_lap_choice_count(race_length: int, stop_count: int, min_stint_length: int) -> int:
    final_stint_min = max(int(min_stint_length), 2)
    remaining = race_length - stop_count * int(min_stint_length) - final_stint_min
    if remaining < 0:
        return 0
    return comb(remaining + stop_count, stop_count)


def constrained_feasible_set_size(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    max_stops: int,
    min_stint_length: int,
) -> int:
    total = 0
    for stop_count in range(1, max_stops + 1):
        total += (
            constrained_pit_lap_choice_count(race_length, stop_count, min_stint_length)
            * compound_sequence_count(starting_compound, stop_count, is_wet_race)
        )
    return total


def strategy_stint_lengths(race_length: int, stops: list[tuple[int, str]]) -> list[int]:
    stop_laps = [int(lap) for lap, _ in stops]
    return compute_stint_lengths(race_length, stop_laps)


def is_strategy_feasible_with_min_stint(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    stops: list[tuple[int, str]],
    max_stops: int,
    min_stint_length: int,
) -> tuple[bool, Optional[str]]:
    base_ok = is_strategy_feasible(
        race_length=race_length,
        starting_compound=starting_compound,
        is_wet_race=is_wet_race,
        stops=stops,
        dry_compounds=DRY_COMPOUNDS,
        max_stops=max_stops,
    )
    if not base_ok:
        stop_count = len(stops)
        if stop_count > max_stops:
            return False, f"stop_count={stop_count} exceeds max_stops={max_stops} by {stop_count - max_stops}"
        return False, "violates existing strategy.py feasibility rules"

    stint_lengths = strategy_stint_lengths(race_length, stops)
    shortest = min(stint_lengths)
    if shortest < min_stint_length:
        return False, (
            f"shortest_stint={shortest} below min_stint_length={min_stint_length} by "
            f"{min_stint_length - shortest}"
        )
    return True, None


def print_stop_count_distribution(driver_race_df: pd.DataFrame) -> tuple[int, float, float]:
    exact_counts = driver_race_df["stop_count"].value_counts().sort_index()
    bucket_counts = Counter()
    for stop_count, count in exact_counts.items():
        bucket = int(stop_count) if int(stop_count) < 5 else "5+"
        bucket_counts[bucket] += int(count)

    total = int(len(driver_race_df))
    print("Step 1 - Stop-count distribution for midfield driver-races")
    for bucket in [0, 1, 2, 3, 4, "5+"]:
        print(f"stop_count_{bucket}={bucket_counts.get(bucket, 0)}")

    stop_quantiles = driver_race_df["stop_count"].quantile([0.5, 0.9, 0.95, 0.99], interpolation="nearest")
    print(f"stop_count_p50_nearest={int(stop_quantiles.loc[0.5])}")
    print(f"stop_count_p90_nearest={int(stop_quantiles.loc[0.9])}")
    print(f"stop_count_p95_nearest={int(stop_quantiles.loc[0.95])}")
    print(f"stop_count_p99_nearest={int(stop_quantiles.loc[0.99])}")

    three_or_fewer = int((driver_race_df["stop_count"] <= 3).sum())
    four_or_more = int((driver_race_df["stop_count"] >= 4).sum())
    proportion_three_or_fewer = three_or_fewer / total
    proportion_four_or_more = four_or_more / total
    print(f"driver_races_total={total}")
    print(f"driver_races_3_or_fewer={three_or_fewer}")
    print(f"driver_races_4_or_more={four_or_more}")
    print(f"proportion_3_or_fewer={proportion_three_or_fewer:.6f}")
    print(f"proportion_4_or_more={proportion_four_or_more:.6f}")
    return int(stop_quantiles.loc[0.99]), proportion_three_or_fewer, proportion_four_or_more


def print_stint_length_distribution(stint_lengths: pd.Series) -> tuple[int, int, int, int, int, int]:
    quantiles = stint_lengths.quantile([0.01, 0.05, 0.10, 0.5], interpolation="nearest")
    counts = stint_lengths.value_counts().sort_index()
    repeated_min = next(int(length) for length, count in counts.items() if int(count) > 10)

    print("\nStep 2 - Stint-length distribution for midfield stints")
    print(f"stint_count_total={int(len(stint_lengths))}")
    print(f"stint_length_min={int(stint_lengths.min())}")
    print(f"stint_length_p01_nearest={int(quantiles.loc[0.01])}")
    print(f"stint_length_p05_nearest={int(quantiles.loc[0.05])}")
    print(f"stint_length_p10_nearest={int(quantiles.loc[0.10])}")
    print(f"stint_length_median_nearest={int(quantiles.loc[0.5])}")
    print(f"shortest_stint_with_count_gt_10={repeated_min}")
    return (
        int(stint_lengths.min()),
        int(quantiles.loc[0.01]),
        int(quantiles.loc[0.05]),
        int(quantiles.loc[0.10]),
        int(quantiles.loc[0.5]),
        repeated_min,
    )


def recommend_constraints(stop_count_p99: int, repeated_min_stint: int) -> tuple[int, int]:
    recommended_max_stops = int(stop_count_p99)
    recommended_min_stint = int(repeated_min_stint)
    print("\nStep 3 - Recommended constraints from historical distributions")
    print(
        "recommended_max_stops="
        f"{recommended_max_stops} (set to the nearest-observed 99th percentile of midfield stop counts)"
    )
    print(
        "recommended_min_stint_length="
        f"{recommended_min_stint} (set to the shortest stint length that appears more than 10 times)"
    )
    return recommended_max_stops, recommended_min_stint


def build_driver_race_history(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    records = []
    stint_lengths = []
    keys = ["Year", "Round", "EventName", "Driver", "Team"]
    for key, driver_laps in base_df.sort_values(keys + ["LapNumber"]).groupby(keys, sort=False):
        year, round_num, event_name, driver, team = key
        race_length = int(driver_laps["LapNumber"].max())
        stop_laps = detect_stop_laps(driver_laps)
        stints = compute_stint_lengths(race_length, stop_laps)
        records.append(
            {
                "Year": int(year),
                "Round": int(round_num),
                "EventName": str(event_name),
                "Driver": str(driver),
                "Team": str(team),
                "race_length": race_length,
                "stop_count": int(len(stop_laps)),
                "stop_laps": stop_laps,
                "stint_lengths": stints,
            }
        )
        stint_lengths.extend(stints)
    return pd.DataFrame.from_records(records), pd.Series(stint_lengths, dtype="int64")


def load_midfield_base_df() -> pd.DataFrame:
    base_df = pd.read_csv(BASE_DF_PATH)
    base_df = base_df[base_df["Year"].isin([2022, 2023, 2024])].copy()
    base_df = base_df[to_bool(base_df["is_midfield"])].copy()
    base_df = base_df.sort_values(["Year", "Round", "Driver", "LapNumber"]).reset_index(drop=True)
    return base_df


def analyze_selected_races(recommended_max_stops: int, recommended_min_stint: int):
    size_rows = []
    feasibility_rows = []
    selected_rows = []

    for label, year, round_num in RACES:
        state = load_race_state(year=year, round_num=round_num)
        race_laps = state["laps"]
        driver, team = pick_reference_driver(race_laps)
        driver_laps = extract_driver_laps(race_laps, driver)
        race_length = int(driver_laps["LapNumber"].max())
        starting_compound = str(driver_laps.loc[driver_laps["LapNumber"] == 1, "TyreCompound"].iloc[0])
        is_wet_race = determine_wet_race(race_laps)
        actual_strategy = extract_actual_strategy(driver_laps)
        actual_stint_lengths = strategy_stint_lengths(race_length, actual_strategy["stops"])

        selected_rows.append({"race": label, "driver": driver, "team": team})

        unconstrained = enumerate_feasible_strategies(
            race_length=race_length,
            starting_compound=starting_compound,
            is_wet_race=is_wet_race,
            max_stops=recommended_max_stops,
        )
        constrained_size = constrained_feasible_set_size(
            race_length=race_length,
            starting_compound=starting_compound,
            is_wet_race=is_wet_race,
            max_stops=recommended_max_stops,
            min_stint_length=recommended_min_stint,
        )
        reduction_factor = (len(unconstrained) / constrained_size) if constrained_size else float("inf")

        size_rows.append(
            {
                "race": label,
                "driver": driver,
                "team": team,
                "max_stops": recommended_max_stops,
                "min_stint_length": recommended_min_stint,
                "unconstrained_feasible_set_size": len(unconstrained),
                "feasible_set_size": constrained_size,
                "reduction_factor": round(reduction_factor, 3) if constrained_size else "inf",
            }
        )

        is_feasible, violation = is_strategy_feasible_with_min_stint(
            race_length=race_length,
            starting_compound=starting_compound,
            is_wet_race=is_wet_race,
            stops=actual_strategy["stops"],
            max_stops=recommended_max_stops,
            min_stint_length=recommended_min_stint,
        )
        feasibility_rows.append(
            {
                "race": label,
                "driver": driver,
                "team": team,
                "actual_stop_count": len(actual_strategy["stops"]),
                "actual_stint_lengths": actual_stint_lengths,
                "actual_strategy_feasible": is_feasible,
                "violation": violation,
                "actual_strategy": actual_strategy["stops"],
            }
        )

    selected_df = pd.DataFrame(selected_rows)
    repeated = selected_df["driver"].value_counts()
    repeated = repeated[repeated > 1]

    print("\nStep 4 - Selected midfield subjects for constrained sizing")
    print(selected_df.to_string(index=False))
    if repeated.empty:
        print("duplicate_selected_driver_flag=None")
    else:
        repeated_blob = ", ".join(f"{driver} x{count}" for driver, count in repeated.items())
        print(f"duplicate_selected_driver_flag={repeated_blob}")

    print("\nStep 4 - Constrained feasible set sizes")
    print(pd.DataFrame(size_rows).to_string(index=False))

    print("\nStep 5 - Actual strategy feasibility under proposed constraints")
    print(pd.DataFrame(feasibility_rows).to_string(index=False))


def main():
    base_df = load_midfield_base_df()
    driver_race_df, stint_lengths = build_driver_race_history(base_df)
    stop_count_p99, _, _ = print_stop_count_distribution(driver_race_df)
    _, _, _, _, _, repeated_min_stint = print_stint_length_distribution(stint_lengths)
    recommended_max_stops, recommended_min_stint = recommend_constraints(
        stop_count_p99=stop_count_p99,
        repeated_min_stint=repeated_min_stint,
    )
    analyze_selected_races(
        recommended_max_stops=recommended_max_stops,
        recommended_min_stint=recommended_min_stint,
    )


if __name__ == "__main__":
    main()