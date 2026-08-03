from math import comb
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation.config import BASE_DF_PATH, is_valid_subject
from source_code.simulation.race_state import load_race_state


VALIDATION_RACES = [
    ("Monaco 2024", 2024, 8),
    ("Monza 2024", 2024, 16),
    ("Hungary 2024", 2024, 13),
]
GRID_SPACINGS = [1, 2, 3, 4]
MAX_STOPS_GRID = [2, 3, 4]

TAIL_CASES = [
    {"label": "Spanish 2022 ALB", "year": 2022, "round": 6, "driver": "ALB", "event": "Spanish Grand Prix", "stop_count": 4},
    {"label": "Monaco 2022 TSU", "year": 2022, "round": 7, "driver": "TSU", "event": "Monaco Grand Prix", "stop_count": 4},
    {"label": "Bahrain 2023 NOR", "year": 2023, "round": 1, "driver": "NOR", "event": "Bahrain Grand Prix", "stop_count": 5},
    {"label": "British 2024 OCO", "year": 2024, "round": 12, "driver": "OCO", "event": "British Grand Prix", "stop_count": 4},
]

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"})


def detect_stop_events(driver_laps: pd.DataFrame) -> List[Tuple[int, str]]:
    events: List[Tuple[int, str]] = []
    prev_compound: Optional[str] = None
    for _, row in driver_laps.sort_values("LapNumber").iterrows():
        compound = row["TyreCompound"]
        if pd.isna(compound):
            continue
        compound = str(compound)
        if prev_compound is None:
            prev_compound = compound
            continue
        if compound != prev_compound:
            events.append((int(row["LapNumber"]), compound))
        prev_compound = compound
    return events


def pick_reference_driver(laps: pd.DataFrame) -> Tuple[str, str]:
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


def determine_wet_race(race_laps: pd.DataFrame) -> bool:
    wet_cols = [c for c in race_laps.columns if "wet" in c.lower() and "compound" not in c.lower()]
    if wet_cols:
        col = wet_cols[0]
        non_null = race_laps[col].dropna()
        if not non_null.empty:
            return bool(non_null.iloc[0])
    return bool(race_laps["TyreCompound"].isin(WET_COMPOUNDS).any())


def candidate_pit_laps(race_length: int, spacing: int) -> List[int]:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    upper = race_length - 1
    return list(range(1, upper + 1, spacing))


def compound_sequence_count(starting_compound: str, stop_count: int, is_wet_race: bool) -> int:
    base = len(DRY_COMPOUNDS) ** stop_count
    if is_wet_race:
        return base
    if starting_compound in set(DRY_COMPOUNDS):
        return base - 1
    return base


def feasible_size_with_grid(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    spacing: int,
    max_stops: int,
) -> int:
    pit_laps = candidate_pit_laps(race_length, spacing)
    n = len(pit_laps)
    total = 0
    for stop_count in range(1, max_stops + 1):
        if stop_count > n:
            continue
        total += comb(n, stop_count) * compound_sequence_count(starting_compound, stop_count, is_wet_race)
    return total


def feasible_size_exact_stop_count_with_grid(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    spacing: int,
    stop_count: int,
) -> int:
    pit_laps = candidate_pit_laps(race_length, spacing)
    n = len(pit_laps)
    if stop_count > n:
        return 0
    return comb(n, stop_count) * compound_sequence_count(starting_compound, stop_count, is_wet_race)


def snapping_errors(actual_pit_laps: List[int], candidates: List[int]) -> Tuple[float, float]:
    if not actual_pit_laps:
        return 0.0, 0.0
    distances = []
    for lap in actual_pit_laps:
        d = min(abs(lap - c) for c in candidates)
        distances.append(float(d))
    mean_err = sum(distances) / len(distances)
    max_err = max(distances)
    return mean_err, max_err


def load_midfield_base_df() -> pd.DataFrame:
    base_df = pd.read_csv(BASE_DF_PATH)
    base_df = base_df[base_df["Year"].isin([2022, 2023, 2024])].copy()
    base_df = base_df[to_bool(base_df["is_midfield"])].copy()
    return base_df.sort_values(["Year", "Round", "Driver", "LapNumber"]).reset_index(drop=True)


def main():
    size_rows: List[Dict] = []
    snap_rows: List[Dict] = []
    selected_rows: List[Dict] = []

    validation_context: Dict[str, Dict] = {}

    for label, year, round_num in VALIDATION_RACES:
        state = load_race_state(year=year, round_num=round_num)
        race_laps = state["laps"]
        driver, team = pick_reference_driver(race_laps)
        driver_laps = race_laps[race_laps["Driver"] == driver].copy().sort_values("LapNumber")
        race_length = int(driver_laps["LapNumber"].max())
        strategy = detect_stop_events(driver_laps)
        actual_pit_laps = [lap for lap, _ in strategy]
        starting_compound = str(driver_laps.loc[driver_laps["LapNumber"] == 1, "TyreCompound"].iloc[0])
        is_wet_race = determine_wet_race(race_laps)

        validation_context[label] = {
            "race_length": race_length,
            "starting_compound": starting_compound,
            "is_wet_race": is_wet_race,
            "actual_pit_laps": actual_pit_laps,
            "actual_stop_count": len(actual_pit_laps),
            "driver": driver,
            "team": team,
        }
        selected_rows.append({"race": label, "driver": driver, "team": team})

        for spacing in GRID_SPACINGS:
            candidates = candidate_pit_laps(race_length, spacing)
            mean_err, max_err = snapping_errors(actual_pit_laps, candidates)
            snap_rows.append(
                {
                    "race": label,
                    "driver": driver,
                    "team": team,
                    "grid_spacing": spacing,
                    "actual_stop_count": len(actual_pit_laps),
                    "mean_snapping_error_laps": round(mean_err, 3),
                    "max_snapping_error_laps": round(max_err, 3),
                }
            )
            for max_stops in MAX_STOPS_GRID:
                size_rows.append(
                    {
                        "race": label,
                        "driver": driver,
                        "team": team,
                        "grid_spacing": spacing,
                        "max_stops": max_stops,
                        "feasible_set_size": feasible_size_with_grid(
                            race_length=race_length,
                            starting_compound=starting_compound,
                            is_wet_race=is_wet_race,
                            spacing=spacing,
                            max_stops=max_stops,
                        ),
                    }
                )

    print("Step 1 - Grid-coarsened feasible-set sizes")
    size_df = pd.DataFrame(size_rows)
    print(size_df.to_string(index=False))

    print("\nStep 2 - Snapping-error fidelity check")
    snap_df = pd.DataFrame(snap_rows)
    print(snap_df.to_string(index=False))

    print("\nStep 3 - Tail check on known 4+ stop midfield driver-races")
    print("provisional_tractability_bar=50000")
    base_df = load_midfield_base_df()
    tail_rows: List[Dict] = []
    below_bar_rows: List[Dict] = []

    for case in TAIL_CASES:
        sub = base_df[
            (base_df["Year"] == case["year"])
            & (base_df["Round"] == case["round"])
            & (base_df["Driver"] == case["driver"])
        ].copy()
        if sub.empty:
            raise RuntimeError(f"Tail case not found in base_df: {case['label']}")

        race_length = int(sub["LapNumber"].max())
        starting_compound = str(sub.loc[sub["LapNumber"] == 1, "TyreCompound"].iloc[0])
        is_wet_race = bool(sub["TyreCompound"].isin(WET_COMPOUNDS).any())
        actual_stop_count = int(case["stop_count"])

        for spacing in [2, 3, 4]:
            size = feasible_size_exact_stop_count_with_grid(
                race_length=race_length,
                starting_compound=starting_compound,
                is_wet_race=is_wet_race,
                spacing=spacing,
                stop_count=actual_stop_count,
            )
            row = {
                "case": case["label"],
                "year": case["year"],
                "round": case["round"],
                "driver": case["driver"],
                "actual_stop_count": actual_stop_count,
                "grid_spacing": spacing,
                "feasible_set_size_exact_stop_count": size,
                "below_50000_provisional_bar": size < 50_000,
            }
            tail_rows.append(row)
            if size < 50_000:
                below_bar_rows.append(row)

    tail_df = pd.DataFrame(tail_rows)
    print(tail_df.to_string(index=False))
    if below_bar_rows:
        print("tail_below_bar_any=True")
        print(pd.DataFrame(below_bar_rows).to_string(index=False))
    else:
        print("tail_below_bar_any=False")

    print("\nStep 4 - Recommendation (not a decision)")
    # Criterion A: 2-stop race sizes in low-thousands to tens-of-thousands range.
    # Use races whose actual historical strategy has 2 stops.
    two_stop_races = [r for r, c in validation_context.items() if c["actual_stop_count"] == 2]
    spacing_summary: List[Dict] = []

    for spacing in GRID_SPACINGS:
        two_stop_sizes = []
        for race in two_stop_races:
            ctx = validation_context[race]
            size = feasible_size_with_grid(
                race_length=ctx["race_length"],
                starting_compound=ctx["starting_compound"],
                is_wet_race=ctx["is_wet_race"],
                spacing=spacing,
                max_stops=2,
            )
            two_stop_sizes.append(size)

        if two_stop_sizes:
            low_thousands_tens_thousands_ok = all(1_000 <= s < 100_000 for s in two_stop_sizes)
            two_stop_size_min = min(two_stop_sizes)
            two_stop_size_max = max(two_stop_sizes)
        else:
            low_thousands_tens_thousands_ok = False
            two_stop_size_min = None
            two_stop_size_max = None

        spacing_snap = snap_df[snap_df["grid_spacing"] == spacing]
        mean_snap_ok = bool((spacing_snap["mean_snapping_error_laps"] < 1.5).all())
        max_mean_snap = float(spacing_snap["mean_snapping_error_laps"].max())

        tail_spacing = tail_df[tail_df["grid_spacing"] == spacing]
        tail_below = bool((tail_spacing["below_50000_provisional_bar"] == True).any())

        spacing_summary.append(
            {
                "grid_spacing": spacing,
                "two_stop_low_thousands_to_tens_thousands_ok": low_thousands_tens_thousands_ok,
                "two_stop_size_min": two_stop_size_min,
                "two_stop_size_max": two_stop_size_max,
                "mean_snapping_error_all_races_lt_1p5": mean_snap_ok,
                "max_mean_snapping_error": round(max_mean_snap, 3),
                "tail_any_case_below_50000_provisional_bar": tail_below,
            }
        )

    summary_df = pd.DataFrame(spacing_summary)
    print(summary_df.to_string(index=False))

    candidates_two_stop = summary_df[
        (summary_df["two_stop_low_thousands_to_tens_thousands_ok"] == True)
        & (summary_df["mean_snapping_error_all_races_lt_1p5"] == True)
    ].copy()

    candidates_both = summary_df[
        (summary_df["two_stop_low_thousands_to_tens_thousands_ok"] == True)
        & (summary_df["mean_snapping_error_all_races_lt_1p5"] == True)
        & (summary_df["tail_any_case_below_50000_provisional_bar"] == True)
    ].copy()

    if candidates_two_stop.empty:
        print(
            "two_stop_screen=No spacing satisfies both 2-stop size target and mean snapping error <1.5 laps"
        )
    else:
        spacings = ", ".join(str(int(x)) for x in sorted(candidates_two_stop["grid_spacing"].tolist()))
        print(
            "two_stop_screen=Spacings passing 2-stop size target and mean snapping error <1.5 laps: "
            + spacings
        )

    if candidates_both.empty:
        print(
            "recommendation=No single grid spacing satisfies both the 2-stop condition "
            "(low-thousands to tens-of-thousands with mean snapping error <1.5 laps) "
            "and acceptable 3-4 stop tail tractability under the provisional <50,000 bar."
        )
    else:
        chosen = candidates_both.sort_values("grid_spacing").iloc[0]
        print(
            "recommendation=Grid spacing "
            f"{int(chosen['grid_spacing'])} satisfies both 2-stop and provisional tail conditions."
        )

    if bool((summary_df["tail_any_case_below_50000_provisional_bar"] == False).all()):
        print(
            "tail_conclusion=Grid coarsening alone does not push any of the known 4+-stop tail cases "
            "below the provisional 50,000 bar; an additional restriction layer may be required."
        )
    else:
        print(
            "tail_conclusion=At least one 4+-stop tail case falls below the provisional 50,000 bar under "
            "grid coarsening alone."
        )


if __name__ == "__main__":
    main()