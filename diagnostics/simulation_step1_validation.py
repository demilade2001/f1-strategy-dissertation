from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.race_state import load_race_state


BASE_DF_PATH = ROOT / "data" / "processed" / "base_df.csv"


def find_round_for_event(year: int, event_name: str) -> int:
    df = pd.read_csv(BASE_DF_PATH, usecols=["Year", "Round", "EventName"])
    match = df[(df["Year"] == year) & (df["EventName"] == event_name)]["Round"].dropna().unique()
    if len(match) != 1:
        raise ValueError(
            f"Expected exactly one round for {event_name} {year}; got {match.tolist()}"
        )
    return int(match[0])


def print_case(label: str, year: int, round_num: int):
    state = load_race_state(year=year, round_num=round_num)
    laps = state["laps"]
    deg_stats = state["deg_stats"]
    pred_merged = state["predictions_merged"]

    matched = int((pred_merged["_merge"] == "both").sum())
    total = int(len(laps))
    success_rate = matched / total if total else 0.0

    print("=" * 100)
    print(f"{label} | Year={year}, Round={round_num}")
    print("=" * 100)
    print(f"laps_row_count={total}")
    print(f"deg_stats_row_count={len(deg_stats)}")
    print(f"xgb_merge_matched={matched}")
    print(f"xgb_merge_total_laps={total}")
    print(f"xgb_merge_success_rate={success_rate:.2%}")
    print(f"circuit_sc_rate={state['circuit_sc_rate']}")
    print(f"circuit_vsc_rate={state['circuit_vsc_rate']}")
    print("drivers_with_teams:")
    for row in state["drivers_with_teams"]:
        print(f"  {row['Driver']}: {row['Team']}")

    if success_rate < 1.0:
        missing = pred_merged[pred_merged["_merge"] != "both"][
            ["Year", "Round", "Driver", "LapNumber"]
        ].drop_duplicates()
        print("unmatched_prediction_keys:")
        for _, row in missing.iterrows():
            print(
                f"  Year={int(row['Year'])}, Round={int(row['Round'])}, "
                f"Driver={row['Driver']}, LapNumber={int(row['LapNumber'])}"
            )


def main():
    monaco_round = find_round_for_event(2023, "Monaco Grand Prix")
    monza_round = find_round_for_event(2023, "Italian Grand Prix")
    hungary_round = find_round_for_event(2023, "Hungarian Grand Prix")

    print_case("Monaco 2023 (Street)", 2023, monaco_round)
    print_case("Monza 2023 (Power)", 2023, monza_round)
    print_case("Hungary 2023 (Technical)", 2023, hungary_round)


if __name__ == "__main__":
    main()