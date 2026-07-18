from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.race_state import load_race_state


BASE_DF_PATH = ROOT / "data" / "processed" / "base_df.csv"
PRED_PATH = ROOT / "data" / "models" / "classweight_test_predictions.csv"


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

        pred_subset = pd.read_csv(PRED_PATH)
        pred_subset = pred_subset[(pred_subset["Year"] == year) & (pred_subset["Round"] == round_num)]
        key_cols = ["Year", "Round", "Driver", "LapNumber"]
        print("merge_key_dtypes_laps:")
        for col in key_cols:
            print(f"  {col}: {laps[col].dtype}")
        print("merge_key_dtypes_predictions:")
        for col in key_cols:
            print(f"  {col}: {pred_subset[col].dtype}")


def main():
    print_case("Monaco 2024 (Street)", 2024, 8)
    print_case("Monza 2024 (Power)", 2024, 16)
    print_case("Hungary 2024 (Technical)", 2024, 13)


if __name__ == "__main__":
    main()