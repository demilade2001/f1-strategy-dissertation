from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.config import RNG_SEED


DEG_STATS_PATH = ROOT / "data" / "diagnostics" / "deg_rate_full_peryear_stats.csv"
BASE_DF_PATH = ROOT / "data" / "processed" / "base_df.csv"

TARGET_CIRCUITS = [
    {"archetype": "Street", "event_name": "Monaco Grand Prix"},
    {"archetype": "Street", "event_name": "Azerbaijan Grand Prix"},
    {"archetype": "Street", "event_name": "Singapore Grand Prix"},
    {"archetype": "Street", "event_name": "Miami Grand Prix"},
    {"archetype": "Power", "event_name": "Italian Grand Prix"},
    {"archetype": "Power", "event_name": "Belgian Grand Prix"},
    {"archetype": "Power", "event_name": "British Grand Prix"},
    {"archetype": "Power", "event_name": "Bahrain Grand Prix"},
    {"archetype": "Technical", "event_name": "Spanish Grand Prix"},
    {"archetype": "Technical", "event_name": "Hungarian Grand Prix"},
    {"archetype": "Technical", "event_name": "Abu Dhabi Grand Prix"},
    {"archetype": "Technical", "event_name": "Japanese Grand Prix"},
]
YEARS = [2022, 2023, 2024]
TARGET_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def main():
    deg = pd.read_csv(DEG_STATS_PATH)
    base_df = pd.read_csv(BASE_DF_PATH)

    # Step 1 - compile exclusions using ONLY low_sample counts from deg stats.
    rows = []
    for circuit in TARGET_CIRCUITS:
        event_name = circuit["event_name"]
        for year in YEARS:
            sub = deg[
                (deg["EventName"] == event_name)
                & (deg["Year"] == year)
                & (deg["TyreCompound"].isin(TARGET_COMPOUNDS))
            ].copy()
            low_sample_count = int((sub["low_sample"] == True).sum())
            excluded = low_sample_count >= 2
            rows.append(
                {
                    "circuit": event_name,
                    "year": int(year),
                    "compounds_low_sample_count": low_sample_count,
                    "excluded": bool(excluded),
                }
            )

    exclusion_df = pd.DataFrame(rows).sort_values(["circuit", "year"]).reset_index(drop=True)
    print("Step 1 - Exclusion list from low_sample flags only")
    print(exclusion_df.to_string(index=False))

    # Step 2 - eligible years per circuit.
    eligible_map = {}
    print("\nStep 2 - Eligible years per circuit")
    for event_name in sorted([x["event_name"] for x in TARGET_CIRCUITS]):
        sub = exclusion_df[(exclusion_df["circuit"] == event_name) & (~exclusion_df["excluded"])].copy()
        eligible_years = sorted(sub["year"].astype(int).tolist())
        eligible_map[event_name] = eligible_years
        print(f"{event_name}: eligible_years={eligible_years}")

    empty = [event for event, years in eligible_map.items() if len(years) == 0]
    if empty:
        print("\nSTOP - at least one circuit has zero eligible years")
        for event in empty:
            print(f"zero_eligible_years: {event}")
        return 1

    # Step 3 - seeded random draw in fixed alphabetical order.
    rng = np.random.default_rng(RNG_SEED)
    draw_rows = []
    print("\nStep 3 - Seeded random draw per circuit (alphabetical EventName order)")
    print(f"rng_seed={RNG_SEED}")
    for event_name in sorted(eligible_map.keys()):
        years = eligible_map[event_name]
        selected_year = int(rng.choice(years))
        draw_rows.append({"event_name": event_name, "drawn_year": selected_year})
        print(f"draw: circuit={event_name} | eligible_years={years} | selected_year={selected_year}")

    draw_df = pd.DataFrame(draw_rows)

    # Step 4 - resolve round numbers in base_df.
    round_lookup = (
        base_df[["EventName", "Year", "Round"]]
        .drop_duplicates()
        .rename(columns={"EventName": "event_name", "Year": "drawn_year", "Round": "round"})
    )
    final_df = (
        pd.DataFrame(TARGET_CIRCUITS)
        .merge(draw_df, on="event_name", how="left")
        .merge(round_lookup, on=["event_name", "drawn_year"], how="left")
        .sort_values(["archetype", "event_name"])
        .reset_index(drop=True)
    )

    if final_df["round"].isna().any():
        missing = final_df[final_df["round"].isna()][["event_name", "drawn_year"]]
        raise RuntimeError(f"Missing round lookup for rows: {missing.to_dict(orient='records')}")

    final_df["round"] = final_df["round"].astype(int)
    print("\nStep 4 - Final locked archetype races (with rounds)")
    print(final_df.to_string(index=False))

    print("\nStep 5 payload for config.py (LOCKED_ARCHETYPE_RACES)")
    for row in final_df.itertuples(index=False):
        print(
            "{"
            f"'archetype': '{row.archetype}', "
            f"'event_name': '{row.event_name}', "
            f"'year': {int(row.drawn_year)}, "
            f"'round': {int(row.round)}"
            "},"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())