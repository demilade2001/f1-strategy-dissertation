from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.config import LOCKED_ARCHETYPE_RACES, RNG_SEED


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
COMPROMISED_MODELS = {"invalid_data", "insufficient", "missing"}


def main():
    deg = pd.read_csv(DEG_STATS_PATH)
    base_df = pd.read_csv(BASE_DF_PATH)

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
            compromised_mask = (sub["low_sample"] == True) | sub["model"].astype(str).str.lower().isin(COMPROMISED_MODELS)
            compromised_count = int(compromised_mask.sum())

            old_excluded = low_sample_count >= 2
            new_excluded = compromised_count >= 2

            rows.append(
                {
                    "circuit": event_name,
                    "year": int(year),
                    "compromised_count": compromised_count,
                    "excluded": bool(new_excluded),
                    "old_excluded": bool(old_excluded),
                    "new_excluded": bool(new_excluded),
                    "changed_exclusion_status": bool(old_excluded != new_excluded),
                }
            )

    exclusion_df = pd.DataFrame(rows).sort_values(["circuit", "year"]).reset_index(drop=True)
    print("Step 1 - Amended exclusion table (compromised = low_sample or invalid_data/insufficient/missing)")
    print(exclusion_df.to_string(index=False))

    print("\nStep 2 - Eligible years per circuit under amended rule")
    eligible_map = {}
    for event_name in sorted([x["event_name"] for x in TARGET_CIRCUITS]):
        sub = exclusion_df[(exclusion_df["circuit"] == event_name) & (~exclusion_df["new_excluded"])].copy()
        eligible_years = sorted(sub["year"].astype(int).tolist())
        eligible_map[event_name] = eligible_years
        print(f"{event_name}: eligible_years={eligible_years}")

    empty = [event for event, years in eligible_map.items() if len(years) == 0]
    if empty:
        print("\nSTOP - at least one circuit has zero eligible years under amended rule")
        for event in empty:
            print(f"zero_eligible_years: {event}")
        return 1

    print("\nStep 3 - Fresh seeded draw from scratch (same RNG_SEED)")
    print(f"rng_seed={RNG_SEED}")
    rng = np.random.default_rng(RNG_SEED)
    new_draw_rows = []
    for event_name in sorted(eligible_map.keys()):
        years = eligible_map[event_name]
        selected_year = int(rng.choice(years))
        new_draw_rows.append({"event_name": event_name, "new_drawn_year": selected_year})
        print(f"draw: circuit={event_name} | eligible_years={years} | selected_year={selected_year}")

    new_draw_df = pd.DataFrame(new_draw_rows)

    print("\nStep 4 - Diff versus original lock")
    old_draw_df = pd.DataFrame(LOCKED_ARCHETYPE_RACES)[["event_name", "year"]].rename(columns={"year": "original_drawn_year"})
    diff_df = (
        old_draw_df
        .merge(new_draw_df, on="event_name", how="inner")
        .sort_values("event_name")
        .reset_index(drop=True)
    )
    diff_df["changed"] = diff_df["original_drawn_year"] != diff_df["new_drawn_year"]
    print(diff_df.to_string(index=False))
    print(f"changed_count={int(diff_df['changed'].sum())}")

    print("\nStep 5 - Resolve rounds and output new lock list")
    round_lookup = (
        base_df[["EventName", "Year", "Round"]]
        .drop_duplicates()
        .rename(columns={"EventName": "event_name", "Year": "new_drawn_year", "Round": "round"})
    )

    final_df = (
        pd.DataFrame(TARGET_CIRCUITS)
        .merge(new_draw_df, on="event_name", how="left")
        .merge(round_lookup, on=["event_name", "new_drawn_year"], how="left")
        .sort_values(["archetype", "event_name"])
        .reset_index(drop=True)
    )

    if final_df["round"].isna().any():
        missing = final_df[final_df["round"].isna()][["event_name", "new_drawn_year"]]
        raise RuntimeError(f"Missing round lookup rows: {missing.to_dict(orient='records')}")

    final_df["round"] = final_df["round"].astype(int)
    print(final_df.to_string(index=False))

    print("\nStep 5 payload for config.py (corrected LOCKED_ARCHETYPE_RACES)")
    for row in final_df.itertuples(index=False):
        print(
            "{"
            f"'archetype': '{row.archetype}', "
            f"'event_name': '{row.event_name}', "
            f"'year': {int(row.new_drawn_year)}, "
            f"'round': {int(row.round)}"
            "},"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
