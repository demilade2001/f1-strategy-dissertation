from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.config import (
    BASE_DF_PATH,
    CAUTION_PACE_RATIO,
    CAUTION_PIT_LOSS_S,
    LOCKED_ARCHETYPE_RACES,
)
from src.simulation.race_model import simulate_race
from src.simulation.race_state import load_race_state


OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step3_validation_output.txt"


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _build_strategy(driver_laps: pd.DataFrame) -> dict:
    ordered = driver_laps.sort_values("LapNumber").copy()
    ordered["TyreCompound"] = ordered["TyreCompound"].ffill()
    ordered = ordered.dropna(subset=["TyreCompound"])

    starting_compound = str(ordered.iloc[0]["TyreCompound"])
    stops = []
    previous_compound = starting_compound
    for _, row in ordered.iloc[1:].iterrows():
        compound = str(row["TyreCompound"])
        if compound != previous_compound:
            stops.append((int(row["LapNumber"]), compound))
        previous_compound = compound

    green = ordered[
        ordered["IsAccurate"].eq(True)
        & ~ordered["sc_active"].fillna(False)
        & ~ordered["vsc_active"].fillna(False)
        & ordered["PitInTime"].isna()
        & ordered["PitOutTime"].isna()
    ]
    non_pit = ordered[
        ordered["IsAccurate"].eq(True)
        & ordered["PitInTime"].isna()
        & ordered["PitOutTime"].isna()
    ]
    accurate_only = ordered[ordered["IsAccurate"].eq(True)]
    candidates = [
        pd.to_numeric(green["fuel_corrected_laptime"], errors="coerce").dropna(),
        pd.to_numeric(non_pit["fuel_corrected_laptime"], errors="coerce").dropna(),
        pd.to_numeric(accurate_only["fuel_corrected_laptime"], errors="coerce").dropna(),
        pd.to_numeric(ordered["fuel_corrected_laptime"], errors="coerce").dropna(),
    ]
    baseline_pace_s = np.nan
    for values in candidates:
        if not values.empty:
            baseline_pace_s = float(values.median())
            break

    return {
        "starting_compound": starting_compound,
        "stops": stops,
        "baseline_pace_s": baseline_pace_s,
    }


def _historical_order(race_laps: pd.DataFrame) -> list[str]:
    final_rows = race_laps.sort_values(["Driver", "LapNumber"]).groupby("Driver", as_index=False).tail(1).copy()
    final_rows["position_sort"] = final_rows["Position"].fillna(np.inf)
    final_rows["lap_sort"] = final_rows["LapNumber"].fillna(-1)
    classified = final_rows[final_rows["Position"].notna()].sort_values(["Position", "Driver"])
    unclassified = final_rows[final_rows["Position"].isna()].sort_values(["lap_sort", "Driver"], ascending=[False, True])
    ordered = pd.concat([classified, unclassified], ignore_index=True)
    return ordered["Driver"].astype(str).tolist()


def main() -> None:
    base_df = pd.read_csv(BASE_DF_PATH)
    base_df["sc_active"] = _bool_series(base_df["sc_active"])
    base_df["vsc_active"] = _bool_series(base_df["vsc_active"])

    output_lines = []
    output_lines.append("Step 0 - Existing points system check")
    output_lines.append("The standard F1 points table already exists in src/features.py, so config.py was not modified for points.")

    output_lines.append("\nStep 1 - Caution sampler interface")
    output_lines.append("sample_caution_schedule(prob_by_lap, rng) returns a per-lap boolean schedule and accepts any per-lap probability source.")

    output_lines.append("\nStep 2 - Locked-race caution pace ratio")
    ratio_rows = []
    for race in LOCKED_ARCHETYPE_RACES:
        race_df = base_df[(base_df["Year"] == race["year"]) & (base_df["Round"] == race["round"])]
        is_caution = race_df["sc_active"] | race_df["vsc_active"]
        is_pit = race_df["PitInTime"].notna() | race_df["PitOutTime"].notna()
        green = race_df[race_df["IsAccurate"].eq(True) & ~is_caution & ~is_pit]
        caution = race_df[race_df["IsAccurate"].eq(True) & is_caution & ~is_pit]
        green_mean = pd.to_numeric(green["fuel_corrected_laptime"], errors="coerce").dropna().mean()
        caution_mean = pd.to_numeric(caution["fuel_corrected_laptime"], errors="coerce").dropna().mean()
        ratio = caution_mean / green_mean if pd.notna(caution_mean) and pd.notna(green_mean) and green_mean else np.nan
        ratio_rows.append({
            "event_name": race["event_name"],
            "year": race["year"],
            "round": race["round"],
            "green_n": int(len(green)),
            "caution_n": int(len(caution)),
            "green_mean": float(green_mean) if pd.notna(green_mean) else np.nan,
            "caution_mean": float(caution_mean) if pd.notna(caution_mean) else np.nan,
            "ratio": float(ratio) if pd.notna(ratio) else np.nan,
        })

    ratio_df = pd.DataFrame(ratio_rows)
    valid_ratios = ratio_df[ratio_df["ratio"].notna()].copy()
    output_lines.append(ratio_df.to_string(index=False))
    output_lines.append(f"ratio_mean={valid_ratios['ratio'].mean():.12f}")
    output_lines.append(f"ratio_std={valid_ratios['ratio'].std(ddof=1):.12f}")
    output_lines.append(f"valid_ratio_races={len(valid_ratios)} / {len(LOCKED_ARCHETYPE_RACES)}")
    output_lines.append(f"config_CAUTION_PACE_RATIO={CAUTION_PACE_RATIO:.12f}")

    output_lines.append("\nStep 3 - Locked-race caution pit-loss calibration")
    pit_rows = []
    for race in LOCKED_ARCHETYPE_RACES:
        race_df = base_df[(base_df["Year"] == race["year"]) & (base_df["Round"] == race["round"])]
        pit_mask = (race_df["PitInTime"].notna() | race_df["PitOutTime"].notna()) & (race_df["sc_active"] | race_df["vsc_active"])
        pit = race_df[pit_mask].copy()
        pit_loss = pd.to_numeric(pit["pit_loss_s"], errors="coerce").dropna()
        pit_rows.append({
            "event_name": race["event_name"],
            "year": race["year"],
            "round": race["round"],
            "caution_pit_stops": int(len(pit)),
            "pit_loss_rows": int(pit_loss.notna().sum()),
            "mean_pit_loss_s": float(pit_loss.mean()) if not pit_loss.empty else np.nan,
        })

    pit_df = pd.DataFrame(pit_rows)
    pit_valid = pit_df[pit_df["mean_pit_loss_s"].notna()].copy()
    caution_pit_stop_count = int(pit_df["caution_pit_stops"].sum())
    caution_pit_loss_mean = float(pit_valid["mean_pit_loss_s"].mean()) if not pit_valid.empty else np.nan
    output_lines.append(pit_df.to_string(index=False))
    output_lines.append(f"caution_pit_stop_count={caution_pit_stop_count}")
    output_lines.append(f"caution_pit_loss_mean={caution_pit_loss_mean:.12f}")
    output_lines.append(f"config_CAUTION_PIT_LOSS_S={CAUTION_PIT_LOSS_S:.12f}")
    output_lines.append("Note: the processed base_df.csv does not include PitDuration_s, so this calibration uses the available pit_loss_s proxy.")

    output_lines.append("\nStep 6 - Japanese Grand Prix 2022 deterministic validation")
    state = load_race_state(2022, 18)
    race_laps = state["laps"].copy()
    race_laps["sc_active"] = _bool_series(race_laps["sc_active"])
    race_laps["vsc_active"] = _bool_series(race_laps["vsc_active"])

    caution_schedule = {}
    for lap_number, lap_group in race_laps.groupby("LapNumber"):
        caution_schedule[int(lap_number)] = bool(lap_group["sc_active"].any() or lap_group["vsc_active"].any())

    strategies_by_driver = {}
    for driver, driver_laps in race_laps.groupby("Driver"):
        strategies_by_driver[str(driver)] = _build_strategy(driver_laps.copy())

    result = simulate_race(state, strategies_by_driver, caution_schedule, lambda_=1.0)
    model_order = result["finishing_order"]
    historical_order = _historical_order(race_laps)

    model_rank = {driver: idx + 1 for idx, driver in enumerate(model_order)}
    historical_rank = {driver: idx + 1 for idx, driver in enumerate(historical_order)}
    shared_drivers = sorted(set(model_rank).intersection(historical_rank))
    exact_matches = sum(1 for driver in shared_drivers if model_rank[driver] == historical_rank[driver])
    mae = float(np.mean([abs(model_rank[driver] - historical_rank[driver]) for driver in shared_drivers]))

    output_lines.append(f"race_length={result['race_length']}")
    output_lines.append(f"model_order={' > '.join(model_order)}")
    output_lines.append(f"historical_order={' > '.join(historical_order)}")
    output_lines.append(f"exact_position_matches={exact_matches} / {len(shared_drivers)}")
    output_lines.append(f"mean_absolute_position_error={mae:.12f}")
    output_lines.append(f"shared_driver_count={len(shared_drivers)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print("\n".join(output_lines))
    print(f"\nSaved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()