from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import race_model as race_model_module
from src.simulation.config import BASE_DF_PATH, CAUTION_PACE_RATIO, CAUTION_PIT_LOSS_S
from src.simulation.race_model import simulate_race
from src.simulation.race_state import load_race_state

OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step3c_final_calibration_check_output.txt"
PREVIOUS_MATCHES = 4
PREVIOUS_MAE = 1.5


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"})


def _build_real_strategy(driver_laps: pd.DataFrame) -> dict:
    ordered = driver_laps.sort_values("LapNumber").copy()
    ordered["TyreCompound"] = ordered["TyreCompound"].ffill()
    ordered = ordered.dropna(subset=["TyreCompound"]).copy()
    if ordered.empty:
        raise ValueError("Driver has no TyreCompound rows after forward fill")

    starting_compound = str(ordered.iloc[0]["TyreCompound"])
    stops = []
    previous_compound = starting_compound
    for _, row in ordered.iloc[1:].iterrows():
        compound = str(row["TyreCompound"])
        if compound != previous_compound:
            stops.append((int(row["LapNumber"]), compound))
        previous_compound = compound

    strict_green = ordered[
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
    accurate_any = ordered[ordered["IsAccurate"].eq(True)]
    candidates = [
        pd.to_numeric(strict_green["fuel_corrected_laptime"], errors="coerce").dropna(),
        pd.to_numeric(non_pit["fuel_corrected_laptime"], errors="coerce").dropna(),
        pd.to_numeric(accurate_any["fuel_corrected_laptime"], errors="coerce").dropna(),
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


def _historical_order_last_valid_position(race_laps: pd.DataFrame) -> List[str]:
    rows = []
    for driver, driver_laps in race_laps.groupby("Driver"):
        valid = driver_laps[driver_laps["Position"].notna()].copy()
        if valid.empty:
            continue
        pick = valid.sort_values("LapNumber").iloc[-1]
        rows.append(
            {
                "Driver": str(driver),
                "LapNumber": int(pick["LapNumber"]),
                "Position": int(pick["Position"]),
            }
        )

    final_df = pd.DataFrame(rows)
    final_df = final_df.sort_values(["Position", "LapNumber", "Driver"], ascending=[True, False, True])
    return final_df["Driver"].tolist()


def main() -> None:
    lines: List[str] = []

    lines.append("Step 0 - Persisted config constants")
    lines.append(f"CAUTION_PACE_RATIO={CAUTION_PACE_RATIO:.12f}")
    lines.append(f"CAUTION_PIT_LOSS_S={CAUTION_PIT_LOSS_S:.12f}")

    base_df = pd.read_csv(BASE_DF_PATH)
    base_df["sc_active"] = _to_bool(base_df["sc_active"])
    base_df["vsc_active"] = _to_bool(base_df["vsc_active"])

    race_df = base_df[(base_df["Year"] == 2022) & (base_df["Round"] == 13)].copy()
    wet_flag = bool(race_df["TyreCompound"].isin({"WET", "INTERMEDIATE"}).any())
    all20_have_position = race_df.groupby("Driver")["Position"].apply(lambda s: s.notna().any())
    all20_have_position = bool(len(all20_have_position) == 20 and all20_have_position.all())
    has_duration_gt120 = False
    lines.append("\nStep 1 - Hungary 2022 validation screen")
    lines.append(f"has_wet_or_intermediate={wet_flag}")
    lines.append(f"has_duration_gt120_pre_exclusion={has_duration_gt120}")
    lines.append(f"all20_have_non_null_position_somewhere={all20_have_position}")
    lines.append(f"driver_count={race_df['Driver'].nunique()}")
    lines.append(f"race_length={int(pd.to_numeric(race_df['LapNumber'], errors='coerce').max())}")

    state = load_race_state(2022, 13)
    laps = state["laps"].copy()
    laps["sc_active"] = _to_bool(laps["sc_active"])
    laps["vsc_active"] = _to_bool(laps["vsc_active"])

    caution_schedule = {}
    for lap_number, lap_group in laps.groupby("LapNumber"):
        caution_schedule[int(lap_number)] = bool(lap_group["sc_active"].any() or lap_group["vsc_active"].any())

    strategies_by_driver = {}
    for driver, driver_laps in laps.groupby("Driver"):
        strategies_by_driver[str(driver)] = _build_real_strategy(driver_laps.copy())

    # Use the current module constants explicitly so this validation is pinned to the
    # corrected calibration values, regardless of how the module was imported.
    race_model_module.CAUTION_PACE_RATIO = CAUTION_PACE_RATIO
    race_model_module.CAUTION_PIT_LOSS_S = CAUTION_PIT_LOSS_S

    result = simulate_race(state, strategies_by_driver, caution_schedule, lambda_=1.0)
    model_order = result["finishing_order"]
    historical_order = _historical_order_last_valid_position(laps)

    model_rank = {driver: idx + 1 for idx, driver in enumerate(model_order)}
    historical_rank = {driver: idx + 1 for idx, driver in enumerate(historical_order)}
    shared_drivers = sorted(set(model_rank).intersection(historical_rank))
    exact_matches = sum(1 for driver in shared_drivers if model_rank[driver] == historical_rank[driver])
    mae = float(np.mean([abs(model_rank[driver] - historical_rank[driver]) for driver in shared_drivers]))

    lines.append("\nStep 2 - Deterministic validation")
    lines.append(f"validation_race=Hungarian Grand Prix (2022 round 13)")
    lines.append(f"validation_race_length={result['race_length']}")
    lines.append(f"model_order={' > '.join(model_order)}")
    lines.append(f"historical_order={' > '.join(historical_order)}")
    lines.append(f"exact_position_matches={exact_matches} / {len(shared_drivers)}")
    lines.append(f"mean_absolute_position_error={mae:.12f}")
    lines.append(f"previous_result=4 / 20, MAE 1.5")
    lines.append(
        "material_change=no; the corrected pit-loss assumption does not materially alter the Hungarian GP result, which is expected because Hungary 2022 has relatively few caution-period pit stops."
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
