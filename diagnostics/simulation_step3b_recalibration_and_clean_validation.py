from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation import race_model as race_model_module
from source_code.simulation.config import BASE_DF_PATH, LOCKED_ARCHETYPE_RACES
from source_code.simulation.race_model import simulate_race
from source_code.simulation.race_state import load_race_state

OUTPUT_PATH = ROOT / "data" / "diagnostics" / "simulation_step3b_recalibration_and_clean_validation_output.txt"

CHAPTER3_CAUTION_PIT_LOSS_EXPECTATION = "~4-6s"
PREVIOUS_CONFIG_CAUTION_PIT_LOSS_S = 23.1
WET_COMPOUNDS = {"WET", "INTERMEDIATE"}


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"})


def _locked_key(row: Dict[str, object]) -> Tuple[str, int, int]:
    return str(row["event_name"]), int(row["year"]), int(row["round"])


def _derive_pit_durations_for_race(race_laps: pd.DataFrame, event_name: str, year: int, round_num: int) -> pd.DataFrame:
    ordered = race_laps.sort_values(["Driver", "Round", "LapNumber"]).reset_index(drop=True)
    rows: List[Dict[str, object]] = []
    for idx in range(len(ordered) - 1):
        current_row = ordered.iloc[idx]
        next_row = ordered.iloc[idx + 1]

        if (
            pd.notna(current_row.get("PitInTime"))
            and current_row.get("Driver") == next_row.get("Driver")
            and current_row.get("Round") == next_row.get("Round")
            and pd.notna(next_row.get("PitOutTime"))
        ):
            try:
                pit_in_td = pd.to_timedelta(current_row["PitInTime"])
                pit_out_td = pd.to_timedelta(next_row["PitOutTime"])
                duration_s = (pit_out_td - pit_in_td).total_seconds()
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "event_name": event_name,
                    "year": int(year),
                    "round": int(round_num),
                    "driver": str(current_row["Driver"]),
                    "pit_in_lap": int(current_row["LapNumber"]),
                    "duration_s": float(duration_s),
                    "is_caution": bool(current_row["sc_active"] or current_row["vsc_active"]),
                    "is_green": bool(not (current_row["sc_active"] or current_row["vsc_active"])),
                }
            )

    return pd.DataFrame(rows)


def _build_real_strategy(driver_laps: pd.DataFrame) -> dict:
    ordered = driver_laps.sort_values("LapNumber").copy()
    ordered["TyreCompound"] = ordered["TyreCompound"].ffill()
    ordered = ordered.dropna(subset=["TyreCompound"]).copy()
    if ordered.empty:
        raise ValueError("Driver has no TyreCompound rows after forward fill")

    starting_compound = str(ordered.iloc[0]["TyreCompound"])
    stops = []
    prev_compound = starting_compound
    for _, row in ordered.iloc[1:].iterrows():
        compound = str(row["TyreCompound"])
        if compound != prev_compound:
            stops.append((int(row["LapNumber"]), compound))
        prev_compound = compound

    # Baseline pace for strategy-level override in a strict-then-relaxed fallback order.
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

    base_df = pd.read_csv(BASE_DF_PATH)
    base_df["sc_active"] = _to_bool(base_df["sc_active"])
    base_df["vsc_active"] = _to_bool(base_df["vsc_active"])

    lock_rows = []
    for item in LOCKED_ARCHETYPE_RACES:
        lock_rows.append(
            {
                "archetype": item["archetype"],
                "event_name": item["event_name"],
                "year": int(item["year"]),
                "round": int(item["round"]),
            }
        )

    lock_df = pd.DataFrame(lock_rows)

    lines.append("Step 0 - Locked races in scope")
    lines.append(lock_df.to_string(index=False))

    # Step 1: derive true pit durations.
    lines.append("\nStep 1 - Rebuild genuine per-stop pit durations from PitInTime/PitOutTime")
    all_duration_frames = []
    for row in lock_rows:
        race = base_df[(base_df["Year"] == row["year"]) & (base_df["Round"] == row["round"])].copy()
        d = _derive_pit_durations_for_race(race, row["event_name"], row["year"], row["round"])
        all_duration_frames.append(d)

    pit_duration_df = pd.concat(all_duration_frames, ignore_index=True)
    lines.append(f"derived_stop_count_total={len(pit_duration_df)}")
    lines.append("derived_duration_summary_by_race:")
    race_summary = (
        pit_duration_df.groupby(["event_name", "year", "round"], as_index=False)
        .agg(
            derived_count=("duration_s", "count"),
            caution_count=("is_caution", "sum"),
            green_count=("is_green", "sum"),
            mean_duration_s=("duration_s", "mean"),
        )
        .sort_values(["year", "round", "event_name"])
    )
    lines.append(race_summary.to_string(index=False))

    # Step 2: exclude implausible durations.
    lines.append("\nStep 2 - Exclude implausible durations > 120s and calibrate CAUTION_PIT_LOSS_S")
    implausible_mask = pit_duration_df["duration_s"] > 120.0
    implausible_df = pit_duration_df[implausible_mask].copy()
    cleaned_df = pit_duration_df[~implausible_mask].copy()

    lines.append(f"implausible_excluded_count={len(implausible_df)}")
    if not implausible_df.empty:
        implausible_by_race = (
            implausible_df.groupby(["event_name", "year", "round"], as_index=False)
            .size()
            .rename(columns={"size": "excluded_count"})
            .sort_values(["year", "round", "event_name"])
        )
        lines.append("implausible_exclusions_by_race:")
        lines.append(implausible_by_race.to_string(index=False))
    else:
        lines.append("implausible_exclusions_by_race: none")

    before_after_source = pd.concat(
        [
            pit_duration_df.assign(stage="before"),
            cleaned_df.assign(stage="after"),
        ],
        ignore_index=True,
    )
    before_after = (
        before_after_source.groupby(["stage", "event_name", "year", "round"], as_index=False)
        .agg(count=("duration_s", "count"))
        .sort_values(["event_name", "year", "round", "stage"])
    )
    lines.append("before_after_counts_by_race:")
    lines.append(before_after.to_string(index=False))

    cleaned_caution = cleaned_df[cleaned_df["is_caution"]].copy()
    caution_pit_loss_s = float(cleaned_caution["duration_s"].mean())
    lines.append(f"cleaned_caution_stop_count={len(cleaned_caution)}")
    lines.append(f"cleaned_caution_mean_duration_s={caution_pit_loss_s:.12f}")
    lines.append(f"previous_config_CAUTION_PIT_LOSS_S={PREVIOUS_CONFIG_CAUTION_PIT_LOSS_S:.12f}")
    lines.append(f"chapter3_documented_expectation={CHAPTER3_CAUTION_PIT_LOSS_EXPECTATION}")
    lines.append(
        "interpretation=Current correction increases CAUTION_PIT_LOSS_S because the earlier 23.1 value was copied from static pit_loss_s, not measured from caution-period stop durations."
    )

    # Step 3: reconcile pace ratio methodologies.
    lines.append("\nStep 3 - Reconcile caution-pace ratio against existing full-dataset finding")

    lines.append("session_step3_filter_logic:")
    lines.append("  - Dataset: only LOCKED_ARCHETYPE_RACES (12 race-year-round tuples)")
    lines.append("  - Green side: IsAccurate == True, sc_active == False, vsc_active == False, PitInTime isna, PitOutTime isna")
    lines.append("  - Caution side: IsAccurate == True, (sc_active OR vsc_active) == True, PitInTime isna, PitOutTime isna")
    lines.append("  - No explicit non-wet compound exclusion")
    lines.append("  - Per-race ratio = caution_mean / green_mean, then cross-race mean over defined ratios")

    lock_ratio_rows = []
    for row in lock_rows:
        race = base_df[(base_df["Year"] == row["year"]) & (base_df["Round"] == row["round"])].copy()
        is_caution = race["sc_active"] | race["vsc_active"]
        is_pit = race["PitInTime"].notna() | race["PitOutTime"].notna()

        green = race[race["IsAccurate"].eq(True) & (~is_caution) & (~is_pit)]
        caution = race[race["IsAccurate"].eq(True) & is_caution & (~is_pit)]

        green_mean = pd.to_numeric(green["fuel_corrected_laptime"], errors="coerce").dropna().mean()
        caution_mean = pd.to_numeric(caution["fuel_corrected_laptime"], errors="coerce").dropna().mean()
        ratio = np.nan
        if pd.notna(green_mean) and pd.notna(caution_mean) and green_mean != 0:
            ratio = caution_mean / green_mean

        lock_ratio_rows.append(
            {
                "event_name": row["event_name"],
                "year": row["year"],
                "round": row["round"],
                "green_n": int(len(green)),
                "caution_n": int(len(caution)),
                "green_mean": float(green_mean) if pd.notna(green_mean) else np.nan,
                "caution_mean": float(caution_mean) if pd.notna(caution_mean) else np.nan,
                "ratio": float(ratio) if pd.notna(ratio) else np.nan,
            }
        )

    lock_ratio_df = pd.DataFrame(lock_ratio_rows)
    lock_valid = lock_ratio_df[lock_ratio_df["ratio"].notna()].copy()
    lock_ratio_mean = float(lock_valid["ratio"].mean())
    lock_ratio_std = float(lock_valid["ratio"].std(ddof=1))

    lines.append("session_step3_ratio_table:")
    lines.append(lock_ratio_df.to_string(index=False))
    lines.append(f"session_step3_ratio_mean={lock_ratio_mean:.12f}")
    lines.append(f"session_step3_ratio_std={lock_ratio_std:.12f}")
    lines.append(f"session_step3_valid_race_count={len(lock_valid)} / {len(lock_ratio_df)}")

    lines.append("phase3_caution_degradation_filter_logic:")
    lines.append("  - Dataset: full 2022-2024 base_df")
    lines.append("  - Base filter: compound_known == True, TyreCompound not in {INTERMEDIATE, WET}, TyreLife notna, fuel_corrected_laptime notna")
    lines.append("  - Caution side: sc_active OR vsc_active")
    lines.append("  - Green side: not sc_active and not vsc_active")
    lines.append("  - No IsAccurate filter")
    lines.append("  - No PitInTime/PitOutTime exclusion")
    lines.append("  - Reported at compound level (SOFT, MEDIUM, HARD) across full dataset")

    compounds = ["SOFT", "MEDIUM", "HARD"]
    non_wet = {"INTERMEDIATE", "WET"}
    base_filter = (
        (base_df["compound_known"] == True)
        & (~base_df["TyreCompound"].isin(non_wet))
        & base_df["TyreLife"].notna()
        & base_df["fuel_corrected_laptime"].notna()
    )
    caution_mask = base_df["sc_active"] | base_df["vsc_active"]
    green_mask = (~base_df["sc_active"]) & (~base_df["vsc_active"])

    phase3_rows = []
    for compound in compounds:
        caution_df = base_df[base_filter & (base_df["TyreCompound"] == compound) & caution_mask].copy()
        green_df = base_df[base_filter & (base_df["TyreCompound"] == compound) & green_mask].copy()
        caution_mean = float(caution_df["fuel_corrected_laptime"].mean()) if not caution_df.empty else np.nan
        green_mean = float(green_df["fuel_corrected_laptime"].mean()) if not green_df.empty else np.nan
        ratio = caution_mean / green_mean if pd.notna(caution_mean) and pd.notna(green_mean) and green_mean != 0 else np.nan
        phase3_rows.append(
            {
                "compound": compound,
                "caution_n": int(len(caution_df)),
                "green_n": int(len(green_df)),
                "caution_mean": caution_mean,
                "green_mean": green_mean,
                "ratio": ratio,
            }
        )

    phase3_df = pd.DataFrame(phase3_rows)
    full_caution = base_df[base_filter & caution_mask].copy()
    full_green = base_df[base_filter & green_mask].copy()
    full_caution_mean = float(full_caution["fuel_corrected_laptime"].mean())
    full_green_mean = float(full_green["fuel_corrected_laptime"].mean())
    full_ratio = float(full_caution_mean / full_green_mean)

    lines.append("phase3_recomputed_compound_table:")
    lines.append(phase3_df.to_string(index=False))
    lines.append(f"phase3_full_caution_mean={full_caution_mean:.12f}")
    lines.append(f"phase3_full_green_mean={full_green_mean:.12f}")
    lines.append(f"phase3_full_ratio={full_ratio:.12f}")

    lines.append("methodology_differences:")
    lines.append("  - Scope: locked sample (12 races) vs full 2022-2024 dataset")
    lines.append("  - Sample size: locked caution laps are sparse (many zero-caution races) vs thousands of full-dataset caution laps")
    lines.append("  - Exclusions: session method excludes pit laps and requires IsAccurate; phase3 method does neither")
    lines.append("  - Wet handling: session method does not explicitly remove wet compounds; phase3 explicitly excludes WET/INTERMEDIATE")
    lines.append("  - Aggregation: session method averages race-level ratios; phase3 compares global compound/full means")

    recommended_caution_pace_ratio = full_ratio
    lines.append(
        "reconciled_recommendation=Use phase3 full-dataset methodology for CAUTION_PACE_RATIO because caution-slowdown is a global pace-regime effect and the locked-sample estimator is underpowered (5/12 races with zero caution laps)."
    )
    lines.append(f"recommended_CAUTION_PACE_RATIO={recommended_caution_pace_ratio:.12f}")

    # Step 4: race screening.
    lines.append("\nStep 4 - Screen all 12 locked races for clean deterministic validation")
    screen_rows = []
    for row in lock_rows:
        race = base_df[(base_df["Year"] == row["year"]) & (base_df["Round"] == row["round"])].copy()
        race_durations = pit_duration_df[
            (pit_duration_df["year"] == row["year"])
            & (pit_duration_df["round"] == row["round"])
            & (pit_duration_df["event_name"] == row["event_name"])
        ]

        has_wet = bool(race["TyreCompound"].isin(WET_COMPOUNDS).any())
        has_red_artifact = bool((race_durations["duration_s"] > 120).any()) if not race_durations.empty else False

        unique_drivers = sorted(race["Driver"].dropna().astype(str).unique().tolist())
        drivers_with_position = 0
        for driver in unique_drivers:
            if race.loc[race["Driver"].astype(str) == driver, "Position"].notna().any():
                drivers_with_position += 1
        all20_have_position = len(unique_drivers) == 20 and drivers_with_position == 20

        race_length = int(pd.to_numeric(race["LapNumber"], errors="coerce").max())

        screen_rows.append(
            {
                "event_name": row["event_name"],
                "year": row["year"],
                "round": row["round"],
                "has_wet_or_intermediate": has_wet,
                "has_duration_gt120_pre_exclusion": has_red_artifact,
                "all20_have_non_null_position_somewhere": all20_have_position,
                "driver_count": len(unique_drivers),
                "drivers_with_any_position": drivers_with_position,
                "race_length": race_length,
            }
        )

    screen_df = pd.DataFrame(screen_rows).sort_values(["year", "round", "event_name"])
    lines.append(screen_df.to_string(index=False))

    clean_candidates = screen_df[
        (~screen_df["has_wet_or_intermediate"])
        & (~screen_df["has_duration_gt120_pre_exclusion"])
        & (screen_df["all20_have_non_null_position_somewhere"])
    ].copy()

    if clean_candidates.empty:
        raise ValueError("No locked race passed all cleanliness screens")

    # Prioritize cleanliness first; among equally clean races choose the longest one
    # so the validation remains deterministic but not trivial.
    selected = clean_candidates.sort_values(["race_length", "year", "round", "event_name"], ascending=[False, True, True, True]).iloc[0]
    selected_event = str(selected["event_name"])
    selected_year = int(selected["year"])
    selected_round = int(selected["round"])

    lines.append("clean_candidates:")
    lines.append(clean_candidates.sort_values(["race_length", "year", "round", "event_name"], ascending=[False, True, True, True]).to_string(index=False))
    lines.append(
        f"selected_validation_race={selected_event} ({selected_year} round {selected_round})"
    )

    # Step 5 + 6: corrected historical extraction + deterministic validation.
    lines.append("\nStep 5 - Ground-truth extraction correction")
    lines.append(
        "historical_order_method=For each driver, pick the row with highest LapNumber where Position is non-null (last valid classified position), not the strictly last row."
    )

    lines.append("\nStep 6 - Deterministic validation on screened clean race")
    state = load_race_state(selected_year, selected_round)
    laps = state["laps"].copy()
    laps["sc_active"] = _to_bool(laps["sc_active"])
    laps["vsc_active"] = _to_bool(laps["vsc_active"])

    # Patch race-model constants in-process so this validation uses corrected Step 2/3 values.
    race_model_module.CAUTION_PIT_LOSS_S = caution_pit_loss_s
    race_model_module.CAUTION_PACE_RATIO = recommended_caution_pace_ratio

    caution_schedule = {}
    for lap_number, lap_group in laps.groupby("LapNumber"):
        caution_schedule[int(lap_number)] = bool(lap_group["sc_active"].any() or lap_group["vsc_active"].any())

    strategies_by_driver = {}
    for driver, group in laps.groupby("Driver"):
        strategies_by_driver[str(driver)] = _build_real_strategy(group.copy())

    sim = simulate_race(
        race_state=state,
        strategies_by_driver=strategies_by_driver,
        caution_schedule=caution_schedule,
        lambda_=1.0,
    )

    model_order = sim["finishing_order"]
    historical_order = _historical_order_last_valid_position(laps)

    model_rank = {driver: idx + 1 for idx, driver in enumerate(model_order)}
    historical_rank = {driver: idx + 1 for idx, driver in enumerate(historical_order)}

    shared = sorted(set(model_rank).intersection(historical_rank))
    exact_matches = sum(1 for driver in shared if model_rank[driver] == historical_rank[driver])
    mae = float(np.mean([abs(model_rank[d] - historical_rank[d]) for d in shared]))

    lines.append(f"validation_race_length={sim['race_length']}")
    lines.append(f"model_order={' > '.join(model_order)}")
    lines.append(f"historical_order={' > '.join(historical_order)}")
    lines.append(f"exact_position_matches={exact_matches} / {len(shared)}")
    lines.append(f"mean_absolute_position_error={mae:.12f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
