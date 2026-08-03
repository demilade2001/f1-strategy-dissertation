from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

from .config import BASE_DF_PATH, CAUTION_PACE_RATIO, CAUTION_PIT_LOSS_S, is_valid_subject
from .race_model import simulate_race_from_prepared, simulate_race_with_driver_laps
from .sc_sampler import sample_caution_schedule

XGB_PROBABILITY_COLUMN = "y_pred_xgb_classweight_cal"


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(bool)


def _extract_actual_strategy(driver_laps: pd.DataFrame) -> dict:
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


def build_actual_strategies(race_laps: pd.DataFrame) -> Dict[str, dict]:
    strategies: Dict[str, dict] = {}
    event_name = str(race_laps["EventName"].dropna().iloc[0])
    year = int(pd.to_numeric(race_laps["Year"], errors="coerce").dropna().iloc[0])
    round_num = int(pd.to_numeric(race_laps["Round"], errors="coerce").dropna().iloc[0])

    for driver, driver_laps in race_laps.groupby("Driver"):
        strategy = _extract_actual_strategy(driver_laps.copy())
        strategy["year"] = year
        strategy["round"] = round_num
        strategy["event_name"] = event_name
        strategies[str(driver)] = strategy

    return strategies


def select_midfield_subject_driver(race_laps: pd.DataFrame) -> str:
    final_rows = []
    for driver, driver_laps in race_laps.groupby("Driver"):
        valid = driver_laps[driver_laps["Position"].notna()].copy()
        if valid.empty:
            continue
        pick = valid.sort_values("LapNumber").iloc[-1]
        team = str(pick["Team"])
        if not is_valid_subject(team):
            continue
        final_rows.append(
            {
                "Driver": str(driver),
                "Position": int(pick["Position"]),
                "LapNumber": int(pick["LapNumber"]),
                "Team": team,
            }
        )

    if not final_rows:
        raise ValueError("No midfield subject driver found in this race")

    chosen = pd.DataFrame(final_rows).sort_values(["Position", "LapNumber", "Driver"], ascending=[True, False, True]).iloc[0]
    return str(chosen["Driver"])


def build_probability_sources(race_state: dict) -> Dict[str, Dict[int, float]]:
    laps = race_state["laps"].copy()
    if "has_lap_level_prob" not in laps.columns:
        raise ValueError("race_state laps must include has_lap_level_prob")
    if XGB_PROBABILITY_COLUMN not in laps.columns:
        raise ValueError(f"race_state laps must include {XGB_PROBABILITY_COLUMN}")

    circuit_sc_rate = float(race_state["circuit_sc_rate"])
    merged: Dict[int, float] = {}
    for lap_number, lap_group in laps.groupby("LapNumber"):
        lap_probs = pd.to_numeric(
            lap_group.loc[lap_group["has_lap_level_prob"].eq(True), XGB_PROBABILITY_COLUMN],
            errors="coerce",
        ).dropna()
        merged[int(lap_number)] = float(lap_probs.mean()) if not lap_probs.empty else circuit_sc_rate

    flat = {lap_number: circuit_sc_rate for lap_number in merged}
    return {
        "merged": merged,
        "flat": flat,
    }


def _points_for_position(position: int) -> int:
    points_map = {
        1: 25,
        2: 18,
        3: 15,
        4: 12,
        5: 10,
        6: 8,
        7: 6,
        8: 4,
        9: 2,
        10: 1,
    }
    return int(points_map.get(int(position), 0))


def run_monte_carlo(race_state, subject_driver, subject_strategy, prob_by_lap, lambda_, n_iterations, rng) -> dict:
    """Run a fixed-strategy Monte Carlo simulation under shared caution schedules.

    All per-driver state is pre-computed once before the iteration loop.
    The loop body calls only ``simulate_race_from_prepared``, which performs
    zero DataFrame access and rebuilds no lookup tables per iteration.
    """

    laps = race_state["laps"].copy()
    laps["sc_active"] = _to_bool(laps["sc_active"])
    laps["vsc_active"] = _to_bool(laps["vsc_active"])

    actual_strategies = build_actual_strategies(laps)
    if subject_driver not in actual_strategies:
        raise ValueError(f"Subject driver {subject_driver} is not present in the race")

    # Merge the caller-supplied subject strategy (may override compound/stops/pace)
    # and ensure all metadata fields are present.
    strategies_by_driver = dict(actual_strategies)
    subject_strategy = dict(subject_strategy)
    year = int(pd.to_numeric(laps["Year"], errors="coerce").dropna().iloc[0])
    round_num = int(pd.to_numeric(laps["Round"], errors="coerce").dropna().iloc[0])
    event_name = str(laps["EventName"].dropna().iloc[0])
    subject_strategy.setdefault("year", year)
    subject_strategy.setdefault("round", round_num)
    subject_strategy.setdefault("event_name", event_name)
    strategies_by_driver[subject_driver] = subject_strategy

    # --- Pre-compute everything that is invariant across iterations ---

    # 1. Race length (constant across all iterations)
    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())

    # 2. Degradation lookup built once from deg_stats
    from .race_model import _build_degradation_lookup, _driver_median_green_pace, _driver_pit_loss
    degradation_lookup = _build_degradation_lookup(race_state.get("deg_stats"))

    # 3. Per-driver lap slices (used only for pit_loss + baseline_pace fallback)
    driver_laps_by_driver = {
        str(driver): driver_laps.copy()
        for driver, driver_laps in laps.groupby("Driver")
    }

    # 4. Race-wide fallback pace (used if a driver has no usable laps)
    race_baseline_pace = _driver_median_green_pace(laps)

    # 5. Fully-resolved strategies (baseline_pace_s guaranteed non-NaN)
    prepared_strategies: Dict[str, dict] = {}
    for driver, strategy in strategies_by_driver.items():
        s = dict(strategy)
        s.setdefault("event_name", event_name)
        s.setdefault("year", year)
        s.setdefault("round", round_num)
        if "baseline_pace_s" not in s or pd.isna(s.get("baseline_pace_s")):
            driver_laps = driver_laps_by_driver.get(driver)
            try:
                s["baseline_pace_s"] = _driver_median_green_pace(driver_laps) if driver_laps is not None else race_baseline_pace
            except ValueError:
                s["baseline_pace_s"] = race_baseline_pace
        prepared_strategies[driver] = s

    # 6. Pit-loss seconds per driver (constant; uses actual recorded pit duration)
    pit_loss_by_driver: Dict[str, float] = {
        driver: _driver_pit_loss(driver_laps_by_driver[driver])
        if driver in driver_laps_by_driver else 0.0
        for driver in prepared_strategies
    }

    # --- Iteration loop: only the caution draw + per-lap simulation runs ---
    points_samples: List[float] = []
    position_samples: List[float] = []

    for _ in range(int(n_iterations)):
        caution_schedule = sample_caution_schedule(prob_by_lap, rng)
        result = simulate_race_from_prepared(
            race_length=race_length,
            prepared_strategies=prepared_strategies,
            pit_loss_by_driver=pit_loss_by_driver,
            caution_schedule=caution_schedule,
            degradation_lookup=degradation_lookup,
            lambda_=lambda_,
            collect_trace=False,
        )
        subject_position = int(result["position_by_driver"][subject_driver])
        subject_points = int(result["points_by_driver"][subject_driver])
        points_samples.append(float(subject_points))
        position_samples.append(float(subject_position))

    points_array = np.asarray(points_samples, dtype=float)
    position_array = np.asarray(position_samples, dtype=float)

    return {
        "subject_driver": subject_driver,
        "n_iterations": int(n_iterations),
        "mean_points": float(points_array.mean()),
        "std_points": float(points_array.std(ddof=1)) if len(points_array) > 1 else 0.0,
        "points_distribution": points_samples,
        "mean_position": float(position_array.mean()),
        "std_position": float(position_array.std(ddof=1)) if len(position_array) > 1 else 0.0,
    }
