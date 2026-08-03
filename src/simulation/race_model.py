from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CAUTION_PACE_RATIO, CAUTION_PIT_LOSS_S


POINTS_BY_POSITION = {
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


def _points_for_position(position: int) -> int:
    return int(POINTS_BY_POSITION.get(int(position), 0))


def _normalise_bool(value: Any) -> bool:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _strategy_stops(strategy: Mapping[str, Any]) -> List[Tuple[int, str]]:
    if "stops" in strategy and strategy["stops"] is not None:
        stops = []
        for stop in strategy["stops"]:
            if isinstance(stop, Mapping):
                pit_lap = int(stop["pit_lap"])
                compound = str(stop["compound"])
            else:
                pit_lap, compound = stop
                pit_lap = int(pit_lap)
                compound = str(compound)
            stops.append((pit_lap, compound))
        return sorted(stops, key=lambda item: item[0])

    pit_laps = list(strategy.get("pit_laps", []))
    compounds = list(strategy.get("compound_sequence", []))
    if pit_laps and len(pit_laps) != len(compounds):
        raise ValueError("pit_laps and compound_sequence must have the same length")
    return sorted([(int(lap), str(compound)) for lap, compound in zip(pit_laps, compounds)], key=lambda item: item[0])


def _strategy_starting_compound(strategy: Mapping[str, Any]) -> str:
    for key in ("starting_compound", "initial_compound"):
        if key in strategy and strategy[key] is not None:
            return str(strategy[key])
    raise ValueError("strategy must include a starting_compound")


def _strategy_baseline_pace(strategy: Mapping[str, Any]) -> float:
    for key in ("baseline_pace_s", "driver_median_green_fuel_corrected_laptime"):
        if key in strategy and strategy[key] is not None:
            return float(strategy[key])
    raise ValueError("strategy must include baseline_pace_s")


def _build_degradation_lookup(degradation_stats: Any) -> Dict[Tuple[Any, ...], float]:
    if degradation_stats is None:
        return {}
    if isinstance(degradation_stats, Mapping):
        lookup: Dict[Tuple[Any, ...], float] = {}
        for key, value in degradation_stats.items():
            lookup[tuple(key) if isinstance(key, tuple) else (key,)] = float(value)
        return lookup

    if not isinstance(degradation_stats, pd.DataFrame):
        raise TypeError("degradation_stats must be a DataFrame or mapping")

    frame = degradation_stats.copy()
    if "deg_rate_final_assigned" not in frame.columns:
        raise ValueError("degradation_stats must include deg_rate_final_assigned")

    lookup: Dict[Tuple[Any, ...], float] = {}
    if {"Year", "EventName", "TyreCompound"}.issubset(frame.columns):
        for _, row in frame.iterrows():
            value = row["deg_rate_final_assigned"]
            if pd.isna(value):
                continue
            lookup[(int(row["Year"]), str(row["EventName"]), str(row["TyreCompound"]))] = float(value)
    elif "TyreCompound" in frame.columns:
        for _, row in frame.iterrows():
            value = row["deg_rate_final_assigned"]
            if pd.isna(value):
                continue
            lookup[(str(row["TyreCompound"]),)] = float(value)
    return lookup


def _lookup_degradation_rate(strategy: Mapping[str, Any], degradation_lookup: Mapping[Tuple[Any, ...], float], compound: str) -> float:
    year = strategy.get("year")
    event_name = strategy.get("event_name")
    if year is not None and event_name is not None:
        key = (int(year), str(event_name), str(compound))
        if key in degradation_lookup:
            return float(degradation_lookup[key])

    compound_key = (str(compound),)
    if compound_key in degradation_lookup:
        return float(degradation_lookup[compound_key])

    return 0.0


def _compound_by_lap(starting_compound: str, stops: Sequence[Tuple[int, str]], race_length: int) -> Dict[int, str]:
    compound_by_lap: Dict[int, str] = {}
    sorted_stops = sorted((int(lap), str(compound)) for lap, compound in stops)
    current_compound = str(starting_compound)
    stop_idx = 0
    next_stop = sorted_stops[stop_idx] if sorted_stops else None

    for lap in range(1, race_length + 1):
        compound_by_lap[lap] = current_compound
        if next_stop is not None and lap == next_stop[0]:
            current_compound = next_stop[1]
            stop_idx += 1
            next_stop = sorted_stops[stop_idx] if stop_idx < len(sorted_stops) else None
    return compound_by_lap


def simulate_car_race(
    strategy,
    caution_schedule,
    degradation_stats,
    race_length,
    lambda_,
    pit_loss_s,
    caution_pit_loss_s,
    caution_pace_ratio,
    degradation_lookup: Optional[Mapping[Tuple[Any, ...], float]] = None,
    collect_trace: bool = True,
) -> dict:
    """Simulate one deterministic race for a single car.

    The strategy is expected to carry the car-specific baseline pace as
    ``baseline_pace_s`` plus the compound sequence / pit laps.
    """

    strategy = dict(strategy)
    starting_compound = _strategy_starting_compound(strategy)
    stops = _strategy_stops(strategy)
    baseline_pace_s = _strategy_baseline_pace(strategy)
    if degradation_lookup is None:
        degradation_lookup = _build_degradation_lookup(degradation_stats)

    compound_by_lap = _compound_by_lap(starting_compound, stops, int(race_length))
    stop_map = {int(lap): str(compound) for lap, compound in stops}

    lap_trace: List[Dict[str, Any]] = []
    cumulative_time_s = 0.0
    effective_age = 1.0

    for lap_number in range(1, int(race_length) + 1):
        is_caution = _normalise_bool(caution_schedule.get(lap_number, False))
        compound = compound_by_lap[lap_number]
        degradation_rate = _lookup_degradation_rate(strategy, degradation_lookup, compound)

        if is_caution:
            lap_time_s = baseline_pace_s * float(caution_pace_ratio)
        else:
            lap_time_s = baseline_pace_s + (degradation_rate * effective_age)

        pit_loss_added = 0.0
        if lap_number in stop_map:
            pit_loss_added = float(caution_pit_loss_s if is_caution else pit_loss_s)
            lap_time_s += pit_loss_added

        cumulative_time_s += float(lap_time_s)
        if collect_trace:
            lap_trace.append(
                {
                    "lap_number": lap_number,
                    "compound": compound,
                    "is_caution": is_caution,
                    "effective_age": float(effective_age),
                    "degradation_rate": float(degradation_rate),
                    "baseline_pace_s": float(baseline_pace_s),
                    "pit_loss_s": float(pit_loss_added),
                    "lap_time_s": float(lap_time_s),
                    "cumulative_time_s": float(cumulative_time_s),
                }
            )

        if lap_number in stop_map:
            effective_age = 1.0
        else:
            effective_age += float(lambda_ if is_caution else 1.0)

    return {
        "race_time_s": float(cumulative_time_s),
        "lap_trace": lap_trace if collect_trace else [],
        "starting_compound": starting_compound,
        "stops": stops,
        "baseline_pace_s": float(baseline_pace_s),
    }


def _driver_median_green_pace(driver_laps: pd.DataFrame) -> float:
    laps = driver_laps.copy()
    if "fuel_corrected_laptime" not in laps.columns:
        raise ValueError("race laps must include fuel_corrected_laptime")
    candidates = []

    green = laps.copy()
    if "IsAccurate" in green.columns:
        green = green[green["IsAccurate"].eq(True)]
    if "sc_active" in green.columns:
        green = green[~green["sc_active"].fillna(False)]
    if "vsc_active" in green.columns:
        green = green[~green["vsc_active"].fillna(False)]
    if "PitInTime" in green.columns:
        green = green[green["PitInTime"].isna()]
    if "PitOutTime" in green.columns:
        green = green[green["PitOutTime"].isna()]
    candidates.append(pd.to_numeric(green["fuel_corrected_laptime"], errors="coerce").dropna())

    non_pit = laps.copy()
    if "IsAccurate" in non_pit.columns:
        non_pit = non_pit[non_pit["IsAccurate"].eq(True)]
    if "PitInTime" in non_pit.columns:
        non_pit = non_pit[non_pit["PitInTime"].isna()]
    if "PitOutTime" in non_pit.columns:
        non_pit = non_pit[non_pit["PitOutTime"].isna()]
    candidates.append(pd.to_numeric(non_pit["fuel_corrected_laptime"], errors="coerce").dropna())

    accurate_only = laps.copy()
    if "IsAccurate" in accurate_only.columns:
        accurate_only = accurate_only[accurate_only["IsAccurate"].eq(True)]
    candidates.append(pd.to_numeric(accurate_only["fuel_corrected_laptime"], errors="coerce").dropna())

    all_values = pd.to_numeric(laps["fuel_corrected_laptime"], errors="coerce").dropna()
    candidates.append(all_values)

    for values in candidates:
        if not values.empty:
            return float(values.median())
    raise ValueError("No fuel_corrected_laptime values available for baseline pace")


def _driver_pit_loss(driver_laps: pd.DataFrame) -> float:
    for column in ("pit_loss_s", "PitDuration_s", "PitTime"):
        if column not in driver_laps.columns:
            continue
        series = driver_laps.loc[:, column]
        if not isinstance(series, pd.Series):
            continue
        values = pd.to_numeric(series, errors="coerce")
        if isinstance(values, pd.Series):
            values = values[values.notna()]
            if not values.empty:
                return float(values.median())
        elif pd.notna(values):
            return float(values)
    return 0.0


def simulate_race_from_prepared(
    race_length: int,
    prepared_strategies: Mapping[str, Mapping[str, Any]],
    pit_loss_by_driver: Mapping[str, float],
    caution_schedule: Mapping[int, bool],
    degradation_lookup: Mapping,
    lambda_: float,
    collect_trace: bool = False,
) -> dict:
    """Low-overhead race simulation using only pre-computed inputs.

    All per-driver data (``baseline_pace_s``, compound sequence, ``pit_loss_s``,
    ``event_name``, ``year``) must already be resolved by the caller. This
    function calls ``simulate_car_race`` directly for every driver without any
    DataFrame access, making it suitable as the inner body of a Monte Carlo loop
    where these values are invariant across iterations.
    """
    driver_results: Dict[str, Dict[str, Any]] = {}
    for driver, strategy in prepared_strategies.items():
        driver_results[driver] = simulate_car_race(
            strategy=strategy,
            caution_schedule=caution_schedule,
            degradation_stats=None,
            race_length=race_length,
            lambda_=lambda_,
            pit_loss_s=float(pit_loss_by_driver[driver]),
            caution_pit_loss_s=CAUTION_PIT_LOSS_S,
            caution_pace_ratio=CAUTION_PACE_RATIO,
            degradation_lookup=degradation_lookup,
            collect_trace=collect_trace,
        )

    ranking = sorted(driver_results.items(), key=lambda item: (item[1]["race_time_s"], item[0]))
    finishing_order = [driver for driver, _ in ranking]
    position_by_driver = {driver: idx + 1 for idx, driver in enumerate(finishing_order)}
    points_by_driver = {driver: _points_for_position(position) for driver, position in position_by_driver.items()}

    return {
        "driver_results": driver_results,
        "finishing_order": finishing_order,
        "position_by_driver": position_by_driver,
        "points_by_driver": points_by_driver,
        "race_length": race_length,
    }


def simulate_race(race_state, strategies_by_driver, caution_schedule, lambda_) -> dict:
    """Simulate a whole race and rank drivers by cumulative time."""

    return simulate_race_with_driver_laps(race_state, strategies_by_driver, caution_schedule, lambda_)


def simulate_race_with_driver_laps(
    race_state,
    strategies_by_driver,
    caution_schedule,
    lambda_,
    driver_laps_by_driver: Optional[Mapping[str, pd.DataFrame]] = None,
    collect_trace: bool = True,
) -> dict:
    """Simulate a whole race and rank drivers by cumulative time.

    When ``driver_laps_by_driver`` is supplied, each driver's lap slice is reused
    directly instead of re-filtering the full race dataframe on every call. That
    keeps repeated Monte Carlo evaluations practical.
    """

    laps = race_state["laps"].copy()
    deg_stats = race_state.get("deg_stats")
    degradation_lookup = _build_degradation_lookup(deg_stats)

    required_drivers = sorted(laps["Driver"].dropna().astype(str).unique().tolist())
    missing = [driver for driver in required_drivers if driver not in strategies_by_driver]
    if missing:
        raise ValueError(f"Missing strategies for drivers: {', '.join(missing)}")

    race_length = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())
    event_name = str(laps["EventName"].dropna().iloc[0])
    year = int(pd.to_numeric(laps["Year"], errors="coerce").dropna().iloc[0])
    round_num = int(pd.to_numeric(laps["Round"], errors="coerce").dropna().iloc[0])
    race_baseline_pace = _driver_median_green_pace(laps)

    driver_results: Dict[str, Dict[str, Any]] = {}
    for driver in required_drivers:
        if driver_laps_by_driver is not None and driver in driver_laps_by_driver:
            driver_laps = driver_laps_by_driver[driver].copy()
        else:
            driver_laps = laps[laps["Driver"].astype(str) == driver].copy()
        strategy = dict(strategies_by_driver[driver])
        strategy.setdefault("event_name", event_name)
        strategy.setdefault("year", year)
        strategy.setdefault("round", round_num)
        if "baseline_pace_s" not in strategy or pd.isna(strategy["baseline_pace_s"]):
            try:
                strategy["baseline_pace_s"] = _driver_median_green_pace(driver_laps)
            except ValueError:
                strategy["baseline_pace_s"] = race_baseline_pace

        pit_loss_s = _driver_pit_loss(driver_laps)
        driver_results[driver] = simulate_car_race(
            strategy=strategy,
            caution_schedule=caution_schedule,
            degradation_stats=deg_stats,
            race_length=race_length,
            lambda_=lambda_,
            pit_loss_s=pit_loss_s,
            caution_pit_loss_s=CAUTION_PIT_LOSS_S,
            caution_pace_ratio=CAUTION_PACE_RATIO,
            degradation_lookup=degradation_lookup,
            collect_trace=collect_trace,
        )

    ranking = sorted(driver_results.items(), key=lambda item: (item[1]["race_time_s"], item[0]))
    finishing_order = [driver for driver, _ in ranking]
    position_by_driver = {driver: idx + 1 for idx, driver in enumerate(finishing_order)}
    points_by_driver = {driver: _points_for_position(position) for driver, position in position_by_driver.items()}

    return {
        "driver_results": driver_results,
        "finishing_order": finishing_order,
        "position_by_driver": position_by_driver,
        "points_by_driver": points_by_driver,
        "race_length": race_length,
        "event_name": event_name,
        "year": year,
        "round": round_num,
    }