from typing import Dict, List

import pandas as pd

from .config import BASE_DF_PATH, DEG_RATE_STATS_PATH, XGB_TEST_PREDICTIONS_PATH


def _get_single_rate(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return None
    return float(non_null.iloc[0])


def load_race_state(year: int, round_num: int) -> dict:
    base_df = pd.read_csv(BASE_DF_PATH)
    race_laps = base_df[(base_df["Year"] == year) & (base_df["Round"] == round_num)].copy()

    if race_laps.empty:
        raise ValueError(f"No race rows found for Year={year}, Round={round_num} in {BASE_DF_PATH}")

    race_laps["is_strategic_stop"] = race_laps["is_strategic_stop"].astype(pd.BooleanDtype())
    race_laps["sc_active"] = race_laps["sc_active"].astype(bool)
    race_laps["vsc_active"] = race_laps["vsc_active"].astype(bool)

    event_name = race_laps["EventName"].dropna().iloc[0]

    deg_df = pd.read_csv(DEG_RATE_STATS_PATH)
    deg_stats_subset = deg_df[(deg_df["Year"] == year) & (deg_df["EventName"] == event_name)].copy()

    pred_df = pd.read_csv(XGB_TEST_PREDICTIONS_PATH)
    pred_subset = pred_df[(pred_df["Year"] == year) & (pred_df["Round"] == round_num)].copy()

    merge_keys = ["Year", "Round", "Driver", "LapNumber"]
    merged_predictions = race_laps[merge_keys].merge(
        pred_subset,
        on=merge_keys,
        how="left",
        indicator=True,
    )

    circuit_sc_rate = _get_single_rate(race_laps["circuit_sc_rate"])
    circuit_vsc_rate = _get_single_rate(race_laps["circuit_vsc_rate"])

    drivers_with_teams_df = (
        race_laps[["Driver", "Team"]]
        .dropna(subset=["Driver"])
        .drop_duplicates()
        .sort_values(["Driver", "Team"])
    )
    drivers_with_teams: List[Dict[str, str]] = drivers_with_teams_df.to_dict(orient="records")

    return {
        "laps": race_laps,
        "deg_stats": deg_stats_subset,
        "predictions_merged": merged_predictions,
        "circuit_sc_rate": circuit_sc_rate,
        "circuit_vsc_rate": circuit_vsc_rate,
        "drivers_with_teams": drivers_with_teams,
    }