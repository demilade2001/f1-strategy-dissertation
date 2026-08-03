from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import BASE_DF_PATH, DEG_RATE_STATS_PATH, XGB_TEST_PREDICTIONS_PATH


XGB_PROBABILITY_COLUMN = "y_pred_xgb_classweight_cal"
XGB_OOF_PREDICTIONS_PATH = Path(XGB_TEST_PREDICTIONS_PATH).parent / "classweight_oof_predictions_2022_2023.csv"


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

    if year <= 2023:
        pred_source_label = "out_of_fold_2022_2023"
        pred_path = XGB_OOF_PREDICTIONS_PATH
    elif year == 2024:
        pred_source_label = "heldout_2024"
        pred_path = XGB_TEST_PREDICTIONS_PATH
    else:
        pred_source_label = "none"
        pred_path = None

    merge_keys = ["Year", "Round", "Driver", "LapNumber"]
    if pred_path is None:
        pred_subset = pd.DataFrame(columns=merge_keys + [XGB_PROBABILITY_COLUMN])
    else:
        if not Path(pred_path).exists():
            raise FileNotFoundError(f"Prediction source file not found: {pred_path}")
        pred_df = pd.read_csv(pred_path)
        pred_subset = pred_df[(pred_df["Year"] == year) & (pred_df["Round"] == round_num)].copy()

    pred_value_cols = [col for col in pred_subset.columns if col not in merge_keys]

    race_laps = race_laps.merge(
        pred_subset[merge_keys + pred_value_cols],
        on=merge_keys,
        how="left",
    )

    if XGB_PROBABILITY_COLUMN not in race_laps.columns:
        race_laps[XGB_PROBABILITY_COLUMN] = pd.NA

    race_laps["prediction_source"] = "none"
    has_prob_mask = race_laps[XGB_PROBABILITY_COLUMN].notna()
    if pred_source_label != "none":
        race_laps.loc[has_prob_mask, "prediction_source"] = pred_source_label

    # Rows without lap-level probability (typically each driver's final 1-3 laps from sc_vsc_next3
    # lookahead truncation) stay in the race timeline and fall back to circuit_sc_rate prior when
    # sc_sampler.py consumes this state, aligned with the same prior used for R_SC.
    race_laps["has_lap_level_prob"] = has_prob_mask.astype(bool)

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