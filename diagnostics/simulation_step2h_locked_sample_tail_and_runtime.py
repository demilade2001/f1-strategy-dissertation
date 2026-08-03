from itertools import combinations, product
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation.config import BASE_DF_PATH, LOCKED_ARCHETYPE_RACES
from source_code.simulation.config import is_valid_subject


WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
POINTS_TABLE = np.array([0, 25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=np.float32)

PREVIOUS_LINEAR_EXTRAPOLATION_10K = {
    ("Italian Grand Prix", 2024): 22.749745,
    ("Monaco Grand Prix", 2024): 71.107631,
}


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"})


def detect_stop_events(driver_laps: pd.DataFrame) -> List[Tuple[int, str]]:
    events: List[Tuple[int, str]] = []
    prev_compound: Optional[str] = None
    for _, row in driver_laps.sort_values("LapNumber").iterrows():
        compound = row["TyreCompound"]
        if pd.isna(compound):
            continue
        compound = str(compound)
        if prev_compound is None:
            prev_compound = compound
            continue
        if compound != prev_compound:
            events.append((int(row["LapNumber"]), compound))
        prev_compound = compound
    return events


def classify_contamination(driver_laps: pd.DataFrame, stop_laps: List[int]) -> Dict[str, object]:
    driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)
    lap_lookup = driver_laps.set_index("LapNumber")

    stop_on_caution_flags = []
    early_stop_flags = []
    for lap in stop_laps:
        if lap in lap_lookup.index:
            row = lap_lookup.loc[lap]
            stop_on_caution = bool(row.get("sc_active", False)) or bool(row.get("vsc_active", False))
        else:
            stop_on_caution = False
        stop_on_caution_flags.append(stop_on_caution)
        early_stop_flags.append(int(lap) <= 3)

    lap_time = pd.to_numeric(driver_laps.get("LapTime_s"), errors="coerce")
    pace_diff = pd.to_numeric(driver_laps.get("pace_differential"), errors="coerce").abs()

    if lap_time.notna().any():
        lap_q1, lap_q3 = lap_time.quantile([0.25, 0.75])
        lap_thr = lap_q3 + 1.5 * (lap_q3 - lap_q1)
    else:
        lap_thr = np.nan

    if pace_diff.notna().any():
        pace_q1, pace_q3 = pace_diff.quantile([0.25, 0.75])
        pace_thr = pace_q3 + 1.5 * (pace_q3 - pace_q1)
    else:
        pace_thr = np.nan

    preceded_by_outlier_flags = []
    outlier_prev_laps = []
    for lap in stop_laps:
        prev_lap = int(lap) - 1
        if prev_lap not in lap_lookup.index:
            preceded_by_outlier_flags.append(False)
            continue
        row = lap_lookup.loc[prev_lap]
        lap_val = pd.to_numeric(row.get("LapTime_s"), errors="coerce")
        pace_val = abs(pd.to_numeric(row.get("pace_differential"), errors="coerce"))
        lap_outlier = bool(pd.notna(lap_thr) and pd.notna(lap_val) and lap_val > lap_thr)
        pace_outlier = bool(pd.notna(pace_thr) and pd.notna(pace_val) and pace_val > pace_thr)
        flag = lap_outlier or pace_outlier
        preceded_by_outlier_flags.append(flag)
        if flag:
            outlier_prev_laps.append(prev_lap)

    stop_on_caution_any = bool(any(stop_on_caution_flags))
    early_stop_any = bool(any(early_stop_flags))
    preceding_outlier_any = bool(any(preceded_by_outlier_flags))

    classification = (
        "reactive"
        if (stop_on_caution_any or early_stop_any or preceding_outlier_any)
        else "strategic_like"
    )

    return {
        "stop_on_caution_any": stop_on_caution_any,
        "stop_on_caution_laps": [lap for lap, flag in zip(stop_laps, stop_on_caution_flags) if flag],
        "early_stop_any": early_stop_any,
        "early_stop_laps": [lap for lap, flag in zip(stop_laps, early_stop_flags) if flag],
        "preceding_outlier_any": preceding_outlier_any,
        "preceding_outlier_prev_laps": outlier_prev_laps,
        "classification": classification,
    }


def pick_reference_driver(laps: pd.DataFrame) -> Tuple[str, str]:
    final_rows = (
        laps.sort_values(["Driver", "LapNumber"])
        .groupby("Driver", as_index=False)
        .tail(1)
        .copy()
    )
    final_rows = final_rows[final_rows["Team"].apply(is_valid_subject)].copy()
    if final_rows.empty:
        raise ValueError("No midfield driver found for race")
    final_rows = final_rows.sort_values(["Position", "LapNumber", "Driver"])
    row = final_rows.iloc[0]
    return str(row["Driver"]), str(row["Team"])


def enumerate_strategy_features(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    grid_spacing: int,
    max_stops: int,
):
    candidate_laps = np.arange(1, race_length, grid_spacing, dtype=np.int16)
    records = []

    for stop_count in range(1, max_stops + 1):
        for pit_laps in combinations(candidate_laps.tolist(), stop_count):
            for compounds in product(DRY_COMPOUNDS, repeat=stop_count):
                if (not is_wet_race) and (starting_compound in DRY_COMPOUNDS):
                    used = {starting_compound, *compounds}
                    if len(used) < 2:
                        continue
                records.append((pit_laps, compounds))

    n = len(records)
    stop_laps_arr = np.full((n, max_stops), -1, dtype=np.int16)
    stop_counts = np.zeros(n, dtype=np.int8)
    hard_count = np.zeros(n, dtype=np.int8)
    soft_count = np.zeros(n, dtype=np.int8)

    for idx, (pit_laps, compounds) in enumerate(records):
        k = len(pit_laps)
        stop_counts[idx] = k
        stop_laps_arr[idx, :k] = np.array(pit_laps, dtype=np.int16)
        hard_count[idx] = sum(1 for c in compounds if c == "HARD")
        soft_count[idx] = sum(1 for c in compounds if c == "SOFT")

    return {
        "strategy_count": n,
        "stop_laps_arr": stop_laps_arr,
        "stop_counts": stop_counts.astype(np.float32),
        "hard_count": hard_count.astype(np.float32),
        "soft_count": soft_count.astype(np.float32),
    }


def run_full_vectorized_benchmark(
    race_length: int,
    strategy_features: Dict[str, np.ndarray],
    schedule_count: int,
) -> Dict[str, float]:
    n = int(strategy_features["strategy_count"])
    stop_laps_arr = strategy_features["stop_laps_arr"]
    stop_counts = strategy_features["stop_counts"]
    hard_count = strategy_features["hard_count"]
    soft_count = strategy_features["soft_count"]

    rng = np.random.default_rng(42)
    caution_schedules = (rng.random((schedule_count, race_length)) < 0.08).astype(np.int8)

    stop_mask = np.zeros((n, race_length), dtype=np.int8)
    idx = np.arange(n)
    for col in range(stop_laps_arr.shape[1]):
        laps = stop_laps_arr[:, col]
        valid = laps >= 1
        if valid.any():
            stop_mask[idx[valid], laps[valid] - 1] = 1

    base_time = (
        race_length * 90.0
        + stop_counts * 22.0
        - hard_count * 0.4
        + soft_count * 0.2
    ).astype(np.float32)

    batch_size = 250
    total_points = np.zeros(n, dtype=np.float64)
    total_rank = np.zeros(n, dtype=np.float64)

    start = time.perf_counter()
    for start_col in range(0, schedule_count, batch_size):
        end_col = min(start_col + batch_size, schedule_count)
        sched_batch = caution_schedules[start_col:end_col, :]  # B x L
        b = sched_batch.shape[0]

        caution_hits = stop_mask @ sched_batch.T  # N x B
        stochastic = rng.normal(loc=0.0, scale=0.35, size=(n, b)).astype(np.float32)
        total_time = base_time[:, None] - caution_hits.astype(np.float32) * 7.5 + stochastic

        order = np.argsort(total_time, axis=0)
        ranks = np.empty_like(order, dtype=np.int32)
        ranks[order, np.arange(b)] = np.arange(1, n + 1, dtype=np.int32)[:, None]

        total_rank += ranks.sum(axis=1, dtype=np.float64)

        capped = np.where(ranks <= 10, ranks, 0)
        points = POINTS_TABLE[capped]
        total_points += points.sum(axis=1, dtype=np.float64)

    elapsed = time.perf_counter() - start

    mean_points = total_points / schedule_count
    mean_rank = total_rank / schedule_count
    checksum = float(mean_points.mean() + mean_rank.mean())

    return {
        "wall_clock_seconds": elapsed,
        "vectorized_checksum": checksum,
    }


def main():
    base_df = pd.read_csv(BASE_DF_PATH)
    base_df = base_df.copy()
    base_df["is_midfield"] = to_bool(base_df["is_midfield"]) if "is_midfield" in base_df.columns else False
    base_df["sc_active"] = to_bool(base_df["sc_active"]) if "sc_active" in base_df.columns else False
    base_df["vsc_active"] = to_bool(base_df["vsc_active"]) if "vsc_active" in base_df.columns else False

    print("Step 1 - Locked sample from config")
    lock_df = pd.DataFrame(LOCKED_ARCHETYPE_RACES)
    print(lock_df[["archetype", "event_name", "year", "round"]].to_string(index=False))

    print("\nStep 2 - Stop counts recomputed directly for exact locked races")
    stop_rows = []
    for row in LOCKED_ARCHETYPE_RACES:
        event_name = row["event_name"]
        year = int(row["year"])
        round_num = int(row["round"])
        archetype = row["archetype"]

        race_df = base_df[
            (base_df["EventName"] == event_name)
            & (base_df["Year"] == year)
            & (base_df["Round"] == round_num)
            & (base_df["is_midfield"] == True)
        ].copy()

        for (driver, team), g in race_df.groupby(["Driver", "Team"], sort=True):
            stops = detect_stop_events(g)
            stop_rows.append(
                {
                    "archetype": archetype,
                    "event_name": event_name,
                    "year": year,
                    "round": round_num,
                    "driver": str(driver),
                    "team": str(team),
                    "stop_count": int(len(stops)),
                    "stop_laps": [lap for lap, _ in stops],
                }
            )

    stop_df = pd.DataFrame(stop_rows).sort_values(["archetype", "event_name", "driver"]).reset_index(drop=True)
    print(stop_df[["archetype", "event_name", "year", "round", "driver", "team", "stop_count"]].to_string(index=False))

    print("\nStep 3 - Locked-sample 4+ stop cases")
    flagged = stop_df[stop_df["stop_count"] >= 4].copy().reset_index(drop=True)
    print(f"flagged_4plus_count={len(flagged)}")
    if flagged.empty:
        print("No 4+ stop cases in the locked sample.")
    else:
        print(flagged[["archetype", "event_name", "year", "round", "driver", "team", "stop_count"]].to_string(index=False))

    print("\nStep 4 + Step 5 - Contamination flags and classification")
    if flagged.empty:
        print("Skipped because there are zero flagged 4+ stop cases.")
    else:
        contam_rows = []
        for row in flagged.itertuples(index=False):
            race_driver_df = base_df[
                (base_df["EventName"] == row.event_name)
                & (base_df["Year"] == int(row.year))
                & (base_df["Round"] == int(row.round))
                & (base_df["Driver"] == row.driver)
            ].copy()
            info = classify_contamination(race_driver_df, list(row.stop_laps))
            contam_rows.append(
                {
                    "archetype": row.archetype,
                    "event_name": row.event_name,
                    "year": int(row.year),
                    "round": int(row.round),
                    "driver": row.driver,
                    "team": row.team,
                    "stop_count": int(row.stop_count),
                    "stop_on_caution_any": info["stop_on_caution_any"],
                    "stop_on_caution_laps": info["stop_on_caution_laps"],
                    "early_stop_any": info["early_stop_any"],
                    "early_stop_laps": info["early_stop_laps"],
                    "preceding_outlier_any": info["preceding_outlier_any"],
                    "preceding_outlier_prev_laps": info["preceding_outlier_prev_laps"],
                    "classification": info["classification"],
                }
            )

        contam_df = pd.DataFrame(contam_rows)
        print(contam_df.to_string(index=False))

    print("\nStep 6 - Real 10,000-schedule runtime benchmark on longest locked race")
    race_lengths = []
    for row in LOCKED_ARCHETYPE_RACES:
        event_name = row["event_name"]
        year = int(row["year"])
        round_num = int(row["round"])
        race_df = base_df[
            (base_df["EventName"] == event_name)
            & (base_df["Year"] == year)
            & (base_df["Round"] == round_num)
        ].copy()
        race_len = int(pd.to_numeric(race_df["LapNumber"], errors="coerce").max())
        race_lengths.append({"event_name": event_name, "year": year, "round": round_num, "race_length": race_len})

    race_len_df = pd.DataFrame(race_lengths).sort_values(["race_length", "event_name", "year"], ascending=[False, True, True]).reset_index(drop=True)
    print("locked_race_lengths:")
    print(race_len_df.to_string(index=False))

    longest = race_len_df.iloc[0]
    longest_event = str(longest["event_name"])
    longest_year = int(longest["year"])
    longest_round = int(longest["round"])
    longest_len = int(longest["race_length"])

    longest_df = base_df[
        (base_df["EventName"] == longest_event)
        & (base_df["Year"] == longest_year)
        & (base_df["Round"] == longest_round)
    ].copy()

    selected_driver, selected_team = pick_reference_driver(longest_df)
    driver_laps = longest_df[longest_df["Driver"] == selected_driver].copy().sort_values("LapNumber")
    starting_compound = str(driver_laps.loc[driver_laps["LapNumber"] == 1, "TyreCompound"].iloc[0])
    is_wet_race = bool(longest_df["TyreCompound"].isin(WET_COMPOUNDS).any())

    strat = enumerate_strategy_features(
        race_length=longest_len,
        starting_compound=starting_compound,
        is_wet_race=is_wet_race,
        grid_spacing=3,
        max_stops=3,
    )

    bench = run_full_vectorized_benchmark(
        race_length=longest_len,
        strategy_features=strat,
        schedule_count=10_000,
    )

    print(f"benchmark_race={longest_event}")
    print(f"benchmark_year={longest_year}")
    print(f"benchmark_round={longest_round}")
    print(f"benchmark_driver={selected_driver}")
    print(f"benchmark_team={selected_team}")
    print(f"benchmark_race_length={longest_len}")
    print(f"benchmark_strategy_count={int(strat['strategy_count'])}")
    print(f"benchmark_schedule_count=10000")
    print(f"benchmark_wall_clock_seconds={bench['wall_clock_seconds']:.6f}")
    print(f"benchmark_vectorized_checksum={bench['vectorized_checksum']:.6f}")

    key = (longest_event, longest_year)
    if key in PREVIOUS_LINEAR_EXTRAPOLATION_10K:
        prev = PREVIOUS_LINEAR_EXTRAPOLATION_10K[key]
        delta = bench["wall_clock_seconds"] - prev
        print(f"previous_linear_extrapolation_10k_seconds={prev:.6f}")
        print(f"comparison_delta_seconds_real_minus_extrapolated={delta:.6f}")
    else:
        print("previous_linear_extrapolation_note=No prior linear extrapolation for this exact race-year among Monaco/Monza/Hungary 2024; this is a fresh measurement.")


if __name__ == "__main__":
    main()
