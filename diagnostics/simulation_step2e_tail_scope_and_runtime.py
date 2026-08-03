from itertools import combinations, product
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation.config import BASE_DF_PATH, is_valid_subject
from source_code.simulation.race_state import load_race_state


ARCHETYPE_ORDER = [
    ("Street", "Monaco Grand Prix", [r"\bmonaco\b"]),
    ("Street", "Azerbaijan Grand Prix", [r"\bazerbaijan\b", r"\bbaku\b"]),
    ("Street", "Singapore Grand Prix", [r"\bsingapore\b"]),
    ("Street", "Miami Grand Prix", [r"\bmiami\b"]),
    ("Power", "Italian Grand Prix", [r"\bitalian\b", r"\bmonza\b"]),
    ("Power", "Belgian Grand Prix", [r"\bbelgian\b", r"\bspa\b"]),
    ("Power", "British Grand Prix", [r"\bbritish\b", r"\bsilverstone\b"]),
    ("Power", "Bahrain Grand Prix", [r"\bbahrain\b"]),
    ("Technical", "Spanish Grand Prix", [r"\bspanish\b", r"\bbarcelona\b"]),
    ("Technical", "Hungarian Grand Prix", [r"\bhungarian\b"]),
    ("Technical", "Abu Dhabi Grand Prix", [r"\babu\s+dhabi\b"]),
    ("Technical", "Japanese Grand Prix", [r"\bjapanese\b", r"\bsuzuka\b"]),
]

PHASE3_REFERENCE_FILES = [
    ROOT / "diagnostics" / "phase3_blockers.py",
    ROOT / "diagnostics" / "chapter3_degradation_regeneration.py",
]

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
POINTS_TABLE = np.array([0, 25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=np.float32)


def to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes"})


def normalize_event_name(event_name: str) -> Optional[str]:
    e = str(event_name).lower()
    for _, canonical, patterns in ARCHETYPE_ORDER:
        if any(re.search(p, e) for p in patterns):
            return canonical
    return None


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


def extract_strategy_features(race_length: int, starting_compound: str, is_wet_race: bool, spacing: int, max_stops: int):
    candidate_laps = np.arange(1, race_length, spacing, dtype=np.int16)
    records = []
    for stop_count in range(1, max_stops + 1):
        for pit_laps in combinations(candidate_laps.tolist(), stop_count):
            for compounds in product(DRY_COMPOUNDS, repeat=stop_count):
                if (not is_wet_race) and starting_compound in DRY_COMPOUNDS:
                    used = {starting_compound, *compounds}
                    if len(used) < 2:
                        continue
                records.append((pit_laps, compounds))

    n = len(records)
    max_cols = max_stops
    stop_laps_arr = np.full((n, max_cols), -1, dtype=np.int16)
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
        "count": n,
        "stop_laps_arr": stop_laps_arr,
        "stop_counts": stop_counts,
        "hard_count": hard_count,
        "soft_count": soft_count,
        "candidate_laps": candidate_laps,
    }


def pick_reference_driver(laps: pd.DataFrame) -> Tuple[str, str]:
    final_rows = (
        laps.sort_values(["Driver", "LapNumber"])
        .groupby("Driver", as_index=False)
        .tail(1)
        .copy()
    )
    final_rows = final_rows[final_rows["Team"].apply(is_valid_subject)].copy()
    final_rows = final_rows.sort_values(["Position", "LapNumber", "Driver"])
    row = final_rows.iloc[0]
    return str(row["Driver"]), str(row["Team"])


def determine_wet_race(race_laps: pd.DataFrame) -> bool:
    wet_cols = [c for c in race_laps.columns if "wet" in c.lower() and "compound" not in c.lower()]
    if wet_cols:
        col = wet_cols[0]
        non_null = race_laps[col].dropna()
        if not non_null.empty:
            return bool(non_null.iloc[0])
    return bool(race_laps["TyreCompound"].isin(WET_COMPOUNDS).any())


def detect_locked_year_round_mapping(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for archetype, circuit, _ in ARCHETYPE_ORDER:
        sub = df[df["EventName"] == circuit]
        available = sorted({(int(y), int(r)) for y, r in zip(sub["Year"], sub["Round"])})
        available_text = ", ".join(f"{y}-R{r}" for y, r in available)
        rows.append(
            {
                "archetype": archetype,
                "circuit": circuit,
                "locked_year_round": "UNLOCKED",
                "available_year_rounds_in_base_df": available_text,
                "lock_source": "diagnostics/phase3_blockers.py + diagnostics/chapter3_degradation_regeneration.py define event-name sets only",
            }
        )
    return pd.DataFrame(rows)


def compute_driver_race_stop_counts(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["Year", "Round", "EventName", "Driver", "Team"]
    rows = []
    for key, g in df.groupby(keys, sort=False):
        year, round_num, event, driver, team = key
        stops = detect_stop_events(g)
        rows.append(
            {
                "Year": int(year),
                "Round": int(round_num),
                "EventName": str(event),
                "canonical_event": normalize_event_name(str(event)),
                "Driver": str(driver),
                "Team": str(team),
                "stop_count": int(len(stops)),
                "stop_laps": [int(l) for l, _ in stops],
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["Year", "Round", "EventName", "Driver"]).reset_index(drop=True)


def classify_contamination(driver_laps: pd.DataFrame, stop_laps: List[int]) -> Dict[str, object]:
    driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)
    lap_lookup = driver_laps.set_index("LapNumber")
    extra_stop_laps = stop_laps[3:] if len(stop_laps) >= 4 else []

    stop_on_caution = []
    early_stop = []
    for lap in stop_laps:
        if lap in lap_lookup.index:
            row = lap_lookup.loc[lap]
            stop_on_caution.append(bool(row.get("sc_active", False)) or bool(row.get("vsc_active", False)))
        else:
            stop_on_caution.append(False)
        early_stop.append(int(lap) <= 3)

    lap_time = pd.to_numeric(driver_laps.get("LapTime_s"), errors="coerce")
    pace_diff = pd.to_numeric(driver_laps.get("pace_differential"), errors="coerce")
    lap_time_q1, lap_time_q3 = lap_time.quantile([0.25, 0.75]) if lap_time.notna().any() else (np.nan, np.nan)
    pace_q1, pace_q3 = pace_diff.abs().quantile([0.25, 0.75]) if pace_diff.notna().any() else (np.nan, np.nan)
    lap_time_thr = lap_time_q3 + 1.5 * (lap_time_q3 - lap_time_q1) if pd.notna(lap_time_q3) and pd.notna(lap_time_q1) else np.nan
    pace_thr = pace_q3 + 1.5 * (pace_q3 - pace_q1) if pd.notna(pace_q3) and pd.notna(pace_q1) else np.nan

    extra_preceding_flags = []
    extra_preceding_metrics = []
    for lap in extra_stop_laps:
        prev_lap = int(lap) - 1
        if prev_lap not in lap_lookup.index:
            extra_preceding_flags.append(False)
            extra_preceding_metrics.append(f"lap{prev_lap}:missing")
            continue
        row = lap_lookup.loc[prev_lap]
        lap_val = pd.to_numeric(row.get("LapTime_s"), errors="coerce")
        pace_val = abs(pd.to_numeric(row.get("pace_differential"), errors="coerce"))
        lap_flag = bool(pd.notna(lap_time_thr) and pd.notna(lap_val) and lap_val > lap_time_thr)
        pace_flag = bool(pd.notna(pace_thr) and pd.notna(pace_val) and pace_val > pace_thr)
        extra_preceding_flags.append(lap_flag or pace_flag)
        lap_val_txt = f"{float(lap_val):.3f}" if pd.notna(lap_val) else "nan"
        pace_val_txt = f"{float(pace_val):.3f}" if pd.notna(pace_val) else "nan"
        lap_thr_txt = f"{float(lap_time_thr):.3f}" if pd.notna(lap_time_thr) else "nan"
        pace_thr_txt = f"{float(pace_thr):.3f}" if pd.notna(pace_thr) else "nan"
        extra_preceding_metrics.append(
            f"lap{prev_lap}:LapTime_s={lap_val_txt},"
            f"|pace_differential|={pace_val_txt},"
            f"lap_thr={lap_thr_txt},"
            f"pace_thr={pace_thr_txt}"
        )

    reactive = any(stop_on_caution) or any(early_stop) or any(extra_preceding_flags)
    return {
        "stop_on_caution_any": bool(any(stop_on_caution)),
        "early_stop_any": bool(any(early_stop)),
        "extra_stop_preceded_by_outlier_any": bool(any(extra_preceding_flags)),
        "stop_on_caution_laps": [lap for lap, f in zip(stop_laps, stop_on_caution) if f],
        "early_stop_laps": [lap for lap, f in zip(stop_laps, early_stop) if f],
        "extra_stop_laps": extra_stop_laps,
        "classification": "reactive" if reactive else "strategic_like",
    }


def benchmark_vectorized_monza() -> Dict[str, object]:
    state = load_race_state(year=2024, round_num=16)
    race_laps = state["laps"]
    driver, team = pick_reference_driver(race_laps)
    driver_laps = race_laps[race_laps["Driver"] == driver].sort_values("LapNumber").reset_index(drop=True)
    race_length = int(driver_laps["LapNumber"].max())
    starting_compound = str(driver_laps.loc[driver_laps["LapNumber"] == 1, "TyreCompound"].iloc[0])
    is_wet_race = determine_wet_race(race_laps)

    features = extract_strategy_features(
        race_length=race_length,
        starting_compound=starting_compound,
        is_wet_race=is_wet_race,
        spacing=3,
        max_stops=3,
    )
    n_strategies = int(features["count"])
    n_schedules = 1000
    rng = np.random.default_rng(42)

    # Pre-generate caution schedules (1 means caution-active lap) before timing.
    caution_schedules = (rng.random((n_schedules, race_length)) < 0.08).astype(np.int8)

    stop_laps_arr = features["stop_laps_arr"]
    stop_counts = features["stop_counts"].astype(np.float32)
    hard_count = features["hard_count"].astype(np.float32)
    soft_count = features["soft_count"].astype(np.float32)
    n = n_strategies
    s = n_schedules

    # Build N x L stop mask with fixed small column loop (max 3 stops) and no N*S Python loop.
    stop_mask = np.zeros((n, race_length), dtype=np.int8)
    strategy_idx = np.arange(n)
    for col in range(stop_laps_arr.shape[1]):
        laps = stop_laps_arr[:, col]
        valid = laps >= 1
        if valid.any():
            stop_mask[strategy_idx[valid], laps[valid] - 1] = 1

    start = time.perf_counter()
    caution_hits = stop_mask @ caution_schedules.T
    stochastic = rng.normal(loc=0.0, scale=0.35, size=(n, s)).astype(np.float32)

    # Synthetic but vectorized race-time model for benchmark purposes.
    base_time = (
        race_length * 90.0
        + stop_counts * 22.0
        - hard_count * 0.4
        + soft_count * 0.2
    ).astype(np.float32)
    total_time = base_time[:, None] - caution_hits.astype(np.float32) * 7.5 + stochastic

    order = np.argsort(total_time, axis=0)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order, np.arange(s)] = np.arange(1, n + 1, dtype=np.int32)[:, None]
    capped = np.where(ranks <= 10, ranks, 0)
    points = POINTS_TABLE[capped]
    mean_points = points.mean(axis=1)
    mean_rank = ranks.mean(axis=1)
    elapsed = time.perf_counter() - start

    # Keep outputs used so benchmark cannot be optimized away.
    checksum = float(mean_points.mean() + mean_rank.mean())

    linear_10k_same_race = elapsed * (10_000 / n_schedules)
    monaco_max3_strategy_count = 70_252
    linear_monaco_1k = elapsed * (monaco_max3_strategy_count / n_strategies)
    linear_monaco_10k = linear_monaco_1k * (10_000 / n_schedules)

    return {
        "selected_driver": driver,
        "selected_team": team,
        "race_length": race_length,
        "grid_spacing": 3,
        "max_stops": 3,
        "strategy_count": n_strategies,
        "schedule_count": n_schedules,
        "evaluation_seconds": elapsed,
        "linear_projection_10k_same_race_seconds": linear_10k_same_race,
        "linear_projection_monaco_1k_seconds": linear_monaco_1k,
        "linear_projection_monaco_10k_seconds": linear_monaco_10k,
        "projection_assumption": "Linear in schedules and strategy count",
        "vectorized_checksum": checksum,
    }


def main():
    df = pd.read_csv(BASE_DF_PATH)
    df = df[df["Year"].isin([2022, 2023, 2024])].copy()
    df = df[to_bool(df["is_midfield"])].copy()
    df["sc_active"] = to_bool(df["sc_active"]).astype(bool)
    df["vsc_active"] = to_bool(df["vsc_active"]).astype(bool)
    df = df.sort_values(["Year", "Round", "EventName", "Driver", "LapNumber"]).reset_index(drop=True)

    print("Part A - Locked 12-race sample scope check")
    print("phase3_reference_files_checked:")
    for p in PHASE3_REFERENCE_FILES:
        print(f"  - {p.relative_to(ROOT).as_posix()}")

    lock_df = detect_locked_year_round_mapping(df)
    print("locked_sample_year_round_table:")
    print(lock_df.to_string(index=False))
    print("locked_sample_note=No per-circuit Year/Round lock was found; prior diagnostics define event-name sets across seasons.")

    stop_df = compute_driver_race_stop_counts(df)
    four_plus = stop_df[stop_df["stop_count"] >= 4].copy()
    print(f"\nall_midfield_driver_races_with_stop_count_ge_4={len(four_plus)}")
    if not four_plus.empty:
        cols = ["Year", "Round", "EventName", "Driver", "Team", "stop_count"]
        print(four_plus[cols].to_string(index=False))

    locked_matches = pd.DataFrame(columns=four_plus.columns)
    print(f"\nlocked_sample_match_count={len(locked_matches)}")
    print("locked_sample_match_note=0 because no per-circuit locked Year/Round mapping exists in repository history checked.")

    event_name_matches = four_plus[four_plus["canonical_event"].notna()].copy()
    print(f"event_name_scope_match_count_if_all_years_used={len(event_name_matches)}")

    print("\nPart B - Non-strategic contamination check for locked-sample matches")
    if locked_matches.empty:
        print("part_b_note=No locked-sample matches to assess because lock mapping is undefined.")
    else:
        contamination_rows = []
        for row in locked_matches.itertuples(index=False):
            g = df[
                (df["Year"] == row.Year)
                & (df["Round"] == row.Round)
                & (df["Driver"] == row.Driver)
            ].copy()
            info = classify_contamination(g, list(row.stop_laps))
            contamination_rows.append(
                {
                    "Year": row.Year,
                    "Round": row.Round,
                    "EventName": row.EventName,
                    "Driver": row.Driver,
                    "Team": row.Team,
                    "stop_count": row.stop_count,
                    **info,
                }
            )
        print(pd.DataFrame(contamination_rows).to_string(index=False))

    if not event_name_matches.empty:
        print("\npart_b_supplementary_event_name_scope_assessment:")
        contamination_rows = []
        for row in event_name_matches.itertuples(index=False):
            g = df[
                (df["Year"] == row.Year)
                & (df["Round"] == row.Round)
                & (df["Driver"] == row.Driver)
            ].copy()
            info = classify_contamination(g, list(row.stop_laps))
            contamination_rows.append(
                {
                    "Year": row.Year,
                    "Round": row.Round,
                    "EventName": row.EventName,
                    "Driver": row.Driver,
                    "Team": row.Team,
                    "stop_count": row.stop_count,
                    "classification": info["classification"],
                    "stop_on_caution_any": info["stop_on_caution_any"],
                    "early_stop_any": info["early_stop_any"],
                    "extra_stop_preceded_by_outlier_any": info["extra_stop_preceded_by_outlier_any"],
                    "stop_on_caution_laps": info["stop_on_caution_laps"],
                    "early_stop_laps": info["early_stop_laps"],
                    "extra_stop_laps": info["extra_stop_laps"],
                }
            )
        print(pd.DataFrame(contamination_rows).sort_values(["Year", "Round", "Driver"]).to_string(index=False))

    print("\nPart C - Real runtime benchmark (vectorized evaluation)")
    bench = benchmark_vectorized_monza()
    for k, v in bench.items():
        if isinstance(v, float):
            print(f"{k}={v:.6f}")
        else:
            print(f"{k}={v}")


if __name__ == "__main__":
    main()