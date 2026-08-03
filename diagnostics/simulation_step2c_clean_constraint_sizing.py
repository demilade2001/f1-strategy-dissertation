from math import comb
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation.config import BASE_DF_PATH, is_valid_subject
from source_code.simulation.race_state import load_race_state
from source_code.simulation.strategy import is_strategy_feasible


RACES = [
    ("Monaco 2024", 2024, 8),
    ("Monza 2024", 2024, 16),
    ("Hungary 2024", 2024, 13),
]

ORIGINAL_UNCONSTRAINED_SIZES = {
    "Monaco 2024": 110_187_462,
    "Monza 2024": 22_243_312,
    "Hungary 2024": 70_541_230,
}

WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


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


def build_stints(driver_laps: pd.DataFrame) -> List[Dict]:
    driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)
    race_length = int(driver_laps["LapNumber"].max())
    stop_events = detect_stop_events(driver_laps)
    stop_laps = [lap for lap, _ in stop_events]
    boundaries = [1] + stop_laps + [race_length + 1]

    sc_map = driver_laps.set_index("LapNumber")["sc_active"] if "sc_active" in driver_laps.columns else pd.Series(dtype=bool)
    vsc_map = driver_laps.set_index("LapNumber")["vsc_active"] if "vsc_active" in driver_laps.columns else pd.Series(dtype=bool)

    stints: List[Dict] = []
    for idx in range(len(boundaries) - 1):
        start_lap = int(boundaries[idx])
        next_boundary = int(boundaries[idx + 1])
        length = int(next_boundary - start_lap)
        ended_by_stop = next_boundary <= race_length
        ending_stop_lap = int(next_boundary) if ended_by_stop else None

        sc_or_vsc_at_ending_stop = False
        if ending_stop_lap is not None:
            sc_val = sc_map.get(ending_stop_lap, False)
            vsc_val = vsc_map.get(ending_stop_lap, False)
            sc_or_vsc_at_ending_stop = bool(sc_val) or bool(vsc_val)

        stints.append(
            {
                "start_lap": start_lap,
                "end_boundary": next_boundary,
                "length": length,
                "ended_by_stop": ended_by_stop,
                "ending_stop_lap": ending_stop_lap,
                "sc_or_vsc_at_ending_stop": sc_or_vsc_at_ending_stop,
            }
        )
    return stints


def extract_actual_strategy(driver_laps: pd.DataFrame) -> Dict:
    lap1 = driver_laps[driver_laps["LapNumber"] == 1]
    if lap1.empty:
        raise ValueError("Driver has no LapNumber == 1 row; cannot determine starting compound")
    starting_compound = str(lap1.iloc[0]["TyreCompound"])
    stops = detect_stop_events(driver_laps)
    return {
        "starting_compound": starting_compound,
        "stops": stops,
    }


def determine_wet_race(race_laps: pd.DataFrame) -> bool:
    wet_cols = [c for c in race_laps.columns if "wet" in c.lower() and "compound" not in c.lower()]
    if wet_cols:
        col = wet_cols[0]
        non_null = race_laps[col].dropna()
        if not non_null.empty:
            return bool(non_null.iloc[0])
    return bool(race_laps["TyreCompound"].isin(WET_COMPOUNDS).any())


def pick_reference_driver(laps: pd.DataFrame) -> Tuple[str, str]:
    final_rows = (
        laps.sort_values(["Driver", "LapNumber"])
        .groupby("Driver", as_index=False)
        .tail(1)
        .copy()
    )
    final_rows = final_rows[final_rows["Team"].apply(is_valid_subject)].copy()
    if final_rows.empty:
        raise ValueError("No midfield driver found for validation race")
    final_rows = final_rows.sort_values(["Position", "LapNumber", "Driver"])
    row = final_rows.iloc[0]
    return str(row["Driver"]), str(row["Team"])


def constrained_pit_lap_choice_count(race_length: int, stop_count: int, min_stint_length: int) -> int:
    # Count compositions of race_length into stop_count + 1 stints, each >= min_stint_length.
    remainder = race_length - (stop_count + 1) * min_stint_length
    if remainder < 0:
        return 0
    return comb(remainder + stop_count, stop_count)


def compound_sequence_count(starting_compound: str, stop_count: int, is_wet_race: bool) -> int:
    base = len(DRY_COMPOUNDS) ** stop_count
    if is_wet_race:
        return base
    if starting_compound in set(DRY_COMPOUNDS):
        return base - 1
    return base


def constrained_feasible_set_size(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    max_stops: int,
    min_stint_length: int,
) -> int:
    total = 0
    for stop_count in range(1, max_stops + 1):
        total += (
            constrained_pit_lap_choice_count(race_length, stop_count, min_stint_length)
            * compound_sequence_count(starting_compound, stop_count, is_wet_race)
        )
    return total


def strategy_stint_lengths(race_length: int, stops: List[Tuple[int, str]]) -> List[int]:
    stop_laps = [int(lap) for lap, _ in stops]
    boundaries = [1] + stop_laps + [race_length + 1]
    return [int(boundaries[i + 1] - boundaries[i]) for i in range(len(boundaries) - 1)]


def is_strategy_feasible_with_min_stint(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    stops: List[Tuple[int, str]],
    max_stops: int,
    min_stint_length: int,
) -> Tuple[bool, Optional[str]]:
    base_ok = is_strategy_feasible(
        race_length=race_length,
        starting_compound=starting_compound,
        is_wet_race=is_wet_race,
        stops=stops,
        dry_compounds=DRY_COMPOUNDS,
        max_stops=max_stops,
    )
    if not base_ok:
        stop_count = len(stops)
        if stop_count > max_stops:
            return False, f"stop_count={stop_count} exceeds max_stops={max_stops} by {stop_count - max_stops}"
        return False, "violates existing strategy.py feasibility rules"

    stint_lengths = strategy_stint_lengths(race_length, stops)
    shortest = min(stint_lengths)
    if shortest < min_stint_length:
        return False, (
            f"shortest_stint={shortest} below min_stint_length={min_stint_length} by "
            f"{min_stint_length - shortest}"
        )
    return True, None


def race_group_key(year: int, round_num: int, event_name: str) -> Tuple[int, int, str]:
    return int(year), int(round_num), str(event_name)


def label_event_in_archetype_sample(event_name: str) -> Optional[str]:
    e = str(event_name).lower()
    archetype_labels = [
        ("Monaco", [r"\bmonaco\b"]),
        ("Azerbaijan/Baku", [r"\bazerbaijan\b", r"\bbaku\b"]),
        ("Singapore", [r"\bsingapore\b"]),
        ("Miami", [r"\bmiami\b"]),
        ("Italian/Monza", [r"\bitalian\b", r"\bmonza\b"]),
        ("Belgian/Spa", [r"\bbelgian\b", r"\bspa\b"]),
        ("British/Silverstone", [r"\bbritish\b", r"\bsilverstone\b"]),
        ("Bahrain", [r"\bbahrain\b"]),
        ("Spanish/Barcelona", [r"\bspanish\b", r"\bbarcelona\b"]),
        ("Hungarian", [r"\bhungarian\b"]),
        ("Abu Dhabi", [r"\babu\s+dhabi\b"]),
        ("Japanese/Suzuka", [r"\bjapanese\b", r"\bsuzuka\b"]),
    ]
    for label, patterns in archetype_labels:
        if any(re.search(pattern, e) for pattern in patterns):
            return label
    return None


def main():
    base_df = pd.read_csv(BASE_DF_PATH)
    base_df = base_df[base_df["Year"].isin([2022, 2023, 2024])].copy()
    base_df = base_df[to_bool(base_df["is_midfield"])].copy()
    base_df = base_df.sort_values(["Year", "Round", "EventName", "Driver", "LapNumber"]).reset_index(drop=True)

    records: List[Dict] = []
    all_stints: List[Dict] = []
    short_stints: List[Dict] = []

    keys = ["Year", "Round", "EventName", "Driver", "Team"]
    for key, driver_laps in base_df.groupby(keys, sort=False):
        year, round_num, event_name, driver, team = key
        stints = build_stints(driver_laps)
        short_count = sum(1 for s in stints if s["length"] <= 2)
        stop_count = len(detect_stop_events(driver_laps))
        rg_key = race_group_key(year, round_num, event_name)

        records.append(
            {
                "Year": int(year),
                "Round": int(round_num),
                "EventName": str(event_name),
                "Driver": str(driver),
                "Team": str(team),
                "stop_count": int(stop_count),
                "short_stint_count": int(short_count),
            }
        )

        repeat_short_signature = short_count >= 2
        for stint in stints:
            stint_row = {
                "Year": int(year),
                "Round": int(round_num),
                "EventName": str(event_name),
                "Driver": str(driver),
                "Team": str(team),
                "race_group": rg_key,
                "length": int(stint["length"]),
                "start_lap": int(stint["start_lap"]),
                "ended_by_stop": bool(stint["ended_by_stop"]),
                "ending_stop_lap": stint["ending_stop_lap"],
                "flag_sc_vsc_proximity": bool(stint["sc_or_vsc_at_ending_stop"]),
                "flag_start_lap_leq3": int(stint["start_lap"]) <= 3,
                "flag_repeat_short_stints": bool(repeat_short_signature),
            }
            all_stints.append(stint_row)
            if stint_row["length"] <= 2:
                short_stints.append(stint_row)

    all_stints_df = pd.DataFrame(all_stints)
    short_df = pd.DataFrame(short_stints)
    driver_race_df = pd.DataFrame(records)

    print("Step 1 - Characterise stints of length <=2")
    total_short = int(len(short_df))
    print(f"short_stints_total={total_short}")
    if total_short == 0:
        raise RuntimeError("No short stints found; cannot continue analysis")

    for col in ["flag_sc_vsc_proximity", "flag_start_lap_leq3", "flag_repeat_short_stints"]:
        cnt = int(short_df[col].sum())
        pct = cnt / total_short
        print(f"{col}_count={cnt}")
        print(f"{col}_pct={pct:.6f}")

    any_flag = (
        short_df["flag_sc_vsc_proximity"]
        | short_df["flag_start_lap_leq3"]
        | short_df["flag_repeat_short_stints"]
    )
    unexplained = int((~any_flag).sum())
    print(f"short_stints_with_any_flag={int(any_flag.sum())}")
    print(f"short_stints_unexplained_count={unexplained}")
    print(f"short_stints_unexplained_pct={unexplained / total_short:.6f}")

    print("\nStep 2 - Re-derive minimum stint length from clean subset")
    contaminated_short_idx = set(short_df.loc[any_flag].index.tolist())
    short_df = short_df.reset_index(drop=True)
    contaminated_short_rows = short_df.loc[any_flag.reset_index(drop=True)]

    contaminated_keys = set(
        zip(
            contaminated_short_rows["Year"],
            contaminated_short_rows["Round"],
            contaminated_short_rows["EventName"],
            contaminated_short_rows["Driver"],
            contaminated_short_rows["start_lap"],
            contaminated_short_rows["length"],
        )
    )

    all_stints_df["is_contaminated_short"] = all_stints_df.apply(
        lambda r: (
            r["length"] <= 2
            and (
                int(r["Year"]),
                int(r["Round"]),
                str(r["EventName"]),
                str(r["Driver"]),
                int(r["start_lap"]),
                int(r["length"]),
            )
            in contaminated_keys
        ),
        axis=1,
    )

    clean_stints = all_stints_df.loc[~all_stints_df["is_contaminated_short"], "length"].astype(int)
    q = clean_stints.quantile([0.01, 0.05, 0.10, 0.5], interpolation="nearest")
    vc = clean_stints.value_counts().sort_index()
    over_10 = vc[vc > 10]
    shortest_over_10 = int(over_10.index.min()) if not over_10.empty else None

    print(f"clean_stints_total={int(len(clean_stints))}")
    print(f"clean_stint_length_min={int(clean_stints.min())}")
    print(f"clean_stint_length_p01_nearest={int(q.loc[0.01])}")
    print(f"clean_stint_length_p05_nearest={int(q.loc[0.05])}")
    print(f"clean_stint_length_p10_nearest={int(q.loc[0.10])}")
    print(f"clean_stint_length_median_nearest={int(q.loc[0.5])}")
    print(f"clean_shortest_stint_with_count_gt_10={shortest_over_10}")

    if shortest_over_10 is None:
        raise RuntimeError("No clean stint length appears more than 10 times")

    revised_min_stint_length = int(shortest_over_10)

    print("\nStep 3 - Re-check max_stops on archetype-sampled races")
    driver_race_df["archetype_race_label"] = driver_race_df["EventName"].apply(label_event_in_archetype_sample)
    sampled = driver_race_df[driver_race_df["archetype_race_label"].notna()].copy()
    high_stop = sampled[sampled["stop_count"] >= 4].copy()

    if high_stop.empty:
        print("archetype_sample_has_stop_count_ge_4=False")
        revised_max_stops = 3
        print("revised_max_stops=3")
    else:
        print("archetype_sample_has_stop_count_ge_4=True")
        print("revised_max_stops=4")
        revised_max_stops = 4
        cols = ["Year", "Round", "EventName", "Driver", "Team", "stop_count", "archetype_race_label"]
        print(high_stop[cols].sort_values(["Year", "Round", "Driver"]).to_string(index=False))

    print("\nStep 4 - Constrained feasible set sizes on validation races")
    size_rows: List[Dict] = []
    selected_rows: List[Dict] = []
    feasibility_rows: List[Dict] = []

    for label, year, round_num in RACES:
        state = load_race_state(year=year, round_num=round_num)
        race_laps = state["laps"]
        driver, team = pick_reference_driver(race_laps)
        selected_rows.append({"race": label, "driver": driver, "team": team})

        driver_laps = race_laps[race_laps["Driver"] == driver].sort_values("LapNumber").reset_index(drop=True)
        race_length = int(driver_laps["LapNumber"].max())
        is_wet_race = determine_wet_race(race_laps)
        actual = extract_actual_strategy(driver_laps)

        constrained_size = constrained_feasible_set_size(
            race_length=race_length,
            starting_compound=actual["starting_compound"],
            is_wet_race=is_wet_race,
            max_stops=revised_max_stops,
            min_stint_length=revised_min_stint_length,
        )
        original_size = ORIGINAL_UNCONSTRAINED_SIZES[label]
        reduction_factor = original_size / constrained_size if constrained_size > 0 else float("inf")

        size_rows.append(
            {
                "race": label,
                "driver": driver,
                "team": team,
                "max_stops": revised_max_stops,
                "min_stint_length": revised_min_stint_length,
                "original_unconstrained_size": original_size,
                "feasible_set_size": constrained_size,
                "reduction_factor_vs_original": round(reduction_factor, 3) if constrained_size > 0 else "inf",
            }
        )

        feasible, violation = is_strategy_feasible_with_min_stint(
            race_length=race_length,
            starting_compound=actual["starting_compound"],
            is_wet_race=is_wet_race,
            stops=actual["stops"],
            max_stops=revised_max_stops,
            min_stint_length=revised_min_stint_length,
        )
        feasibility_rows.append(
            {
                "race": label,
                "driver": driver,
                "team": team,
                "actual_stop_count": len(actual["stops"]),
                "actual_stint_lengths": strategy_stint_lengths(race_length, actual["stops"]),
                "actual_strategy_feasible": feasible,
                "violation": violation,
                "actual_strategy": actual["stops"],
            }
        )

    selected_df = pd.DataFrame(selected_rows)
    dup = selected_df["driver"].value_counts()
    dup = dup[dup > 1]
    print(selected_df.to_string(index=False))
    if dup.empty:
        print("duplicate_selected_driver_flag=None")
    else:
        dup_text = ", ".join(f"{driver} x{count}" for driver, count in dup.items())
        print(f"duplicate_selected_driver_flag={dup_text}")

    size_df = pd.DataFrame(size_rows)
    print(size_df.to_string(index=False))

    print("\nStep 5 - Actual strategy feasibility under revised constraints")
    feasibility_df = pd.DataFrame(feasibility_rows)
    print(feasibility_df.to_string(index=False))

    still_millions = size_df[size_df["feasible_set_size"] >= 1_000_000]
    if still_millions.empty:
        print("tractability_status=No race remains at feasible-set size >= 1,000,000 under revised constraints")
    else:
        rows = ", ".join(
            f"{r.race}: {int(r.feasible_set_size)}"
            for r in still_millions.itertuples(index=False)
        )
        print(
            "tractability_status=UNRESOLVED: feasible-set size still in the millions for "
            + rows
        )


if __name__ == "__main__":
    main()