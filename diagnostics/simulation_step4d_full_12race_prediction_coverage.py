from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source_code.simulation.config import LOCKED_ARCHETYPE_RACES
from source_code.simulation.race_state import load_race_state
TARGET_7_RACES: List[Tuple[int, int, str]] = [
    (2023, 10, "British Grand Prix"),
    (2023, 6, "Monaco Grand Prix"),
    (2022, 17, "Singapore Grand Prix"),
    (2022, 22, "Abu Dhabi Grand Prix"),
    (2022, 13, "Hungarian Grand Prix"),
    (2022, 18, "Japanese Grand Prix"),
    (2023, 7, "Spanish Grand Prix"),
]


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:
    print_header("Step 4d - Full 12-race prediction coverage validation")
    print(f"locked_race_count={len(LOCKED_ARCHETYPE_RACES)}")

    rows = []
    for race in LOCKED_ARCHETYPE_RACES:
        year = int(race["year"])
        round_num = int(race["round"])
        event_name = str(race["event_name"])

        state = load_race_state(year, round_num)
        laps = state["laps"].copy()

        total_rows = int(len(laps))
        true_count = int(laps["has_lap_level_prob"].eq(True).sum())
        false_count = int(laps["has_lap_level_prob"].eq(False).sum())
        coverage_pct = (true_count / total_rows) * 100.0 if total_rows else 0.0

        source_counts = laps["prediction_source"].fillna("none").value_counts().to_dict()
        heldout_count = int(source_counts.get("heldout_2024", 0))
        oof_count = int(source_counts.get("out_of_fold_2022_2023", 0))
        none_count = int(source_counts.get("none", 0))

        rows.append(
            {
                "Year": year,
                "Round": round_num,
                "EventName": event_name,
                "laps_row_count": total_rows,
                "has_lap_level_prob_true": true_count,
                "has_lap_level_prob_false": false_count,
                "coverage_pct": coverage_pct,
                "prediction_source_heldout_2024": heldout_count,
                "prediction_source_out_of_fold_2022_2023": oof_count,
                "prediction_source_none": none_count,
            }
        )

        print(f"race={event_name} ({year} R{round_num})")
        print(f"  laps_row_count={total_rows}")
        print(f"  has_lap_level_prob_true={true_count}")
        print(f"  has_lap_level_prob_false={false_count}")
        print(f"  coverage_pct={coverage_pct:.2f}")
        print(
            "  prediction_source_counts="
            f"heldout_2024:{heldout_count}, "
            f"out_of_fold_2022_2023:{oof_count}, "
            f"none:{none_count}"
        )

    summary_df = pd.DataFrame(rows).sort_values(["Year", "Round", "EventName"]).reset_index(drop=True)

    print_header("Coverage summary table")
    print(
        summary_df.to_string(
            index=False,
            formatters={
                "coverage_pct": lambda x: f"{x:.2f}",
            },
        )
    )

    target_pairs = {(y, r) for y, r, _ in TARGET_7_RACES}
    target_df = summary_df[summary_df[["Year", "Round"]].apply(lambda s: (int(s["Year"]), int(s["Round"])) in target_pairs, axis=1)].copy()
    heldout_2024_df = summary_df[summary_df["Year"].eq(2024)].copy()

    print_header("Seven previously zero-coverage races: post-OOF coverage check")
    if target_df.empty:
        print("No target races found in LOCKED_ARCHETYPE_RACES")
        return

    target_mean = float(target_df["coverage_pct"].mean())
    target_min = float(target_df["coverage_pct"].min())
    target_max = float(target_df["coverage_pct"].max())

    heldout_mean = float(heldout_2024_df["coverage_pct"].mean()) if not heldout_2024_df.empty else float("nan")
    heldout_min = float(heldout_2024_df["coverage_pct"].min()) if not heldout_2024_df.empty else float("nan")
    heldout_max = float(heldout_2024_df["coverage_pct"].max()) if not heldout_2024_df.empty else float("nan")

    print(f"target_7_mean_coverage_pct={target_mean:.2f}")
    print(f"target_7_min_coverage_pct={target_min:.2f}")
    print(f"target_7_max_coverage_pct={target_max:.2f}")
    print(f"heldout_2024_mean_coverage_pct={heldout_mean:.2f}")
    print(f"heldout_2024_min_coverage_pct={heldout_min:.2f}")
    print(f"heldout_2024_max_coverage_pct={heldout_max:.2f}")

    # "Vast majority" threshold check and comparability statement to prior 94-96% expectation.
    vast_majority = bool((target_df["coverage_pct"] >= 90.0).all())
    comparable_band = bool((target_df["coverage_pct"] >= 92.0).all() and (target_df["coverage_pct"] <= 98.0).all())

    print(f"target_7_vast_majority_coverage_ge_90pct={vast_majority}")
    print(f"target_7_comparable_to_2024_like_94_96_band={comparable_band}")


if __name__ == "__main__":
    main()
