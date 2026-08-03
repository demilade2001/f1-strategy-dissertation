from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "diagnostics" / "phase3_sc_direction_magnitude_analysis.csv"
FIG_DIR = ROOT / "data" / "eda_outputs_phase1_base_df" / "figures"

FIG1_PATH = FIG_DIR / "phase3_sc_magnitude_by_direction_archetype.png"
TABLE_PATH = FIG_DIR / "phase3_sc_direction_by_race_table.md"

ARCHETYPE_ORDER = ["Power", "Street", "Technical"]
DIRECTION_ORDER = ["underweighting", "overweighting"]

# Keep the same direction colors used in diagnostics/phase3_visualization.py figure 3.
DIRECTION_COLORS = {
    "underweighting": "#F58025",
    "overweighting": "#9E3A26",
}


def _load_deduped_lambda_one() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df = df[np.isclose(df["lambda"], 1.0)].copy()

    # One row per driver-race-direction at lambda=1.0.
    dedup_cols = ["archetype", "event_name", "year", "subject_driver", "direction"]
    df = df.sort_values(["archetype", "year", "round", "event_name", "subject_driver", "direction"])
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

    df["year"] = df["year"].astype(int)
    df["round"] = df["round"].astype(int)
    return df


def figure_1_magnitude_by_direction(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8), sharey=True)

    for ax, archetype in zip(axes, ARCHETYPE_ORDER):
        sub_arch = df[df["archetype"] == archetype].copy()

        y_max_local = 0.0
        for x_pos, direction in enumerate(DIRECTION_ORDER):
            sub = sub_arch[sub_arch["direction"] == direction].copy()
            n = len(sub)

            if n == 0:
                continue

            vals = sub["magnitude"].to_numpy(dtype=float)
            y_max_local = max(y_max_local, float(np.max(vals)))

            # For low n, show explicit points only; otherwise combine with a light box.
            if n >= 3:
                ax.boxplot(
                    [vals],
                    positions=[x_pos],
                    widths=0.5,
                    patch_artist=True,
                    boxprops={"facecolor": DIRECTION_COLORS[direction], "alpha": 0.22, "edgecolor": "black"},
                    medianprops={"color": "black", "linewidth": 1.1},
                    whiskerprops={"color": "black", "linewidth": 0.9},
                    capprops={"color": "black", "linewidth": 0.9},
                    flierprops={"marker": "", "markersize": 0},
                )

            jitter = np.linspace(-0.12, 0.12, n) if n > 1 else np.array([0.0])
            ax.scatter(
                np.full(n, x_pos) + jitter,
                vals,
                s=36,
                color=DIRECTION_COLORS[direction],
                edgecolor="black",
                linewidth=0.45,
                zorder=3,
            )

            mean_val = float(np.mean(vals))
            ax.hlines(mean_val, x_pos - 0.2, x_pos + 0.2, color="black", linewidth=1.0, zorder=4)

            # Place mean labels to the side (not on the mean line/points), with a small boxed background.
            # Keep inside panel bounds even when values are near the axis ceiling.
            local_top = float(np.max(vals))
            label_y = min(local_top + 0.55, 17.45)
            if direction == "underweighting":
                label_x = x_pos - 0.27
                label_ha = "right"
            else:
                label_x = x_pos + 0.27
                label_ha = "left"

            ax.text(
                label_x,
                label_y,
                f"mean={mean_val:.2f}",
                ha=label_ha,
                va="center",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.6", "linewidth": 0.4, "pad": 1.5},
                zorder=5,
            )

        n_under = len(sub_arch[sub_arch["direction"] == "underweighting"])
        n_over = len(sub_arch[sub_arch["direction"] == "overweighting"])
        ax.text(
            0.98,
            0.98,
            f"n(under)={n_under}, n(over)={n_over}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2.0},
        )

        ax.set_title(archetype, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Underweighting", "Overweighting"], rotation=15)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(bottom=0)

    all_max = float(df["magnitude"].max()) if not df.empty else 0.0
    y_upper = max(18.0, np.ceil(all_max / 3.0) * 3.0)
    yticks = np.arange(0.0, y_upper + 0.1, 3.0)
    for ax in axes:
        ax.set_ylim(0, y_upper)
        ax.set_yticks(yticks)

    axes[0].set_ylabel("Magnitude = |B - X| (points)")
    fig.suptitle("SC-Axis Deviation Magnitude by Direction, Faceted by Archetype", fontsize=14, y=0.99)
    fig.text(
        0.01,
        0.01,
        "Note: Power and Street have minimal underweighting cases to compare against; "
        "Technical is the only archetype with a substantive mix of both directions.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(FIG1_PATH, dpi=300)
    plt.close(fig)


def write_race_direction_table(df: pd.DataFrame) -> None:
    race_rows = (
        df[["archetype", "event_name", "year", "round"]]
        .drop_duplicates()
        .assign(archetype_order=lambda x: x["archetype"].map({k: i for i, k in enumerate(ARCHETYPE_ORDER)}))
        .sort_values(["archetype_order", "year", "round", "event_name"])
        .drop(columns=["archetype_order"])
    )

    table_records = []
    for rr in race_rows.itertuples(index=False):
        sub = df[
            (df["archetype"] == rr.archetype)
            & (df["event_name"] == rr.event_name)
            & (df["year"] == int(rr.year))
            & (df["round"] == int(rr.round))
        ]

        under_drivers = sorted(sub[sub["direction"] == "underweighting"]["subject_driver"].unique().tolist())
        over_drivers = sorted(sub[sub["direction"] == "overweighting"]["subject_driver"].unique().tolist())

        table_records.append(
            {
                "Archetype": rr.archetype,
                "Race": rr.event_name,
                "Year": int(rr.year),
                "Drivers Underweighting": ", ".join(under_drivers) if under_drivers else "-",
                "Count Underweighting": len(under_drivers),
                "Drivers Overweighting": ", ".join(over_drivers) if over_drivers else "-",
                "Count Overweighting": len(over_drivers),
            }
        )

    out_df = pd.DataFrame(table_records)

    cols = [
        "Archetype",
        "Race",
        "Year",
        "Drivers Underweighting",
        "Count Underweighting",
        "Drivers Overweighting",
        "Count Overweighting",
    ]

    def _escape(text: object) -> str:
        return str(text).replace("|", "\\|")

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in out_df[cols].iterrows():
        lines.append("| " + " | ".join(_escape(row[c]) for c in cols) + " |")

    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_deduped_lambda_one()
    figure_1_magnitude_by_direction(df)
    write_race_direction_table(df)
    print(FIG1_PATH)
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
