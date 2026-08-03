from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIAG_DIR = ROOT / "data" / "diagnostics"
FIG_DIR = ROOT / "data" / "eda_outputs_phase1_base_df" / "figures"

SYNTHESIS_CSV = DIAG_DIR / "phase3_cross_archetype_synthesis.csv"
SIGNED_CSV = DIAG_DIR / "phase3_signed_bias_distributions.csv"

FIG1_PATH = FIG_DIR / "phase3_cost_by_team_archetype.png"
FIG2_PATH = FIG_DIR / "phase3_bias_decomposition_stacked.png"
FIG3_PATH = FIG_DIR / "phase3_sc_directional_skew_by_archetype.png"


ARCHETYPE_ORDER = ["Power", "Street", "Technical"]

# Colorblind-safe palette from matplotlib tab10.
ARCHETYPE_COLORS = {
    "Power": "#2C2A4A",
    "Street": "#AD4E3D",
    "Technical": "#1C4C42",
}

BIAS_COLORS = {
    "avg_cost_conservatism": "#6B2737",
    "avg_cost_anchoring": "#2C2A4A",
    "avg_cost_sc_underweighting": plt.get_cmap("tab10")(1),
    "avg_cost_unattributed": "#355E3B",
}


def _load_lambda_one() -> pd.DataFrame:
    df = pd.read_csv(SYNTHESIS_CSV)
    df = df[np.isclose(df["lambda"], 1.0)].copy()
    df["race_support_count"] = df["race_support_count"].astype(int)
    df["low_confidence_support"] = df["low_confidence_support"].astype(bool)
    return df


def figure_1_cost_by_team_archetype(df: pd.DataFrame) -> None:
    teams = sorted(df["team"].unique())
    x = np.arange(len(teams), dtype=float)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6.8))

    all_bar_heights = []

    for idx, archetype in enumerate(ARCHETYPE_ORDER):
        sub = (
            df[df["archetype"] == archetype]
            .set_index("team")
            .reindex(teams)
            .reset_index()
        )
        offsets = x + (idx - 1) * width
        bars = ax.bar(
            offsets,
            sub["avg_total_cost"].values,
            width=width,
            label=archetype,
            color=ARCHETYPE_COLORS[archetype],
            edgecolor="black",
            linewidth=0.6,
        )

        # Mark lower-confidence support bars with hatch.
        for bar, low_conf in zip(bars, sub["low_confidence_support"].values):
            all_bar_heights.append(float(bar.get_height()))
            if bool(low_conf):
                bar.set_hatch("///")

            # Exact value label inside top edge of each bar.
            h = float(bar.get_height())
            label_y = max(h - 0.55, h * 0.62)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{h:.1f}",
                ha="center",
                va="top",
                fontsize=8,
                color="white",
                fontweight="semibold",
            )

    ax.set_title("Average Total Cost from Model-Optimal by Team and Archetype", fontsize=13, pad=12)
    ax.set_ylabel("Average Total Cost", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    if all_bar_heights:
        ax.set_ylim(0, max(all_bar_heights) * 1.14)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0.0)

    fig.text(
        0.01,
        0.01,
        "Notes: λ=1.0 baseline shown; lambda sensitivity is reported separately and found no material shift.\n"
        "Hatched bars indicate low_confidence_support=True (built from fewer than full archetype race support).",
        fontsize=8.5,
        ha="left",
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG1_PATH, dpi=300)
    plt.close(fig)


def figure_2_bias_decomposition_stacked(df: pd.DataFrame) -> None:
    teams = sorted(df["team"].unique())
    bias_cols = [
        "avg_cost_conservatism",
        "avg_cost_anchoring",
        "avg_cost_sc_underweighting",
        "avg_cost_unattributed",
    ]
    bias_labels = {
        "avg_cost_conservatism": "Conservatism",
        "avg_cost_anchoring": "Anchoring",
        "avg_cost_sc_underweighting": "SC Underweighting",
        "avg_cost_unattributed": "Unattributed",
    }

    fig, axes = plt.subplots(1, 3, figsize=(22, 18), sharey=False)
    for ax, archetype in zip(axes, ARCHETYPE_ORDER):
        sub = (
            df[df["archetype"] == archetype]
            .set_index("team")
            .reindex(teams)
            .reset_index()
        )

        x = np.arange(len(teams))
        bottom = np.zeros(len(teams), dtype=float)
        for col in bias_cols:
            vals = sub[col].values.astype(float)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=BIAS_COLORS[col],
                edgecolor="black",
                linewidth=0.4,
                label=bias_labels[col],
            )
            bottom += vals

        panel_max = float(np.max(bottom)) if len(bottom) else 0.0
        ax.set_ylim(0, panel_max * 1.14 + 0.2)

        ax.set_title(archetype, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(teams, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Average Cost Contribution", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Cost Decomposition by Bias Type, Team, and Archetype", fontsize=14, y=1.07)
    fig.tight_layout()
    fig.savefig(FIG2_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_3_sc_directional_skew() -> None:
    signed_df = pd.read_csv(SIGNED_CSV)
    sc = signed_df[
        (signed_df["bias"] == "sc_underweighting")
        & (signed_df["scope"] == "per_archetype")
        & (signed_df["archetype"].isin(ARCHETYPE_ORDER))
    ].copy()
    sc["archetype"] = pd.Categorical(sc["archetype"], categories=ARCHETYPE_ORDER, ordered=True)
    sc = sc.sort_values("archetype")

    lt_counts = sc["materially_identifiable_lt_1_count"].astype(int).to_numpy()
    gt_counts = sc["materially_identifiable_gt_1_count"].astype(int).to_numpy()
    sample_sizes = sc["materially_identifiable_count"].astype(int).to_numpy()
    labels = sc["archetype"].astype(str).tolist()

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(y, -lt_counts, color="#F58025", edgecolor="black", linewidth=0.5, label="SC Underweighting")
    ax.barh(y, gt_counts, color="#9E3A26", edgecolor="black", linewidth=0.5, label="SC Overweighting")
    ax.axvline(0, color="black", linewidth=0.8)

    max_count = max(int(max(lt_counts.max(), gt_counts.max())), 1)
    pad = max(0.2, max_count * 0.015)

    for i, (arch, lt, gt, n) in enumerate(zip(labels, lt_counts, gt_counts, sample_sizes)):
        ax.text(-lt - pad, i, f"{lt}", va="center", ha="right", fontsize=10)
        ax.text(gt + pad, i, f"{gt}", va="center", ha="left", fontsize=10)
        ax.text(0, i + 0.28, f"{arch}: {lt} vs {gt} (n={n})", va="bottom", ha="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count of materially identifiable rows")
    ax.set_title(
        "Directional Skew in SC-Underweighting Bias by Archetype\n"
        "(Materially Identifiable Cases Only)",
        fontsize=13,
    )
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    fig.text(
        0.01,
        0.01,
        "Note: Materially identifiable sample sizes are modest, especially for Technical.",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG3_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    lambda_one = _load_lambda_one()
    figure_1_cost_by_team_archetype(lambda_one)
    figure_2_bias_decomposition_stacked(lambda_one)
    figure_3_sc_directional_skew()

    print(FIG1_PATH)
    print(FIG2_PATH)
    print(FIG3_PATH)


if __name__ == "__main__":
    main()
