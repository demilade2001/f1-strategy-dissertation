"""
LinkedIn figure: Anchoring cost share by team (lambda 1.0)

Reads phase3_cross_archetype_synthesis.csv, filters to lambda==1.0,
groups by team across three archetypes, computes anchoring share,
and generates a horizontal 100% bar chart for LinkedIn.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams


def main() -> None:
    # Read data
    root = Path(__file__).resolve().parents[2]  # Code Base directory
    data_path = root / "data" / "diagnostics" / "phase3_cross_archetype_synthesis.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Filter to lambda == 1.0
    df_filtered = df[df["lambda"] == 1.0].copy()
    print(f"Filtered to lambda 1.0: {len(df_filtered)} rows")
    
    if len(df_filtered) != 21:
        raise ValueError(f"Expected 21 rows at lambda 1.0, got {len(df_filtered)}")
    
    # Compute per-row anchoring share and verify anchoring is the largest component
    df_filtered["row_anchoring_share"] = (
        df_filtered["avg_cost_anchoring"] / df_filtered["avg_total_cost"]
    )
    
    # Check anchoring is largest component
    for _, row in df_filtered.iterrows():
        components = {
            "conservatism": row["avg_cost_conservatism"],
            "anchoring": row["avg_cost_anchoring"],
            "sc_underweighting": row["avg_cost_sc_underweighting"],
            "unattributed": row["avg_cost_unattributed"],
        }
        if row["avg_cost_anchoring"] != max(components.values()):
            print(f"ERROR: Anchoring not largest in {row['team']} {row['archetype']}")
            print(f"  Components: {components}")
            raise ValueError("Anchoring is not the largest component in all rows")
    
    print("✓ Anchoring is the largest component in all 21/21 rows")
    
    # Find minimum per-cell share
    min_row_idx = df_filtered["row_anchoring_share"].idxmin()
    min_row = df_filtered.loc[min_row_idx]
    min_share = min_row["row_anchoring_share"]
    print(f"\nMinimum per-cell anchoring share: {min_share:.4f} ({min_share*100:.1f}%)")
    print(f"  Row: {min_row['team']} {min_row['archetype']} (lambda {min_row['lambda']})")
    
    # Group by team and sum across archetypes
    team_stats = df_filtered.groupby("team").agg({
        "avg_cost_anchoring": "sum",
        "avg_total_cost": "sum",
    }).reset_index()
    
    team_stats["anchoring_share"] = (
        team_stats["avg_cost_anchoring"] / team_stats["avg_total_cost"]
    )
    team_stats = team_stats.sort_values("anchoring_share", ascending=True)
    
    # Compute overall share
    total_anchoring = team_stats["avg_cost_anchoring"].sum()
    total_cost = team_stats["avg_total_cost"].sum()
    overall_share = total_anchoring / total_cost
    
    print("\n=== Per-Team Anchoring Shares (lambda 1.0) ===")
    for _, row in team_stats.iterrows():
        pct = row["anchoring_share"] * 100
        print(f"  {row['team']:16s}: {pct:6.1f}%")
    
    print(f"\n=== Overall Anchoring Share (lambda 1.0) ===")
    print(f"  Overall: {overall_share*100:.1f}%")
    
    # Verify expected values
    expected_values = {
        "Williams": 0.78,
        "Sauber": 0.81,
        "Aston Martin": 0.81,
        "McLaren": 0.83,
        "Haas": 0.84,
        "Alpine": 0.85,
        "RB": 0.85,
    }
    
    print("\n=== Verification vs Expected Values ===")
    all_match = True
    for team, expected in expected_values.items():
        team_row = team_stats[team_stats["team"] == team]
        if team_row.empty:
            print(f"  {team}: MISSING")
            all_match = False
        else:
            actual = team_row.iloc[0]["anchoring_share"]
            match = "✓" if abs(actual - expected) < 0.01 else "✗"
            print(f"  {team}: {actual*100:.1f}% (expected {expected*100:.1f}%) {match}")
            if abs(actual - expected) >= 0.01:
                all_match = False
    
    if abs(overall_share - 0.82) >= 0.01:
        print(f"  Overall: {overall_share*100:.1f}% (expected 82%) ✗")
        all_match = False
    else:
        print(f"  Overall: {overall_share*100:.1f}% (expected 82%) ✓")
    
    if min_share < 0.50 or min_share > 0.60:
        print(f"  Min per-cell (Williams Tech): {min_share*100:.1f}% (expected ~55%) ✗")
        all_match = False
    else:
        print(f"  Min per-cell (Williams Tech): {min_share*100:.1f}% (expected ~55%) ✓")
    
    if not all_match:
        raise ValueError("Verification failed: values do not match expected")
    
    print("\n✓ All verifications passed")
    
    # Create figure
    fig_width = 10.8
    fig_height = 10.8
    dpi = 100
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    # Adjust layout to reserve space for titles and labels
    fig.subplots_adjust(left=0.20, right=0.90, top=0.79, bottom=0.15)
    
    # Colors
    grey_bg = "#d9d9de"
    navy_fg = "#23264a"
    grey_text = "#6b6b75"
    
    # Plot bars
    y_pos = range(len(team_stats))
    teams = team_stats["team"].tolist()
    shares = team_stats["anchoring_share"].tolist()
    
    # Background grey bars
    ax.barh(y_pos, [1.0] * len(team_stats), color=grey_bg, height=0.66)
    
    # Navy anchoring share bars
    bars = ax.barh(y_pos, shares, color=navy_fg, height=0.66)
    
    # Add percentage labels inside the right end of navy bars
    for i, (team, share) in enumerate(zip(teams, shares)):
        label_x = share - 0.02  # Slightly inside the right edge
        pct_text = f"{int(round(share * 100))}%"
        ax.text(
            label_x, i, pct_text,
            ha="right", va="center",
            color="white", fontweight="bold", fontsize=16
        )
    
    # Y-axis styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(teams, fontsize=20)
    ax.set_ylim(-0.5, len(team_stats) - 0.5)
    ax.tick_params(left=False)
    
    # X-axis styling
    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    
    # Remove spines and gridlines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    
    # Title
    fig.text(
        0.06, 0.92,
        "Anchoring was the biggest cost for every team",
        fontsize=25, fontweight="bold", color="black",
        va="top", ha="left"
    )
    
    # Subtitle
    overall_pct = int(round(overall_share * 100))
    subtitle_line1 = "Share of strategy cost from timing stops off rivals (navy)"
    subtitle_line2 = f"vs all other biases (grey). Overall: {overall_pct}%."
    
    fig.text(
        0.06, 0.865,
        subtitle_line1,
        fontsize=19, color=grey_text,
        va="top", ha="left"
    )
    fig.text(
        0.06, 0.84,
        subtitle_line2,
        fontsize=19, color=grey_text,
        va="top", ha="left"
    )
    
    # Footer
    footer_line1 = "Midfield F1 teams, 2022–24 · 12 races across power, street & technical circuits"
    footer_line2 = "Monte Carlo race simulation vs model-optimal strategy · Demilade Alatise, MSc Business Analytics"
    
    fig.text(
        0.06, 0.05,
        footer_line1,
        fontsize=13, color=grey_text,
        va="bottom", ha="left"
    )
    fig.text(
        0.06, 0.01,
        footer_line2,
        fontsize=13, color=grey_text,
        va="bottom", ha="left"
    )
    
    # Save
    out_dir = root / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "linkedin_anchoring_share.png"
    

    plt.savefig(out_path, dpi=dpi, facecolor="white", pad_inches=0)
    print(f"\n✓ Saved figure to {out_path}")
    
    # Verify file
    if not out_path.exists():
        raise FileNotFoundError(f"Figure not created: {out_path}")
    
    file_size = out_path.stat().st_size
    img = plt.imread(out_path)
    img_height, img_width = img.shape[:2]
    
    print(f"  File size: {file_size} bytes")
    print(f"  Dimensions: {img_width}×{img_height} pixels")
    
    if img_width != 1080 or img_height != 1080:
        raise ValueError(f"Image dimensions {img_width}×{img_height} != 1080×1080")
    
    print("✓ Image dimensions verified (1080×1080)")
    plt.close()


if __name__ == "__main__":
    main()
