from __future__ import annotations

import inspect
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from source_code.features import build_tyre_degradation  # noqa: E402


BASE_DF_PATH = REPO_ROOT / "data" / "processed" / "base_df.csv"
DIAG_DIR = REPO_ROOT / "data" / "diagnostics"
FULL_STATS_PATH = DIAG_DIR / "deg_rate_full_peryear_stats.csv"
FIG_DIR = REPO_ROOT / "data" / "eda_outputs_phase1_base_df" / "figures"

ARCHETYPE_CIRCUITS = [
    "Monaco Grand Prix",
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "Miami Grand Prix",
    "Italian Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Bahrain Grand Prix",
    "Spanish Grand Prix",
    "Hungarian Grand Prix",
    "Abu Dhabi Grand Prix",
    "Japanese Grand Prix",
]

YEARS = [2022, 2023, 2024]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _extract_source_lines(func, start: int, end: int) -> str:
    src_lines, src_start = inspect.getsourcelines(func)
    out = []
    for idx, line in enumerate(src_lines, start=src_start):
        if start <= idx <= end:
            out.append(f"{idx:4d}: {line.rstrip()}")
    return "\n".join(out)


def run_build_with_trace(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    captured: dict[str, object] = {}

    def tracer(frame, event, arg):
        if event == "return" and frame.f_code is build_tyre_degradation.__code__:
            for key in ["deg_rates", "group_stats", "deg_eligible"]:
                if key in frame.f_locals:
                    captured[key] = frame.f_locals[key]
        return tracer

    old_trace = sys.gettrace()
    try:
        sys.settrace(tracer)
        out_df = build_tyre_degradation(df)
    finally:
        sys.settrace(old_trace)

    return out_df, captured


def pooled_unfiltered_stats(df: pd.DataFrame) -> pd.DataFrame:
    curve_base = df[
        df["TyreCompound"].isin(COMPOUNDS)
        & df["TyreLife"].notna()
        & df["fuel_corrected_laptime"].notna()
    ].copy()

    rows = []
    for circuit in ARCHETYPE_CIRCUITS:
        for compound in COMPOUNDS:
            sub = curve_base[(curve_base["EventName"] == circuit) & (curve_base["TyreCompound"] == compound)].copy()
            x = pd.to_numeric(sub["TyreLife"], errors="coerce")
            y = pd.to_numeric(sub["fuel_corrected_laptime"], errors="coerce")
            mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            n = len(x)
            slope = float(np.polyfit(x, y, 1)[0]) if n >= 5 else np.nan
            rows.append(
                {
                    "EventName": circuit,
                    "TyreCompound": compound,
                    "old_pooled_unfiltered_n": n,
                    "old_pooled_unfiltered_slope": slope,
                }
            )

    return pd.DataFrame(rows)


def make_full_stats_table(out_df: pd.DataFrame, captured: dict) -> pd.DataFrame:
    group_stats = captured["group_stats"]

    base_keys = (
        out_df.loc[out_df["TyreCompound"].isin(COMPOUNDS), ["TyreCompound", "EventName", "Year"]]
        .drop_duplicates()
        .sort_values(["TyreCompound", "EventName", "Year"])
    )

    rows = []
    for row in base_keys.itertuples(index=False):
        key = (row.TyreCompound, row.EventName, row.Year)
        stats = group_stats.get(key)

        n_laps = int(stats["n_laps"]) if stats is not None else 0
        model = stats["model"] if stats is not None else "missing"
        slope = float(stats["deg_rate"]) if stats is not None and pd.notna(stats["deg_rate"]) else np.nan
        r2_linear = float(stats["r2_linear"]) if stats is not None and pd.notna(stats["r2_linear"]) else np.nan
        r2_quadratic = float(stats["r2_quadratic"]) if stats is not None and pd.notna(stats["r2_quadratic"]) else np.nan

        mask = (
            (out_df["TyreCompound"] == row.TyreCompound)
            & (out_df["EventName"] == row.EventName)
            & (out_df["Year"] == row.Year)
        )
        final_vals = out_df.loc[mask, "deg_rate"].dropna().unique()
        final_deg_rate = float(final_vals[0]) if len(final_vals) > 0 else np.nan

        rows.append(
            {
                "TyreCompound": row.TyreCompound,
                "EventName": row.EventName,
                "Year": int(row.Year),
                "n_laps": n_laps,
                "model": model,
                "deg_rate_internal": slope,
                "deg_rate_final_assigned": final_deg_rate,
                "r2_linear": r2_linear,
                "r2_quadratic": r2_quadratic,
                "low_sample": bool(n_laps < 30),
            }
        )

    return pd.DataFrame(rows).sort_values(["TyreCompound", "EventName", "Year"]).reset_index(drop=True)


def choose_section364_combos(old_stats: pd.DataFrame) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    mandatory = [
        ("Monaco Grand Prix", "HARD"),
        ("British Grand Prix", "SOFT"),
        ("Hungarian Grand Prix", "HARD"),
    ]

    remain = old_stats.copy()
    remain["abs_slope"] = remain["old_pooled_unfiltered_slope"].abs()
    remain = remain[~remain[["EventName", "TyreCompound"]].apply(tuple, axis=1).isin(mandatory)]
    extras = (
        remain.sort_values("abs_slope", ascending=False)
        .dropna(subset=["old_pooled_unfiltered_slope"])
        .head(3)
    )
    extra_pairs = list(extras[["EventName", "TyreCompound"]].itertuples(index=False, name=None))

    combos = mandatory + extra_pairs

    info_rows = []
    for event, comp in combos:
        source = "section_prompt_mandatory" if (event, comp) in mandatory else "fallback_top_abs_old_pooled_slope"
        info_rows.append({"EventName": event, "TyreCompound": comp, "selection_source": source})

    return combos, pd.DataFrame(info_rows)


def regenerate_plots(full_stats: pd.DataFrame, eligible_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    color_map = {2022: "#1f77b4", 2023: "#2ca02c", 2024: "#d62728"}
    marker_map = {2022: "o", 2023: "s", 2024: "^"}

    for compound in COMPOUNDS:
        fig, axes = plt.subplots(3, 4, figsize=(24, 15), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, circuit in enumerate(ARCHETYPE_CIRCUITS):
            ax = axes[idx]
            panel = eligible_df[(eligible_df["TyreCompound"] == compound) & (eligible_df["EventName"] == circuit)].copy()

            ax.set_title(circuit)
            notes = []

            for year in YEARS:
                ys = panel[panel["Year"] == year].copy()
                x = pd.to_numeric(ys["TyreLife"], errors="coerce")
                y = pd.to_numeric(ys["LapTime_s"], errors="coerce")
                valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]
                n = len(x)

                stats_match = full_stats[
                    (full_stats["TyreCompound"] == compound)
                    & (full_stats["EventName"] == circuit)
                    & (full_stats["Year"] == year)
                ]
                low_sample = bool(stats_match["low_sample"].iloc[0]) if not stats_match.empty else True
                slope_internal = float(stats_match["deg_rate_internal"].iloc[0]) if not stats_match.empty else np.nan
                model = stats_match["model"].iloc[0] if not stats_match.empty else "missing"

                if n == 0:
                    notes.append(f"{year}: insufficient sample (n=0)")
                    continue

                if low_sample:
                    ax.scatter(x, y, s=10, alpha=0.3, color="#777777", marker=marker_map[year], label=f"{year} raw")
                    notes.append(f"{year}: insufficient sample (n={n})")
                    continue

                ax.scatter(x, y, s=12, alpha=0.55, color=color_map[year], marker=marker_map[year], label=f"{year} raw")

                if model == "quadratic" and n >= 5:
                    coeff = np.polyfit(x, y, 2)
                    xx = np.linspace(float(x.min()), float(x.max()), 120)
                    yy = np.polyval(coeff, xx)
                    ax.plot(xx, yy, color=color_map[year], linewidth=2.0)
                elif n >= 5:
                    coeff = np.polyfit(x, y, 1)
                    xx = np.linspace(float(x.min()), float(x.max()), 120)
                    yy = coeff[0] * xx + coeff[1]
                    ax.plot(xx, yy, color=color_map[year], linewidth=2.0)

                notes.append(f"{year}: slope={slope_internal:.4f} (n={n})")

            if notes:
                ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, va="top", fontsize=8)
            ax.set_xlabel("TyreLife")
            ax.set_ylabel("LapTime_s")

        for j in range(len(ARCHETYPE_CIRCUITS), len(axes)):
            axes[j].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)

        fig.suptitle(f"{compound} degradation curves by circuit (per-year, build_tyre_degradation eligibility)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = FIG_DIR / f"tyre_degradation_curves_{compound.lower()}_peryear.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved figure: {out_path}")


def build_comparison_table(
    full_stats: pd.DataFrame,
    old_pooled: pd.DataFrame,
    combos: list[tuple[str, str]],
    combo_sources: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    old_lookup = old_pooled.set_index(["EventName", "TyreCompound"])

    for circuit, compound in combos:
        old_n = int(old_lookup.loc[(circuit, compound), "old_pooled_unfiltered_n"])
        old_slope = float(old_lookup.loc[(circuit, compound), "old_pooled_unfiltered_slope"])
        src = combo_sources[(combo_sources["EventName"] == circuit) & (combo_sources["TyreCompound"] == compound)]["selection_source"].iloc[0]

        subset = full_stats[(full_stats["EventName"] == circuit) & (full_stats["TyreCompound"] == compound)].copy()
        for year in YEARS:
            row_match = subset[subset["Year"] == year]
            if row_match.empty:
                rows.append(
                    {
                        "EventName": circuit,
                        "TyreCompound": compound,
                        "Year": year,
                        "old_pooled_unfiltered_n": old_n,
                        "old_pooled_unfiltered_slope": old_slope,
                        "new_peryear_filtered_n": 0,
                        "new_peryear_filtered_slope": np.nan,
                        "low_sample": True,
                        "combo_source": src,
                    }
                )
            else:
                rm = row_match.iloc[0]
                rows.append(
                    {
                        "EventName": circuit,
                        "TyreCompound": compound,
                        "Year": year,
                        "old_pooled_unfiltered_n": old_n,
                        "old_pooled_unfiltered_slope": old_slope,
                        "new_peryear_filtered_n": int(rm["n_laps"]),
                        "new_peryear_filtered_slope": float(rm["deg_rate_internal"]) if pd.notna(rm["deg_rate_internal"]) else np.nan,
                        "low_sample": bool(rm["low_sample"]),
                        "combo_source": src,
                    }
                )

    out = pd.DataFrame(rows)
    return out.sort_values(["EventName", "TyreCompound", "Year"]).reset_index(drop=True)


def main() -> None:
    print_header("Load base_df and call build_tyre_degradation directly")
    if not BASE_DF_PATH.exists():
        raise FileNotFoundError(f"Missing base_df.csv at {BASE_DF_PATH}")

    base_df = pd.read_csv(BASE_DF_PATH)
    print(f"Loaded base_df: {BASE_DF_PATH}")
    print(f"Shape before build_tyre_degradation: {base_df.shape[0]} x {base_df.shape[1]}")

    out_df, captured = run_build_with_trace(base_df)

    if not {"group_stats", "deg_rates", "deg_eligible"}.issubset(captured.keys()):
        print_header("Function internals are not externally visible; printing source lines 1145-1170")
        print(_extract_source_lines(build_tyre_degradation, 1145, 1170))
        raise RuntimeError("Could not capture build_tyre_degradation internals via tracing.")

    full_stats = make_full_stats_table(out_df, captured)

    print_header("Step 1 output — per (TyreCompound, EventName, Year) table from build_tyre_degradation internals")
    print(full_stats.to_string(index=False))

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    full_stats.to_csv(FULL_STATS_PATH, index=False)
    print_header("Step 2 + 3 output")
    print(f"Saved full per-year stats CSV: {FULL_STATS_PATH}")
    print("Included boolean low_sample flag where n_laps < 30.")

    print_header("Step 4 + 5 output — regenerate 12-circuit scoped per-year figures")
    scoped_stats = full_stats[full_stats["EventName"].isin(ARCHETYPE_CIRCUITS)].copy()

    deg_eligible = captured["deg_eligible"]
    if not isinstance(deg_eligible, pd.Series):
        deg_eligible = pd.Series(deg_eligible, index=out_df.index)
    eligible_df = out_df.loc[deg_eligible.fillna(False)].copy()

    regenerate_plots(scoped_stats, eligible_df)

    print_header("Step 6 output — before/after comparison table")
    old_pooled = pooled_unfiltered_stats(out_df)
    combos, combo_sources = choose_section364_combos(old_pooled)

    if any(src == "fallback_top_abs_old_pooled_slope" for src in combo_sources["selection_source"]):
        print(
            "WARNING: Could not locate explicit Section 3.6.4 list in repository text. "
            "Using the three mandated combinations plus three fallback combinations with the largest absolute old pooled slopes."
        )
        print(combo_sources.to_string(index=False))

    comparison = build_comparison_table(scoped_stats, old_pooled, combos, combo_sources)
    print(comparison.to_string(index=False))

    hungarian_hard = comparison[
        (comparison["EventName"] == "Hungarian Grand Prix")
        & (comparison["TyreCompound"] == "HARD")
    ].copy()
    print("\nHungarian Hard explicit subset:")
    print(hungarian_hard.to_string(index=False))


if __name__ == "__main__":
    main()
