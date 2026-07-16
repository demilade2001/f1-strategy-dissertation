from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "processed" / "base_df.csv"
OUT_PATH = ROOT / "data" / "diagnostics" / "phase3_caution_degradation_audit.csv"

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
NON_WET = {"INTERMEDIATE", "WET"}


def fit_linear_stats(df: pd.DataFrame, x_col: str, y_col: str):
    if len(df) < 2:
        return np.nan, len(df), np.nan
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    if np.allclose(x, x[0]):
        return np.nan, len(df), np.nan
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)
    return float(slope), len(df), float(r2) if not np.isnan(r2) else np.nan


def apply_107_trim(df: pd.DataFrame) -> pd.Series:
    keep = pd.Series(False, index=df.index)
    grouped = df.groupby(["Driver", "Round"], dropna=False)
    for _, g in grouped:
        med = g["LapTime_s"].median()
        threshold = med * 1.07
        keep.loc[g.index] = g["LapTime_s"] <= threshold
    return keep


def main():
    print("Step 0 - Load and cast")
    df = pd.read_csv(BASE_PATH)
    df["is_strategic_stop"] = df["is_strategic_stop"].astype(pd.BooleanDtype())
    df["sc_active"] = df["sc_active"].astype(bool)
    df["vsc_active"] = df["vsc_active"].astype(bool)
    print(f"shape={df.shape[0]}x{df.shape[1]}")
    print(f"row_count_is_74536={len(df) == 74536}")

    caution_mask = df["sc_active"] | df["vsc_active"]
    green_mask = (~df["sc_active"]) & (~df["vsc_active"])

    print("\nStep 1 - Population counts")
    step1_rows = []
    valid_common = df["TyreLife"].notna() & df["fuel_corrected_laptime"].notna()
    for compound in COMPOUNDS:
        comp_mask = df["TyreCompound"] == compound
        caution_n = int((comp_mask & caution_mask & valid_common).sum())
        green_n = int((comp_mask & green_mask & valid_common).sum())
        step1_rows.append(
            {
                "table": "step1_population",
                "compound": compound,
                "caution_n": caution_n,
                "green_n": green_n,
            }
        )
        print(f"{compound}: caution_n={caution_n}, green_n={green_n}")

    print("\nStep 2 - Naive caution-period slope (negative control)")
    base_filter = (
        (df["compound_known"] == True)
        & (~df["TyreCompound"].isin(NON_WET))
        & df["TyreLife"].notna()
        & df["fuel_corrected_laptime"].notna()
    )
    step2_rows = []
    for compound in COMPOUNDS:
        comp_mask = df["TyreCompound"] == compound
        caution_df = df[base_filter & comp_mask & caution_mask].copy()
        green_df = df[base_filter & comp_mask & green_mask].copy()

        slope, n, r2 = fit_linear_stats(caution_df, "TyreLife", "fuel_corrected_laptime")
        caution_mean = float(caution_df["fuel_corrected_laptime"].mean()) if n > 0 else np.nan
        caution_std = float(caution_df["fuel_corrected_laptime"].std()) if n > 1 else np.nan
        green_mean = float(green_df["fuel_corrected_laptime"].mean()) if len(green_df) > 0 else np.nan
        green_std = float(green_df["fuel_corrected_laptime"].std()) if len(green_df) > 1 else np.nan

        step2_rows.append(
            {
                "table": "step2_naive_caution_slope",
                "compound": compound,
                "slope": slope,
                "n": n,
                "r2": r2,
                "caution_mean_fuel_corrected_laptime": caution_mean,
                "caution_std_fuel_corrected_laptime": caution_std,
                "green_mean_fuel_corrected_laptime": green_mean,
                "green_std_fuel_corrected_laptime": green_std,
            }
        )
        print(
            f"{compound}: slope={slope}, n={n}, r2={r2}, "
            f"caution_mean={caution_mean}, caution_std={caution_std}, "
            f"green_mean={green_mean}, green_std={green_std}"
        )

    print("\nStep 3 - Build stint-level caution exposure")
    stint = (
        df.groupby(["Driver", "Round", "Stint"], dropna=False)
        .agg(
            total_laps=("LapNumber", "count"),
            caution_laps=("sc_active", lambda s: int((s | df.loc[s.index, "vsc_active"]).sum())),
        )
        .reset_index()
    )
    stint["caution_lap_proportion"] = stint["caution_laps"] / stint["total_laps"]

    capped = stint["caution_laps"].clip(upper=10)
    counts = capped.value_counts().sort_index()
    print("caution_lap_count_distribution_capped_10plus:")
    for k, v in counts.items():
        label = "10+" if k == 10 else str(int(k))
        print(f"  {label}: {int(v)}")
    zero_stints = int((stint["caution_laps"] == 0).sum())
    ge3_stints = int((stint["caution_laps"] >= 3).sum())
    print(f"stints_zero_caution={zero_stints}")
    print(f"stints_ge3_caution={ge3_stints}")

    print("\nStep 4 - Matched-TyreLife pace comparison")
    stint_key = ["Driver", "Round", "Stint"]
    df2 = df.merge(stint[stint_key + ["caution_laps"]], on=stint_key, how="left")

    elig = (
        (df2["compound_known"] == True)
        & (df2["IsAccurate"] == True)
        & (~df2["sc_active"])
        & (~df2["vsc_active"])
        & (df2["PitInTime"].isna())
        & (df2["PitOutTime"].isna())
        & (~df2["TyreCompound"].isin(NON_WET))
        & df2["TyreLife"].notna()
        & df2["fuel_corrected_laptime"].notna()
        & df2["LapTime_s"].notna()
    )
    eligible_df = df2[elig].copy()
    trim_keep = apply_107_trim(eligible_df)
    eligible_df = eligible_df[trim_keep].copy()

    eligible_df["tyrelife_int"] = pd.to_numeric(eligible_df["TyreLife"], errors="coerce").round().astype("Int64")
    eligible_df = eligible_df[eligible_df["tyrelife_int"].between(1, 25)].copy()

    step4_rows = []
    dropped_by_compound = {}
    for compound in COMPOUNDS:
        comp = eligible_df[eligible_df["TyreCompound"] == compound].copy()
        g0 = comp[comp["caution_laps"] == 0]
        g3 = comp[comp["caution_laps"] >= 3]

        dropped = 0
        for tl in range(1, 26):
            g0_tl = g0[g0["tyrelife_int"] == tl]
            g3_tl = g3[g3["tyrelife_int"] == tl]
            n0 = len(g0_tl)
            n3 = len(g3_tl)
            if n0 < 20 or n3 < 20:
                dropped += 1
                continue
            m0 = float(g0_tl["fuel_corrected_laptime"].mean())
            m3 = float(g3_tl["fuel_corrected_laptime"].mean())
            step4_rows.append(
                {
                    "table": "step4_matched_tyrelife",
                    "compound": compound,
                    "tyrelife": tl,
                    "zero_caution_n": int(n0),
                    "zero_caution_mean_fuel_corrected_laptime": m0,
                    "ge3_caution_n": int(n3),
                    "ge3_caution_mean_fuel_corrected_laptime": m3,
                }
            )
            print(
                f"{compound} TyreLife={tl}: zero_caution_mean={m0} (n={n0}), "
                f"ge3_caution_mean={m3} (n={n3})"
            )
        dropped_by_compound[compound] = dropped
        print(f"{compound}: dropped_tyrelife_values_for_insufficient_sample={dropped}")

    print("\nStep 5 - Candidate multiplier")
    step5_rows = []
    for compound in COMPOUNDS:
        comp = eligible_df[eligible_df["TyreCompound"] == compound].copy()
        g0 = comp[comp["caution_laps"] == 0]
        g3 = comp[comp["caution_laps"] >= 3]

        slope0, n0, r20 = fit_linear_stats(g0, "TyreLife", "fuel_corrected_laptime")
        slope3, n3, r23 = fit_linear_stats(g3, "TyreLife", "fuel_corrected_laptime")
        ratio = np.nan
        if slope0 is not None and not np.isnan(slope0) and slope0 != 0 and slope3 is not None and not np.isnan(slope3):
            ratio = slope3 / slope0

        step5_rows.append(
            {
                "table": "step5_candidate_multiplier",
                "compound": compound,
                "zero_caution_slope": slope0,
                "zero_caution_n": int(n0),
                "zero_caution_r2": r20,
                "ge3_caution_slope": slope3,
                "ge3_caution_n": int(n3),
                "ge3_caution_r2": r23,
                "slope_ratio_ge3_over_zero": ratio,
            }
        )
        print(
            f"{compound}: zero_slope={slope0}, zero_n={n0}, zero_r2={r20}, "
            f"ge3_slope={slope3}, ge3_n={n3}, ge3_r2={r23}, ratio={ratio}"
        )

    out_df = pd.concat(
        [
            pd.DataFrame(step2_rows),
            pd.DataFrame(step4_rows),
            pd.DataFrame(step5_rows),
        ],
        ignore_index=True,
        sort=False,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print("\nStep 6 - Save")
    print(f"saved_csv={OUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()