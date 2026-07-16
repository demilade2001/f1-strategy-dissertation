from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "data" / "processed" / "base_df.csv"
OUT_PATH = ROOT / "data" / "diagnostics" / "phase3_caution_degradation_within_circuit.csv"

STINT_KEYS = ["Driver", "Round", "Stint"]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
NON_WET = {"INTERMEDIATE", "WET"}


def fit_slope(df: pd.DataFrame, x_col: str, y_col: str):
    n = len(df)
    if n < 2:
        return np.nan
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    if np.allclose(x, x[0]):
        return np.nan
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def apply_107_trim(df: pd.DataFrame) -> pd.Series:
    keep = pd.Series(False, index=df.index)
    for (_, _), g in df.groupby(["Driver", "Round"], dropna=False):
        med = g["LapTime_s"].median()
        threshold = med * 1.07
        keep.loc[g.index] = g["LapTime_s"] <= threshold
    return keep


def top_events(df: pd.DataFrame, group_label: str) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["EventName", f"{group_label}_count", f"{group_label}_pct"])
    vc = df["EventName"].value_counts().head(10)
    out = vc.rename_axis("EventName").reset_index(name=f"{group_label}_count")
    out[f"{group_label}_pct"] = out[f"{group_label}_count"] / len(df)
    return out


def main():
    print("Step 0 - Load and cast")
    df = pd.read_csv(BASE_PATH)
    df["is_strategic_stop"] = df["is_strategic_stop"].astype(pd.BooleanDtype())
    df["sc_active"] = df["sc_active"].astype(bool)
    df["vsc_active"] = df["vsc_active"].astype(bool)
    print(f"shape={df.shape[0]}x{df.shape[1]}")
    print(f"row_count_is_74536={len(df) == 74536}")

    caution_mask = df["sc_active"] | df["vsc_active"]

    # Step 1
    print("\nStep 1 - Confound documentation from original split")
    stint = (
        df.groupby(STINT_KEYS, dropna=False)
        .agg(
            total_laps=("LapNumber", "count"),
            caution_laps=("sc_active", lambda s: int((s | df.loc[s.index, "vsc_active"]).sum())),
        )
        .reset_index()
    )

    stint["orig_group"] = np.where(
        stint["caution_laps"] == 0,
        "zero_caution",
        np.where(stint["caution_laps"] >= 3, "ge3_caution", "mid_caution"),
    )

    df1 = df.merge(stint[STINT_KEYS + ["orig_group", "total_laps"]], on=STINT_KEYS, how="left")
    zero_laps = df1[df1["orig_group"] == "zero_caution"]
    ge3_laps = df1[df1["orig_group"] == "ge3_caution"]

    top_zero = top_events(zero_laps, "zero")
    top_ge3 = top_events(ge3_laps, "ge3")
    side = pd.concat([top_zero, top_ge3], axis=1)
    print("Top 10 EventName by row count (side-by-side):")
    print(side.to_string(index=False))

    zero_stints = stint[stint["orig_group"] == "zero_caution"]
    ge3_stints = stint[stint["orig_group"] == "ge3_caution"]
    print(
        "zero_caution_stint_length: "
        f"mean={zero_stints['total_laps'].mean()}, median={zero_stints['total_laps'].median()}"
    )
    print(
        "ge3_caution_stint_length: "
        f"mean={ge3_stints['total_laps'].mean()}, median={ge3_stints['total_laps'].median()}"
    )

    # Step 2
    print("\nStep 2 - Early-stint exposure split")
    early_caution = (
        (df["TyreLife"].notna())
        & (pd.to_numeric(df["TyreLife"], errors="coerce") <= 5)
        & caution_mask
    )
    early_counts = (
        df.assign(early_caution_flag=early_caution)
        .groupby(STINT_KEYS, dropna=False)["early_caution_flag"]
        .sum()
        .reset_index(name="early_caution_laps")
    )
    early_counts["early_class"] = np.where(
        early_counts["early_caution_laps"] == 0, "early_zero", "early_exposed"
    )
    n_early_zero = int((early_counts["early_class"] == "early_zero").sum())
    n_early_exposed = int((early_counts["early_class"] == "early_exposed").sum())
    print(f"stints_early_zero={n_early_zero}")
    print(f"stints_early_exposed={n_early_exposed}")

    # Step 3
    print("\nStep 3 - Within-circuit-year matched comparison")
    df2 = df.merge(early_counts[STINT_KEYS + ["early_class"]], on=STINT_KEYS, how="left")

    elig = (
        (df2["compound_known"] == True)
        & (df2["IsAccurate"] == True)
        & (~df2["sc_active"])
        & (~df2["vsc_active"])
        & (df2["PitInTime"].isna())
        & (df2["PitOutTime"].isna())
        & (~df2["TyreCompound"].isin(NON_WET))
        & df2["LapTime_s"].notna()
        & df2["TyreLife"].notna()
        & df2["fuel_corrected_laptime"].notna()
        & df2["early_class"].isin(["early_zero", "early_exposed"])
    )
    eligible = df2[elig].copy()
    trim_keep = apply_107_trim(eligible)
    eligible = eligible[trim_keep].copy()

    rows = []
    dropped_groups = 0
    group_cols = ["EventName", "Year", "TyreCompound"]
    for keys, g in eligible.groupby(group_cols, dropna=False):
        event, year, compound = keys
        g0 = g[g["early_class"] == "early_zero"]
        g1 = g[g["early_class"] == "early_exposed"]

        n0 = int(len(g0))
        n1 = int(len(g1))

        slope0 = np.nan
        slope1 = np.nan
        if n0 >= 50:
            slope0 = fit_slope(g0, "TyreLife", "fuel_corrected_laptime")
        if n1 >= 50:
            slope1 = fit_slope(g1, "TyreLife", "fuel_corrected_laptime")

        diff = np.nan
        computable = bool(not np.isnan(slope0) and not np.isnan(slope1))
        if computable:
            diff = slope1 - slope0
        else:
            dropped_groups += 1

        rows.append(
            {
                "EventName": event,
                "Year": int(year) if pd.notna(year) else year,
                "TyreCompound": compound,
                "early_zero_slope": slope0,
                "early_zero_n": n0,
                "early_exposed_slope": slope1,
                "early_exposed_n": n1,
                "difference_exposed_minus_zero": diff,
                "both_computable": computable,
            }
        )

    step3_df = pd.DataFrame(rows).sort_values(["Year", "EventName", "TyreCompound"]).reset_index(drop=True)
    if len(step3_df) > 0:
        print(step3_df.to_string(index=False))
    print(f"groups_dropped_for_insufficient_sample={dropped_groups}")

    # Step 4
    print("\nStep 4 - Direction consistency")
    valid = step3_df[step3_df["both_computable"] == True].copy()

    def summary_block(sub: pd.DataFrame, label: str):
        d = sub["difference_exposed_minus_zero"].dropna()
        total = int(len(d))
        pos = int((d > 0).sum())
        neg = int((d < 0).sum())
        mean = float(d.mean()) if total > 0 else np.nan
        median = float(d.median()) if total > 0 else np.nan
        std = float(d.std()) if total > 1 else np.nan
        print(
            f"{label}: total={total}, positive={pos}, negative={neg}, "
            f"mean_diff={mean}, median_diff={median}, std_diff={std}"
        )
        return {
            "segment": label,
            "total_count": total,
            "positive_diff_count": pos,
            "negative_diff_count": neg,
            "mean_difference": mean,
            "median_difference": median,
            "std_difference": std,
        }

    step4_rows = []
    step4_rows.append(summary_block(valid, "overall"))
    for compound in COMPOUNDS:
        sub = valid[valid["TyreCompound"] == compound]
        step4_rows.append(summary_block(sub, compound))
    step4_df = pd.DataFrame(step4_rows)

    # Step 5
    out_step3 = step3_df.copy()
    out_step3.insert(0, "table", "step3_within_circuit_year")

    out_step4 = step4_df.copy()
    out_step4.insert(0, "table", "step4_direction_consistency")

    out_df = pd.concat([out_step3, out_step4], ignore_index=True, sort=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print("\nStep 5 - Save")
    print(f"saved_csv={OUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()