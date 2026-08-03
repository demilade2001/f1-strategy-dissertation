"""Compute SMOTE per-archetype AUC diagnostics and compare to class-weight references."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def get_archetype_code(df: pd.DataFrame) -> pd.Series:
    if "circuit_archetype" in df.columns and pd.api.types.is_numeric_dtype(df["circuit_archetype"]):
        return df["circuit_archetype"].astype(int)
    if "circuit_archetype_enc" in df.columns:
        return df["circuit_archetype_enc"].astype(int)
    if "circuit_archetype" in df.columns:
        mapping = {"Other": 0, "Power": 1, "Street": 2, "Technical": 3}
        mapped = df["circuit_archetype"].map(mapping)
        if mapped.isna().any():
            unknown = sorted(df.loc[mapped.isna(), "circuit_archetype"].dropna().unique().tolist())
            raise ValueError(f"Unknown circuit_archetype labels: {unknown}")
        return mapped.astype(int)
    raise KeyError("Neither circuit_archetype nor circuit_archetype_enc found")


def main() -> None:
    root = Path(__file__).parent.parent
    preds_path = root / "data" / "models" / "test_predictions.csv"
    features_path = root / "data" / "processed" / "phase2_features_final.csv"

    print_header("Load and split")
    preds = pd.read_csv(preds_path)
    df = pd.read_csv(features_path)

    # Standing serialization casts.
    df["is_strategic_stop"] = df["is_strategic_stop"].astype(pd.BooleanDtype())
    df["sc_active"] = df["sc_active"].astype(bool)
    df["vsc_active"] = df["vsc_active"].astype(bool)
    df["caution_active"] = df["caution_active"].astype(bool)

    test_df = df[df["Year"] == 2024].copy()
    train_df = df[df["Year"] <= 2023].copy()
    print(f"Train rows (Year <= 2023): {len(train_df)}")
    print(f"Test rows (Year == 2024): {len(test_df)}")
    if len(test_df) != len(preds):
        raise ValueError(f"Row mismatch: test_df={len(test_df)}, predictions={len(preds)}")

    print_header("Join archetype onto SMOTE predictions")
    join_cols = ["Year", "Round", "Driver", "LapNumber"]
    missing_pred_cols = [c for c in join_cols + ["y_true", "y_pred_xgb_cal"] if c not in preds.columns]
    if missing_pred_cols:
        raise KeyError(f"Missing expected prediction columns: {missing_pred_cols}")

    test_join = test_df[join_cols].copy()
    test_join["circuit_archetype"] = get_archetype_code(test_df)
    test_join["sc_vsc_next3_truth"] = test_df["sc_vsc_next3"].astype(int)

    merged = preds.merge(test_join, on=join_cols, how="left", validate="one_to_one")
    if merged["circuit_archetype"].isna().any():
        raise ValueError("Failed to align some predictions with circuit_archetype after join")

    # Extra sanity: joined truth should match stored y_true.
    truth_match = (merged["y_true"].astype(int) == merged["sc_vsc_next3_truth"].astype(int)).all()
    print(f"Truth alignment check (y_true vs joined test truth): {truth_match}")
    if not truth_match:
        raise ValueError("y_true does not match joined sc_vsc_next3 truth; split/join likely incorrect")

    merged["circuit_archetype"] = merged["circuit_archetype"].astype(int)

    print_header("SMOTE per-archetype metrics")
    name_map = {0: "Other", 1: "Power", 2: "Street", 3: "Technical"}
    smote_ref_brier = {3: 0.054, 0: 0.100, 1: 0.113, 2: 0.115}
    classweight_ref = {
        3: {"brier": 0.061, "auc": 0.648},
        2: {"brier": 0.089, "auc": 0.535},
        0: {"brier": 0.093, "auc": 0.784},
        1: {"brier": 0.118, "auc": 0.766},
    }

    rows = []
    sanity_rows = []
    for code in [0, 1, 2, 3]:
        sub = merged[merged["circuit_archetype"] == code]
        n = len(sub)
        if n == 0:
            rows.append(
                {
                    "archetype": name_map[code],
                    "row_count": 0,
                    "base_rate": np.nan,
                    "smote_brier": np.nan,
                    "smote_auc_roc": np.nan,
                    "classweight_brier": classweight_ref[code]["brier"],
                    "classweight_auc_roc": classweight_ref[code]["auc"],
                    "auc_gap_cw_minus_smote": np.nan,
                }
            )
            sanity_rows.append((name_map[code], False, np.nan, smote_ref_brier[code]))
            continue

        y_true = sub["y_true"].astype(int)
        y_prob = sub["y_pred_xgb_cal"].astype(float)
        base_rate = float(y_true.mean())
        brier = float(brier_score_loss(y_true, y_prob))
        auc = float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else np.nan

        cw_auc = classweight_ref[code]["auc"]
        rows.append(
            {
                "archetype": name_map[code],
                "row_count": n,
                "base_rate": base_rate,
                "smote_brier": brier,
                "smote_auc_roc": auc,
                "classweight_brier": classweight_ref[code]["brier"],
                "classweight_auc_roc": cw_auc,
                "auc_gap_cw_minus_smote": cw_auc - auc if not pd.isna(auc) else np.nan,
            }
        )

        # Sanity check against documented reference values to 3 decimals.
        sanity_ok = round(brier, 3) == round(smote_ref_brier[code], 3)
        sanity_rows.append((name_map[code], sanity_ok, brier, smote_ref_brier[code]))

    for archetype, ok, got, ref in sanity_rows:
        if pd.isna(got):
            print(f"Sanity {archetype}: no rows, cannot verify")
        else:
            status = "PASS" if ok else "FAIL"
            print(f"Sanity {archetype}: {status} (recomputed={got:.3f}, reference={ref:.3f})")

    if not all(ok for _, ok, got, _ in sanity_rows if not pd.isna(got)):
        print("Warning: At least one Brier sanity check failed; inspect split/join assumptions before trusting AUC.")

    result_df = pd.DataFrame(rows)
    print("\nConsolidated archetype table:")
    print(
        result_df.to_string(
            index=False,
            formatters={
                "base_rate": lambda x: "nan" if pd.isna(x) else f"{x:.4%}",
                "smote_brier": lambda x: "nan" if pd.isna(x) else f"{x:.3f}",
                "smote_auc_roc": lambda x: "nan" if pd.isna(x) else f"{x:.3f}",
                "classweight_brier": lambda x: f"{x:.3f}",
                "classweight_auc_roc": lambda x: f"{x:.3f}",
                "auc_gap_cw_minus_smote": lambda x: "nan" if pd.isna(x) else f"{x:+.3f}",
            },
        )
    )

    print_header("Street archetype conclusion")
    street_row = result_df[result_df["archetype"] == "Street"]
    if street_row.empty or pd.isna(street_row["smote_auc_roc"].iloc[0]):
        print("Street conclusion unavailable: Street AUC could not be computed.")
        return

    street_smote_auc = float(street_row["smote_auc_roc"].iloc[0])
    street_cw_auc = 0.535

    near_point_five = abs(street_smote_auc - 0.5) <= 0.05
    meaningfully_higher_than_cw = (street_smote_auc - street_cw_auc) > 0.05

    if near_point_five:
        print(
            f"Street AUC-ROC is {street_smote_auc:.3f}, within 0.05 of 0.5. "
            "This indicates weak discrimination is a shared feature-set limitation independent of resampling."
        )
    elif meaningfully_higher_than_cw:
        print(
            f"Street AUC-ROC is {street_smote_auc:.3f}, meaningfully higher than class-weight Street AUC 0.535. "
            "This indicates class-weighting is degrading Street discrimination specifically."
        )
    else:
        print(
            f"Street AUC-ROC is {street_smote_auc:.3f}: not near 0.5 and not meaningfully above 0.535. "
            "This suggests no clear evidence that class-weighting specifically degrades Street discrimination."
        )


if __name__ == "__main__":
    main()