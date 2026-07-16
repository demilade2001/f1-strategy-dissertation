from pathlib import Path
import re
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "data" / "models" / "classweight_test_predictions.csv"
BASE_PATH = ROOT / "data" / "processed" / "base_df.csv"
OUT_PATH = ROOT / "data" / "diagnostics" / "phase3_archetype_percentile_thresholds.csv"

ARCHETYPE_COL_CANDIDATES = [
    "circuit_archetype",
    "archetype",
    "circuit_type",
]

PRED_COL_PRIORITY = [
    "xgb_calibrated_prob",
    "xgb_calibrated_probability",
    "xgb_calibrated_pred_proba",
    "calibrated_xgb_prob",
    "calibrated_xgb_probability",
    "xgb_prob_calibrated",
    "xgb_calibrated_proba",
    "pred_prob_calibrated_xgb",
    "y_pred_xgb_classweight_cal",
]

ARCHETYPE_LABELS = {
    0: "Other",
    1: "Power",
    2: "Street",
    3: "Technical",
}


def find_probability_column(df: pd.DataFrame) -> str:
    for col in PRED_COL_PRIORITY:
        if col in df.columns:
            return col

    regex_ranked = []
    for col in df.columns:
        c = col.lower()
        if ("xgb" in c or "xgboost" in c) and "cal" in c and (
            "prob" in c or "proba" in c or "pred" in c
        ):
            regex_ranked.append(col)

    if regex_ranked:
        return regex_ranked[0]

    float_like = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    raise ValueError(
        "Could not find calibrated XGBoost probability column. "
        f"Numeric columns available: {float_like}"
    )


def find_existing_archetype_column(df: pd.DataFrame) -> Optional[str]:
    lower_to_orig = {c.lower(): c for c in df.columns}
    for c in ARCHETYPE_COL_CANDIDATES:
        if c.lower() in lower_to_orig:
            return lower_to_orig[c.lower()]
    return None


def map_archetype_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map(ARCHETYPE_LABELS)
    as_text = series.astype("string")

    # Keep known labels if already textual.
    known_labels = set(ARCHETYPE_LABELS.values())
    keep_text_mask = as_text.isin(known_labels)
    mapped = mapped.astype("string")
    mapped[keep_text_mask] = as_text[keep_text_mask]
    return mapped


def attach_archetype(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    archetype_col = find_existing_archetype_column(pred_df)
    if archetype_col is not None:
        out = pred_df.copy()
        out["circuit_archetype_label"] = map_archetype_label(out[archetype_col])
        matched = out["circuit_archetype_label"].notna().sum()
        rate = matched / len(out) if len(out) else 0.0
        print(f"Step 2: using existing archetype column: {archetype_col}")
        print(f"Step 2: archetype availability success rate = {matched}/{len(out)} ({rate:.2%})")
        return out, "existing"

    base_df = pd.read_csv(BASE_PATH)
    candidate_keys = ["Year", "Round", "Driver", "LapNumber", "CarNumber", "EventName"]
    merge_keys = [k for k in candidate_keys if k in pred_df.columns and k in base_df.columns]
    if not merge_keys:
        raise ValueError(
            "No shared merge keys found between predictions and base_df. "
            f"Pred columns: {list(pred_df.columns)}"
        )

    print("Step 2: no archetype column in predictions; merging from base_df")
    print(f"Step 2: merge keys used: {merge_keys}")

    base_sel = base_df[merge_keys + ["circuit_archetype"]].drop_duplicates(merge_keys)
    merged = pred_df.merge(base_sel, on=merge_keys, how="left")
    merged["circuit_archetype_label"] = map_archetype_label(merged["circuit_archetype"])

    matched = merged["circuit_archetype_label"].notna().sum()
    rate = matched / len(merged) if len(merged) else 0.0
    print(f"Step 2: merge success rate = {matched}/{len(merged)} ({rate:.2%})")
    return merged, "merged"


def compute_percentile_table(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    rows = []
    ordered = ["Other", "Power", "Street", "Technical"]
    for archetype in ordered:
        sub = df[df["circuit_archetype_label"] == archetype][prob_col].dropna()
        if len(sub) == 0:
            rows.append(
                {
                    "archetype": archetype,
                    "row_count": 0,
                    "min": pd.NA,
                    "max": pd.NA,
                    "mean": pd.NA,
                    "p50": pd.NA,
                    "p75": pd.NA,
                    "p90": pd.NA,
                    "p95": pd.NA,
                    "p99": pd.NA,
                }
            )
            continue

        rows.append(
            {
                "archetype": archetype,
                "row_count": int(len(sub)),
                "min": float(sub.min()),
                "max": float(sub.max()),
                "mean": float(sub.mean()),
                "p50": float(sub.quantile(0.50)),
                "p75": float(sub.quantile(0.75)),
                "p90": float(sub.quantile(0.90)),
                "p95": float(sub.quantile(0.95)),
                "p99": float(sub.quantile(0.99)),
            }
        )

    return pd.DataFrame(rows)


def compute_threshold_table(df: pd.DataFrame, prob_col: str, pct_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pct_df.iterrows():
        archetype = r["archetype"]
        sub = df[df["circuit_archetype_label"] == archetype][prob_col].dropna()
        n = int(len(sub))

        if n == 0 or pd.isna(r["p90"]) or pd.isna(r["p95"]):
            rows.append(
                {
                    "archetype": archetype,
                    "row_count": n,
                    "p90_threshold": pd.NA,
                    "p90_count_ge": 0,
                    "p90_pct_ge": pd.NA,
                    "p95_threshold": pd.NA,
                    "p95_count_ge": 0,
                    "p95_pct_ge": pd.NA,
                    "p90_usable_raw_count": 0,
                    "p95_usable_raw_count": 0,
                }
            )
            continue

        p90 = float(r["p90"])
        p95 = float(r["p95"])
        p90_count = int((sub >= p90).sum())
        p95_count = int((sub >= p95).sum())

        rows.append(
            {
                "archetype": archetype,
                "row_count": n,
                "p90_threshold": p90,
                "p90_count_ge": p90_count,
                "p90_pct_ge": p90_count / n,
                "p95_threshold": p95,
                "p95_count_ge": p95_count,
                "p95_pct_ge": p95_count / n,
                "p90_usable_raw_count": p90_count,
                "p95_usable_raw_count": p95_count,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    pred_df = pd.read_csv(PRED_PATH)

    print("Step 1: load and inspect")
    print(f"row_count={len(pred_df)}")
    print("columns_begin")
    for col in pred_df.columns:
        print(col)
    print("columns_end")

    prob_col = find_probability_column(pred_df)
    print(f"calibrated_xgboost_probability_column={prob_col}")

    with_archetype_df, source = attach_archetype(pred_df)
    print(f"Step 2: archetype_source={source}")

    pct_df = compute_percentile_table(with_archetype_df, prob_col)
    print("\nStep 3: percentile summary per archetype")
    print(pct_df.to_string(index=False))

    th_df = compute_threshold_table(with_archetype_df, prob_col, pct_df)
    print("\nStep 4: threshold sanity check")
    print(th_df.to_string(index=False))

    out_df = pd.concat(
        [
            pct_df.assign(table="step3_percentiles"),
            th_df.assign(table="step4_threshold_sanity"),
        ],
        ignore_index=True,
        sort=False,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nStep 5: saved combined output: {OUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()