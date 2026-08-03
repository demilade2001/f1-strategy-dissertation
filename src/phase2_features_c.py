"""Phase 2 feature construction script C.

Builds the final modeling table from phase2_features_b.csv and writes:
- data/processed/phase2_features_final.csv
- data/processed/model_feature_list.txt
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fit_encoder_on_train(train_series: pd.Series) -> tuple[LabelEncoder, dict]:
    encoder = LabelEncoder()
    clean = train_series.dropna().astype(str)
    encoder.fit(clean)
    mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
    return encoder, mapping


def transform_with_unseen_to_minus_one(series: pd.Series, mapping: dict) -> pd.Series:
    as_string = series.astype("string")
    encoded = as_string.map(mapping).fillna(-1).astype(int)
    return encoded


def main() -> None:
    root = Path(__file__).parent.parent
    input_path = root / "data" / "processed" / "phase2_features_b.csv"
    output_path = root / "data" / "processed" / "phase2_features_final.csv"
    feature_list_path = root / "data" / "processed" / "model_feature_list.txt"

    print_header("Step 0 - Load and validate")
    df = pd.read_csv(input_path)
    bool_cols = [
        "sc_active",
        "vsc_active",
        "caution_active",
        "yellow_flag",
        "wet_race_flag",
        "undercut_threat_flag",
        "overcut_threat_flag",
        "high_sc_circuit",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(bool)
    df["is_strategic_stop"] = df["is_strategic_stop"].astype(pd.BooleanDtype())
    print(f"Loaded shape: {df.shape[0]} x {df.shape[1]}")
    if df.shape != (74536, 68):
        raise ValueError(f"Expected shape (74536, 68), found {df.shape}")

    print_header("Step 1 - Drop cars_within_1s_proxy")
    if "cars_within_1s_proxy" not in df.columns:
        raise KeyError("cars_within_1s_proxy column not found")
    df = df.drop(columns=["cars_within_1s_proxy"])
    print("Dropped column: cars_within_1s_proxy")
    print(f"Current column count: {len(df.columns)}")

    print_header("Step 2 - Define final model feature list")
    MODEL_FEATURES = [
        "year_encoded",
        "race_fraction",
        "LapNumber",
        "TyreLife",
        "Stint",
        "TyreCompound",
        "circuit_archetype",
        "deg_rate_corrected",
        "deg_rate_corrected_reliable",
        "pit_window_delta",
        "undercut_threat_flag",
        "overcut_threat_flag",
        "circuit_sc_rate",
        "circuit_vsc_rate",
        "sector_incident_rate",
        "high_sc_circuit",
        "caution_active",
        "sc_active",
        "vsc_active",
        "yellow_flag",
        "wet_race_flag",
        "field_mean_tyre_age",
        "field_std_tyre_age",
        "field_laptime_std",
        "position_at_stake",
        "pace_differential",
    ]
    print("MODEL_FEATURES:")
    for feature in MODEL_FEATURES:
        print(f"  - {feature}")
    print(f"MODEL_FEATURES count: {len(MODEL_FEATURES)}")
    if len(MODEL_FEATURES) != 26:
        raise ValueError(f"Expected 26 model features, found {len(MODEL_FEATURES)}")

    print_header("Step 3 - Label encode categoricals")
    train_mask = df["Year"] <= 2023
    train = df.loc[train_mask].copy()

    tyre_encoder, tyre_mapping = fit_encoder_on_train(train["TyreCompound"])
    archetype_encoder, archetype_mapping = fit_encoder_on_train(train["circuit_archetype"])

    df["TyreCompound_enc"] = transform_with_unseen_to_minus_one(df["TyreCompound"], tyre_mapping)
    df["circuit_archetype_enc"] = transform_with_unseen_to_minus_one(df["circuit_archetype"], archetype_mapping)

    MODEL_FEATURES = [
        "TyreCompound_enc" if f == "TyreCompound" else "circuit_archetype_enc" if f == "circuit_archetype" else f
        for f in MODEL_FEATURES
    ]

    print("TyreCompound LabelEncoder mapping:")
    for label, idx in tyre_mapping.items():
        print(f"  {idx}: {label}")
    print("circuit_archetype LabelEncoder mapping:")
    for label, idx in archetype_mapping.items():
        print(f"  {idx}: {label}")
    unseen_tyre = int((df["TyreCompound_enc"] == -1).sum())
    unseen_arch = int((df["circuit_archetype_enc"] == -1).sum())
    print(f"Unseen TyreCompound labels mapped to -1: {unseen_tyre}")
    print(f"Unseen circuit_archetype labels mapped to -1: {unseen_arch}")

    print_header("Step 4 - Impute nulls in model features")
    print("Null counts before imputation:")
    for feature in MODEL_FEATURES:
        print(f"  {feature}: {int(df[feature].isna().sum())}")

    train_for_stats = df.loc[train_mask]
    field_laptime_std_mean = train_for_stats["field_laptime_std"].mean(skipna=True)
    df["field_laptime_std"] = df["field_laptime_std"].fillna(field_laptime_std_mean)
    print(f"Imputed field_laptime_std with training mean: {field_laptime_std_mean:.6f}")

    df["deg_rate_corrected"] = df["deg_rate_corrected"].fillna(0)
    print("Imputed deg_rate_corrected with 0")

    df["pit_window_delta"] = df["pit_window_delta"].fillna(0)
    print("Imputed pit_window_delta with 0")

    df["position_at_stake"] = df["position_at_stake"].fillna(0)
    print("Imputed position_at_stake with 0")

    pace_nulls = int(df["pace_differential"].isna().sum())
    if pace_nulls > 0:
        df["pace_differential"] = df["pace_differential"].fillna(0)
        print("Imputed pace_differential with 0")
    else:
        print("pace_differential has no nulls; no imputation applied")

    # Ensure all MODEL_FEATURES are fully usable for modeling by imputing any
    # remaining nulls from training-only statistics.
    remaining_before = {
        f: int(df[f].isna().sum()) for f in MODEL_FEATURES if int(df[f].isna().sum()) > 0
    }
    if remaining_before:
        print("Applying fallback imputations for remaining MODEL_FEATURES nulls:")
        for feature, count in remaining_before.items():
            train_feature = train_for_stats[feature]
            if pd.api.types.is_numeric_dtype(df[feature]):
                fill_value = train_feature.median(skipna=True)
                if pd.isna(fill_value):
                    fill_value = 0
            else:
                mode_vals = train_feature.mode(dropna=True)
                fill_value = mode_vals.iloc[0] if len(mode_vals) else -1
            df[feature] = df[feature].fillna(fill_value)
            print(f"  {feature}: filled {count} nulls using training fallback value {fill_value}")

    print("Null counts after imputation:")
    null_after = {}
    for feature in MODEL_FEATURES:
        count = int(df[feature].isna().sum())
        null_after[feature] = count
        print(f"  {feature}: {count}")
    if any(count != 0 for count in null_after.values()):
        raise ValueError("Not all MODEL_FEATURES are fully imputed to zero nulls")
    print("All MODEL_FEATURES null counts are 0")

    print_header("Step 5 - Cast boolean model features to int")
    bool_model_features = [
        "undercut_threat_flag",
        "overcut_threat_flag",
        "high_sc_circuit",
        "caution_active",
        "sc_active",
        "vsc_active",
        "yellow_flag",
        "wet_race_flag",
        "deg_rate_corrected_reliable",
    ]
    for col in bool_model_features:
        df[col] = df[col].fillna(False).astype(int)

    print("MODEL_FEATURES dtypes:")
    for feature in MODEL_FEATURES:
        print(f"  {feature}: {df[feature].dtype}")

    print_header("Step 6 - Drop null-target rows")
    before_rows = len(df)
    df = df[df["sc_vsc_next3"].notna()].copy()
    after_rows = len(df)
    rows_dropped = before_rows - after_rows
    print(f"Rows dropped where sc_vsc_next3 is null: {rows_dropped}")
    print(f"Remaining rows: {after_rows}")
    if rows_dropped != 3987 or after_rows != 70549:
        raise ValueError(
            f"Expected drop=3987 and remaining=70549, found drop={rows_dropped}, remaining={after_rows}"
        )
    print("Confirmed: dropped 3,987 rows and retained 70,549 rows")

    print_header("Step 7 - Final class balance report")
    overall_rate = df["sc_vsc_next3"].mean()
    print(f"Overall sc_vsc_next3 positive rate: {overall_rate:.4%}")

    print("Positive rate by Year:")
    year_rates = df.groupby("Year", sort=True)["sc_vsc_next3"].mean()
    for year, rate in year_rates.items():
        print(f"  {int(year)}: {rate:.4%}")

    print("Positive rate by circuit_archetype_enc:")
    archetype_name_by_code = {idx: label for label, idx in archetype_mapping.items()}
    archetype_name_by_code[-1] = "UNSEEN_OR_MISSING"
    archetype_rates = df.groupby("circuit_archetype_enc", sort=True)["sc_vsc_next3"].mean()
    for code, rate in archetype_rates.items():
        name = archetype_name_by_code.get(int(code), "UNSEEN_OR_MISSING")
        print(f"  {int(code)} ({name}): {rate:.4%}")

    print_header("Step 8 - Null audit")
    audit_cols = MODEL_FEATURES + ["sc_vsc_next3", "sc_vsc_next5"]
    for col in audit_cols:
        print(f"  {col}: {int(df[col].isna().sum())}")
    sc_vsc_next5_nulls = int(df["sc_vsc_next5"].isna().sum())
    print(f"sc_vsc_next5 null count is expected: {sc_vsc_next5_nulls}")

    print_header("Step 9 - Save")
    df.to_csv(output_path, index=False)
    print(f"Saved full DataFrame to: {output_path}")
    print(f"Final shape: {df.shape[0]} x {df.shape[1]}")

    feature_list_path.write_text("\n".join(MODEL_FEATURES) + "\n", encoding="utf-8")
    print(f"Saved MODEL_FEATURES list to: {feature_list_path}")
    print(f"MODEL_FEATURES saved count: {len(MODEL_FEATURES)}")


if __name__ == "__main__":
    main()