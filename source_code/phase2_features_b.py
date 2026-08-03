"""Phase 2 feature construction script B.

Loads phase2_features_a.csv, adds additional race-context features, and saves
to phase2_features_b.csv without overwriting the Phase 2 A file.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def print_header(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print(f"{'=' * 78}")


def cars_within_one_second_proxy(df: pd.DataFrame) -> pd.Series:
    """Count other cars within 1 second on the same lap via a vectorized self-merge."""
    lap_cols = ['Year', 'Round', 'LapNumber']
    work = df.loc[df['LapTime_s'].notna(), ['row_id'] + lap_cols + ['LapTime_s']].copy()
    left = work.rename(columns={'row_id': 'left_id', 'LapTime_s': 'left_lap_time'})
    right = work.rename(columns={'row_id': 'right_id', 'LapTime_s': 'right_lap_time'})

    merged = left.merge(right, on=lap_cols, how='left')
    within_window = (
        (merged['left_id'] != merged['right_id']) &
        ((merged['left_lap_time'] - merged['right_lap_time']).abs() <= 1.0)
    )
    counts = merged.loc[within_window].groupby('left_id').size()

    proxy = pd.Series(pd.NA, index=df.index, dtype='Int64')
    proxy.loc[work['row_id'].to_numpy()] = work['row_id'].map(counts).fillna(0).astype('Int64').to_numpy()
    return proxy


def main() -> None:
    root = Path(__file__).parent.parent
    input_path = root / 'data' / 'processed' / 'phase2_features_a.csv'
    output_path = root / 'data' / 'processed' / 'phase2_features_b.csv'

    print_header('Step 0 — Load and validate')
    df = pd.read_csv(input_path)
    df['sc_active'] = df['sc_active'].astype(bool)
    df['vsc_active'] = df['vsc_active'].astype(bool)
    df['caution_active'] = df['caution_active'].astype(bool)
    df['is_strategic_stop'] = df['is_strategic_stop'].astype(pd.BooleanDtype())
    print(f'Loaded shape: {df.shape[0]} x {df.shape[1]}')
    if df.shape != (74536, 62):
        raise ValueError(f'Expected shape (74536, 62), found {df.shape}')

    # Helper identifier to support the vectorized proxy merge.
    df['row_id'] = np.arange(len(df), dtype=np.int64)

    print_header('Step 1 — Fix field_std_tyre_age null')
    before_nulls = int(df['field_std_tyre_age'].isna().sum())
    print(f'field_std_tyre_age nulls before fix: {before_nulls:,}')
    df['field_std_tyre_age'] = df.groupby(['Year', 'Round', 'LapNumber'], sort=False)['field_std_tyre_age'].ffill()
    after_ffill_nulls = int(df['field_std_tyre_age'].isna().sum())
    if after_ffill_nulls > 0:
        global_mean = df['field_std_tyre_age'].mean(skipna=True)
        df['field_std_tyre_age'] = df['field_std_tyre_age'].fillna(global_mean)
    after_fill_nulls = int(df['field_std_tyre_age'].isna().sum())
    print(f'field_std_tyre_age nulls after forward-fill: {after_ffill_nulls:,}')
    print(f'field_std_tyre_age nulls after global-mean fill: {after_fill_nulls:,}')
    if after_fill_nulls != 0:
        raise ValueError('field_std_tyre_age still has nulls after fix')

    print_header('Step 2 — Build wet_race_flag')
    wet_race_flag = df.groupby(['Year', 'Round'], sort=False)['TyreCompound'].transform(
        lambda s: s.isin(['INTERMEDIATE', 'WET']).any()
    )
    df['wet_race_flag'] = wet_race_flag.astype(bool)
    total_true_rows = int(df['wet_race_flag'].sum())
    wet_groups = df.loc[df['wet_race_flag'], ['Year', 'Round']].drop_duplicates()
    flagged_events = (
        df.loc[df['wet_race_flag'], ['Year', 'Round', 'EventName']]
        .drop_duplicates()
        .sort_values(['Year', 'EventName'])
    )
    print(f'Total True rows: {total_true_rows:,}')
    print(f'Distinct Year/Round combinations flagged as wet: {len(wet_groups):,}')
    print('Flagged EventNames with Year:')
    for _, row in flagged_events.iterrows():
        print(f"  {int(row['Year'])} — {row['EventName']}")

    print_header('Step 3 — Build field_laptime_std')
    lap_group = df.groupby(['Year', 'Round', 'LapNumber'], sort=False)['LapTime_s']
    lap_count = lap_group.transform('count')
    field_laptime_std = lap_group.transform('std').where(lap_count >= 5)
    df['field_laptime_std'] = field_laptime_std
    print('field_laptime_std describe():')
    print(df['field_laptime_std'].describe().to_string())
    print(f'field_laptime_std null count: {int(df["field_laptime_std"].isna().sum()):,}')

    print_header('Step 4 — Build cars_within_1s_proxy')
    df['cars_within_1s_proxy'] = cars_within_one_second_proxy(df)
    print('cars_within_1s_proxy describe():')
    print(df['cars_within_1s_proxy'].describe().to_string())
    print(f'cars_within_1s_proxy null count: {int(df["cars_within_1s_proxy"].isna().sum()):,}')
    proxy_valid = df['cars_within_1s_proxy'].dropna()
    prop_ge_3 = ((proxy_valid >= 3).mean() * 100) if len(proxy_valid) else 0.0
    print(f'Proportion of rows where cars_within_1s_proxy >= 3: {prop_ge_3:.2f}%')

    print_header('Step 5 — Build sector_incident_rate')
    yellow_flag = (
        df['TrackStatus'].astype(str).str.contains('2', regex=False) &
        (~df['sc_active']) &
        (~df['vsc_active'])
    )
    df['yellow_flag'] = yellow_flag
    df['sector_incident_rate'] = df.groupby('EventName', sort=False)['yellow_flag'].transform('mean')
    circuit_rates = (
        df[['EventName', 'sector_incident_rate']]
        .drop_duplicates()
        .sort_values('sector_incident_rate', ascending=False)
    )
    print('Top 5 circuits by sector_incident_rate:')
    for _, row in circuit_rates.head(5).iterrows():
        print(f"  {row['EventName']}: {row['sector_incident_rate']:.4f}")
    print('Bottom 5 circuits by sector_incident_rate:')
    for _, row in circuit_rates.tail(5).iterrows():
        print(f"  {row['EventName']}: {row['sector_incident_rate']:.4f}")
    print(f'sector_incident_rate null count: {int(df["sector_incident_rate"].isna().sum()):,}')

    print_header('Step 6 — Build Year feature')
    df['year_encoded'] = df['Year'].astype(int)
    print('year_encoded value counts:')
    print(df['year_encoded'].value_counts().sort_index().to_string())

    print_header('Step 7 — Null audit')
    for col in [
        'field_std_tyre_age',
        'wet_race_flag',
        'field_laptime_std',
        'cars_within_1s_proxy',
        'sector_incident_rate',
        'year_encoded',
    ]:
        print(f'{col}: {int(df[col].isna().sum()):,}')

    print_header('Step 8 — Correlation check')
    train = df[(df['Year'] <= 2023) & df['sc_vsc_next3'].notna()].copy()
    corr_features = [
        'wet_race_flag',
        'field_laptime_std',
        'cars_within_1s_proxy',
        'sector_incident_rate',
        'year_encoded',
    ]
    corr_rows = []
    target = train['sc_vsc_next3'].astype(float)
    for feature in corr_features:
        series = train[feature]
        if pd.api.types.is_bool_dtype(series):
            x = series.astype(int)
        else:
            x = pd.to_numeric(series, errors='coerce')
        corr_frame = pd.concat([x, target], axis=1).dropna()
        corr_val = corr_frame.iloc[:, 0].corr(corr_frame.iloc[:, 1]) if len(corr_frame) > 1 else np.nan
        corr_rows.append({'feature': feature, 'pearson_corr': corr_val})
    corr_table = pd.DataFrame(corr_rows)
    corr_table['abs_corr'] = corr_table['pearson_corr'].abs()
    corr_table = corr_table.sort_values('abs_corr', ascending=False).drop(columns=['abs_corr'])
    print('Pearson correlation against sc_vsc_next3 (Year <= 2023, non-null target rows):')
    print(corr_table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print_header('Step 9 — Save')
    output_path = root / 'data' / 'processed' / 'phase2_features_b.csv'
    df.to_csv(output_path, index=False)
    print(f'Saved to: {output_path}')
    print(f'Final shape: {df.shape[0]} x {df.shape[1]}')
    print(f'Final column count: {len(df.columns)}')
    if len(df.columns) != 68:
        raise ValueError(f'Expected 68 columns, found {len(df.columns)}')


if __name__ == '__main__':
    main()