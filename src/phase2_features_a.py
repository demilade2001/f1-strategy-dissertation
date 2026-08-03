"""Phase 2 feature construction script A.

Builds forward-looking caution targets and supporting race-context features
from the processed Phase 1 dataset without modifying the original base_df.csv.
"""

from pathlib import Path

import pandas as pd


def print_header(title: str) -> None:
    print(f"\n{'=' * 78}")
    print(title)
    print(f"{'=' * 78}")


def build_forward_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Return 1/0 forward-looking caution target within each driver-race group.

    The final `horizon` laps in each group are set to NaN because there is not a
    full forward window available.
    """
    group_cols = ['Year', 'Round', 'Driver']
    shifted_cols = [
        df.groupby(group_cols, sort=False)['caution_active'].shift(-step)
        for step in range(1, horizon + 1)
    ]
    forward_stack = pd.concat(shifted_cols, axis=1)
    target = forward_stack.max(axis=1)

    group_sizes = df.groupby(group_cols, sort=False)['LapNumber'].transform('size')
    lap_index_in_group = df.groupby(group_cols, sort=False).cumcount() + 1
    valid_window = lap_index_in_group <= (group_sizes - horizon)
    target = target.where(valid_window)
    return target


def main() -> None:
    root = Path(__file__).parent.parent
    input_path = root / 'data' / 'processed' / 'base_df.csv'
    output_path = root / 'data' / 'processed' / 'phase2_features_a.csv'

    print_header('Step 0 — Load base_df')
    df = pd.read_csv(input_path)
    df['is_strategic_stop'] = df['is_strategic_stop'].astype(pd.BooleanDtype())
    df['sc_active'] = df['sc_active'].astype(bool)
    df['vsc_active'] = df['vsc_active'].astype(bool)
    print(f'Loaded shape: {df.shape[0]} x {df.shape[1]}')
    if df.shape != (74536, 55):
        raise ValueError(f'Expected shape (74536, 55), found {df.shape}')

    print_header('Step 1 — Build caution_active')
    df['caution_active'] = df['sc_active'] | df['vsc_active']
    caution_rate = df['caution_active'].mean() * 100
    print(f'caution_active positive rate: {caution_rate:.2f}%')

    print_header('Step 2 — Build forward-looking target variables')
    df = df.sort_values(['Year', 'Round', 'Driver', 'LapNumber']).reset_index(drop=True)
    df['sc_vsc_next3'] = build_forward_target(df, horizon=3)
    df['sc_vsc_next5'] = build_forward_target(df, horizon=5)

    for col in ['sc_vsc_next3', 'sc_vsc_next5']:
        non_null = int(df[col].notna().sum())
        nulls = int(df[col].isna().sum())
        positives = int((df[col] == 1).sum())
        rate = (positives / non_null * 100) if non_null else 0.0
        print(f'{col}:')
        print(f'  Non-null rows: {non_null:,}')
        print(f'  Null rows: {nulls:,}')
        print(f'  Positive count: {positives:,}')
        print(f'  Positive rate: {rate:.2f}%')

    print_header('Step 3 — Build yellow_flag')
    track_status_str = df['TrackStatus'].astype(str)
    df['yellow_flag'] = track_status_str.str.contains('2', regex=False) & (~df['sc_active']) & (~df['vsc_active'])
    yellow_rate = df['yellow_flag'].mean() * 100
    print(f'yellow_flag positive rate: {yellow_rate:.2f}%')
    top3_yellow = df.loc[df['yellow_flag']].groupby('EventName').size().sort_values(ascending=False).head(3)
    print('Top 3 EventNames by yellow_flag frequency:')
    for event_name, count in top3_yellow.items():
        print(f'  {event_name}: {int(count):,}')

    print_header('Step 4 — Build field tyre age features')
    lap_groups = df.groupby(['Year', 'Round', 'LapNumber'], sort=False)['TyreLife']
    df['field_mean_tyre_age'] = lap_groups.transform('mean')
    df['field_std_tyre_age'] = lap_groups.transform('std')
    print('field_mean_tyre_age describe():')
    print(df['field_mean_tyre_age'].describe().to_string())
    print(f'field_mean_tyre_age null count: {int(df["field_mean_tyre_age"].isna().sum()):,}')
    print('\nfield_std_tyre_age describe():')
    print(df['field_std_tyre_age'].describe().to_string())
    print(f'field_std_tyre_age null count: {int(df["field_std_tyre_age"].isna().sum()):,}')

    print_header('Step 5 — Build race_fraction')
    max_lap = df.groupby(['Year', 'Round'], sort=False)['LapNumber'].transform('max')
    df['race_fraction'] = df['LapNumber'] / max_lap
    print(df['race_fraction'].describe().to_string())
    print(f'race_fraction null count: {int(df["race_fraction"].isna().sum()):,}')

    print_header('Step 6 — Class balance diagnostic')
    for col in ['sc_vsc_next3', 'sc_vsc_next5']:
        non_null_df = df[df[col].notna()].copy()
        print(f'\n{col}:')
        overall_rate = non_null_df[col].mean() * 100
        print(f'  Positive rate overall: {overall_rate:.2f}%')

        by_year = non_null_df.groupby('Year')[col].mean().mul(100)
        print('  Positive rate by Year:')
        for year, rate in by_year.items():
            print(f'    {int(year)}: {rate:.2f}%')

        by_arch = non_null_df.groupby('circuit_archetype')[col].mean().mul(100)
        print('  Positive rate by circuit_archetype:')
        for archetype, rate in by_arch.items():
            print(f'    {archetype}: {rate:.2f}%')

        pivot = (
            non_null_df
            .pivot_table(index='Year', columns='circuit_archetype', values=col, aggfunc='mean')
            .mul(100)
            .round(2)
        )
        print('  Positive rate by Year x circuit_archetype (%):')
        print(pivot.to_string())

    print_header('Step 7 — Null audit')
    new_cols = [
        'caution_active',
        'sc_vsc_next3',
        'sc_vsc_next5',
        'yellow_flag',
        'field_mean_tyre_age',
        'field_std_tyre_age',
        'race_fraction',
    ]
    for col in new_cols:
        print(f'{col}: {int(df[col].isna().sum()):,}')

    print_header('Step 8 — Save')
    df.to_csv(output_path, index=False)
    print(f'Saved to: {output_path}')
    print(f'Final shape: {df.shape[0]} x {df.shape[1]}')
    print(f'Final column count: {len(df.columns)}')
    if len(df.columns) != 62:
        raise ValueError(f'Expected 62 columns, found {len(df.columns)}')


if __name__ == '__main__':
    main()