import argparse
import os
from pathlib import Path

import fastf1
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_get(row: pd.Series, key: str):
    if key not in row.index:
        return None
    value = row.get(key)
    if pd.isna(value):
        return None
    return value


def timedelta_seconds(value):
    if value is None:
        return None
    try:
        return float(value.total_seconds())
    except Exception:
        return None


def extract_lap_row(year, round_number, event_name, lap):
    return {
        'Year': year,
        'Round': round_number,
        'EventName': event_name,
        'Driver': safe_get(lap, 'Driver'),
        'CarNumber': safe_get(lap, 'CarNumber'),
        'LapNumber': safe_get(lap, 'LapNumber'),
        'LapTime_s': timedelta_seconds(safe_get(lap, 'LapTime')),
        'Sector1_s': timedelta_seconds(safe_get(lap, 'Sector1Time')),
        'Sector2_s': timedelta_seconds(safe_get(lap, 'Sector2Time')),
        'Sector3_s': timedelta_seconds(safe_get(lap, 'Sector3Time')),
        'SpeedI1': safe_get(lap, 'SpeedI1'),
        'SpeedI2': safe_get(lap, 'SpeedI2'),
        'SpeedFL': safe_get(lap, 'SpeedFL'),
        'TyreCompound': safe_get(lap, 'Compound') or safe_get(lap, 'TyreCompound'),
        'TyreLife': safe_get(lap, 'TyreLife'),
        'Stint': safe_get(lap, 'Stint'),
        'FreshTyre': safe_get(lap, 'FreshTyre'),
        'Position': safe_get(lap, 'Position'),
        'TrackStatus': safe_get(lap, 'TrackStatus'),
        'IsAccurate': safe_get(lap, 'IsAccurate'),
        'PitInTime': safe_get(lap, 'PitInTime'),
        'PitOutTime': safe_get(lap, 'PitOutTime'),
        'PitTime': safe_get(lap, 'PitTime'),
    }


def derive_pit_events(laps_df: pd.DataFrame) -> pd.DataFrame:
    pit_laps = laps_df[laps_df['PitInTime'].notna()].copy()
    if pit_laps.empty or 'PitInTime' not in laps_df.columns:
        return pd.DataFrame(
            columns=[
                'Year', 'Round', 'EventName', 'Driver', 'LapNumber', 'Stint', 'StintOut',
                'TyreCompoundOld', 'TyreCompoundNew', 'TyreLifeAtStop', 'PitDuration_s',
                'TrackStatusAtStop', 'PositionAtStop',
            ]
        )

    pit_rows = []
    group_keys = ['Year', 'Round', 'EventName', 'Driver']
    for group_values, driver_laps in laps_df.groupby(group_keys, sort=False):
        if driver_laps.empty:
            continue
        driver_laps = driver_laps.sort_values(['LapNumber', 'Stint']).reset_index(drop=True)
        driver_pit_laps = driver_laps[driver_laps['PitInTime'].notna()]
        for _, pit in driver_pit_laps.iterrows():
            lap_number = pit['LapNumber']
            stint = pit['Stint']
            prev_lap = driver_laps[driver_laps['LapNumber'] == lap_number - 1]
            next_lap = driver_laps[(driver_laps['Stint'] == stint + 1) & (driver_laps['LapNumber'] > lap_number)]
            if next_lap.empty:
                next_lap = driver_laps[driver_laps['LapNumber'] == lap_number + 1]
            previous_compound = None
            if not prev_lap.empty:
                previous_compound = safe_get(prev_lap.iloc[-1], 'TyreCompound') or safe_get(prev_lap.iloc[-1], 'Compound')
            new_compound = None
            if not next_lap.empty:
                new_compound = safe_get(next_lap.iloc[0], 'TyreCompound') or safe_get(next_lap.iloc[0], 'Compound')
            pit_rows.append({
                'Year': pit['Year'],
                'Round': pit['Round'],
                'EventName': pit['EventName'],
                'Driver': pit['Driver'],
                'LapNumber': lap_number,
                'Stint': stint,
                'StintOut': (stint + 1) if stint is not None else None,
                'TyreCompoundOld': previous_compound,
                'TyreCompoundNew': new_compound,
                'TyreLifeAtStop': pit['TyreLife'],
                'PitDuration_s': timedelta_seconds(pit['PitTime']),
                'TrackStatusAtStop': pit['TrackStatus'],
                'PositionAtStop': pit['Position'],
            })
    return pd.DataFrame(pit_rows)


def find_round_number(event: pd.Series, fallback_index: int) -> int:
    for key in ('RoundNumber', 'round', 'Round'):
        value = event.get(key)
        if value is not None and not pd.isna(value):
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
    return fallback_index + 1


def find_event_name(event: pd.Series, round_number: int) -> str:
    for key in ('EventName', 'Event', 'Name', 'EventFullName'):
        value = event.get(key)
        if value is not None and not pd.isna(value):
            return str(value)
    return f'Round {round_number}'


def main(year: int, cache_dir: str, out_dir: str):
    cache_path = Path(cache_dir)
    ensure_dir(cache_path)
    fastf1.Cache.enable_cache(str(cache_path))

    output_base = Path(out_dir) / 'raw' / str(year)
    ensure_dir(output_base)

    print(f'Fetching {year} F1 race lap and pit data...')
    schedule = fastf1.get_event_schedule(year)
    if schedule is None or schedule.empty:
        raise RuntimeError(f'No event schedule found for {year}')

    schedule = schedule[schedule['RoundNumber'] >= 1].reset_index(drop=True)
    laps_rows = []
    failed_rounds = []
    extracted_rounds = 0

    for idx, (_, event) in enumerate(schedule.iterrows()):
        round_number = find_round_number(event, idx)
        event_name = find_event_name(event, round_number)
        print(f'Round {round_number}: {event_name}')

        try:
            session = fastf1.get_session(year, round_number, 'R')
            session.load(laps=True, telemetry=False, weather=False, messages=False)
        except Exception as exc:
            error_message = str(exc)
            failed_rounds.append((round_number, event_name, error_message))
            print(f'  ERROR loading round {round_number}: {error_message}')
            continue

        laps_df = session.laps
        if laps_df is None or laps_df.empty:
            print(f'  SKIP round {round_number}: no laps available')
            continue

        row_count = 0
        for _, lap in laps_df.iterrows():
            lap_row = extract_lap_row(year, round_number, event_name, lap)
            laps_rows.append(lap_row)
            row_count += 1

        extracted_rounds += 1
        print(f'  Round {round_number}: {event_name} — {row_count} laps extracted')

    laps_df_out = pd.DataFrame(laps_rows)
    if laps_df_out.empty:
        print('No lap rows were extracted for the season.')
    else:
        laps_csv = output_base / f'f1_{year}_laps.csv'
        laps_df_out.to_csv(laps_csv, index=False)
        print(f'  Saved laps file: {laps_csv}')

        pits_df_out = derive_pit_events(laps_df_out)
        pits_csv = output_base / f'f1_{year}_pits.csv'
        pits_df_out.to_csv(pits_csv, index=False)
        print(f'  Saved pits file: {pits_csv}')

    pit_count = 0
    if not laps_df_out.empty:
        pit_count = len(derive_pit_events(laps_df_out))

    print(f'Finished {year}: {extracted_rounds} rounds with lap data extracted.')
    if failed_rounds:
        print('Failed rounds:')
        for rnd, name, err in failed_rounds:
            print(f'  Round {rnd}: {name} — {err}')

    return {
        'year': year,
        'rounds_extracted': extracted_rounds,
        'lap_count': len(laps_rows),
        'pit_count': pit_count,
        'failures': failed_rounds,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch F1 season lap and pit stop data with FastF1')
    parser.add_argument('--year', type=int, required=True, help='Season year to fetch')
    parser.add_argument('--cache', type=str, default='./fastf1_cache', help='FastF1 cache directory')
    parser.add_argument('--out', type=str, default='./data', help='Output base directory')
    args = parser.parse_args()
    main(args.year, args.cache, args.out)
