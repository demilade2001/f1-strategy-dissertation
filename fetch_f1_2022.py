import os
import argparse
import pandas as pd
import fastf1


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def lap_time_seconds(lap_row):
    lt = lap_row.get('LapTime')
    if pd.isna(lt):
        return None
    try:
        return lt.total_seconds()
    except Exception:
        return float(lt) if lt is not None else None


def get_tyre(lap_row):
    for key in ('TyreCompound', 'Compound', 'Tyre'):
        if key in lap_row and pd.notna(lap_row.get(key)):
            return lap_row.get(key)
    return None


def main(year: int, cache_dir: str, out_dir: str):
    fastf1.Cache.enable_cache(cache_dir)
    
    print(f"Fetching {year} F1 data...")
    schedule = fastf1.get_event_schedule(year)
    
    laps_rows = []
    pit_tel_rows = []
    
    for idx, (_, event) in enumerate(schedule.iterrows()):
        # Extract round number safely
        rnd = None
        for key in ('RoundNumber', 'round', 'Round'):
            val = event.get(key)
            if val is not None and pd.notna(val):
                try:
                    rnd = int(val)
                    break
                except (ValueError, TypeError):
                    pass
        
        if rnd is None:
            rnd = idx + 1
        
        event_name = event.get('EventName') or event.get('Event') or str(rnd)
        print(f"  Round {rnd}: {event_name}")
        
        try:
            session = fastf1.get_session(year, rnd, 'R')
            session.load()
        except Exception as e:
            print(f"    Error loading round {rnd}: {e}")
            continue
        
        laps_df = session.laps
        
        # Extract lap time + tyre data
        for _, lap in laps_df.iterrows():
            laps_rows.append({
                'Year': year,
                'Round': rnd,
                'EventName': event_name,
                'Session': 'R',
                'Driver': lap.get('Driver'),
                'CarNumber': lap.get('CarNumber'),
                'LapNumber': lap.get('LapNumber'),
                'LapTime_s': lap_time_seconds(lap),
                'TyreCompound': get_tyre(lap),
                'TyreLife': lap.get('TyreLife'),
                'PitInTime': lap.get('PitInTime'),
                'PitOutTime': lap.get('PitOutTime'),
            })
        
        # Extract telemetry for pit laps
        pit_laps = laps_df[laps_df['PitInTime'].notna() | laps_df['PitOutTime'].notna()]
        if len(pit_laps) == 0:
            continue
        
        for _, pit in pit_laps.iterrows():
            try:
                lap_idx = pit.name
                tel = session.laps.get_telemetry(lap_idx)
            except Exception:
                try:
                    driver_laps = session.laps[session.laps['Driver'] == pit.get('Driver')]
                    matched = driver_laps[driver_laps['LapNumber'] == pit.get('LapNumber')]
                    if not matched.empty:
                        tel = session.laps.get_telemetry(matched.index[0])
                    else:
                        tel = None
                except Exception:
                    tel = None
            
            if tel is None or tel.empty:
                continue
            
            tel = tel.reset_index()
            keep_cols = [c for c in ('SessionTime', 'Time', 'X', 'Y', 'Speed', 'Throttle', 'Brake', 'nGear', 'ERPM') if c in tel.columns]
            
            for _, row in tel[keep_cols].iterrows():
                r = {
                    'Year': year,
                    'Round': rnd,
                    'EventName': event_name,
                    'Session': 'R',
                    'Driver': pit.get('Driver'),
                    'CarNumber': pit.get('CarNumber'),
                    'LapNumber': pit.get('LapNumber'),
                }
                for c in keep_cols:
                    r[c] = row.get(c)
                pit_tel_rows.append(r)
    
    ensure_dir(out_dir)
    laps_df_out = pd.DataFrame(laps_rows)
    pit_tel_df_out = pd.DataFrame(pit_tel_rows)
    
    laps_csv = os.path.join(out_dir, f'f1_{year}_laps_tyres.csv')
    pit_csv = os.path.join(out_dir, f'f1_{year}_pit_telemetry.csv')
    
    laps_df_out.to_csv(laps_csv, index=False)
    pit_tel_df_out.to_csv(pit_csv, index=False)
    
    print(f'\nSuccess!')
    print(f'Laps+tyre data: {laps_csv}')
    print(f'Pit telemetry: {pit_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch F1 lap/tyre/pit telemetry via FastF1')
    parser.add_argument('--year', type=int, default=2022)
    parser.add_argument('--cache', type=str, default='fastf1_cache')
    parser.add_argument('--out', type=str, default='data')
    args = parser.parse_args()
    main(args.year, args.cache, args.out)
