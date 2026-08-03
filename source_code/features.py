"""Feature engineering utilities for building base DataFrames from raw F1 data."""
import pandas as pd
import numpy as np
from pathlib import Path
from source_code.utils import get_constructor, decode_track_status, MIDFIELD_CONSTRUCTORS


def build_base_df():
    """Load, process, and combine all F1 laps and pits data for 2022-2024.
    
    Steps:
    1. Load laps/pits CSVs for 2022, 2023, 2024
    2. Add Team column to laps using get_constructor() (drop rows that fail)
    3. Decode TrackStatus into sc_active and vsc_active boolean columns
    4. Investigate and handle TyreCompound null rows
    5. Derive PitDuration_s for pits from laps data (consecutive lap method)
    6. Concatenate all years and add is_midfield and compound_known flags
    7. Save to data/processed/base_df.csv
    
    Returns:
        pd.DataFrame: Combined base dataframe with all processing applied
    """
    root = Path(__file__).parent.parent
    data_dir = root / 'data'
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    all_laps = []
    all_pits = []
    pit_duration_derived_by_year = {}
    compound_null_by_year = {}
    strategic_validation_by_year = {}
    
    # Process each year
    for year in [2022, 2023, 2024]:
        print(f'\n{"="*70}')
        print(f'Processing {year}...')
        print(f'{"="*70}')
        
        # Load CSVs
        laps_path = raw_dir / str(year) / f'f1_{year}_laps.csv'
        pits_path = raw_dir / str(year) / f'f1_{year}_pits.csv'
        
        laps = pd.read_csv(laps_path)
        pits = pd.read_csv(pits_path)

        if 'Year' not in laps.columns:
            laps['Year'] = year
        if 'Year' not in pits.columns:
            pits['Year'] = year
        
        print(f'\nLaps: loaded {len(laps)} rows')
        print(f'Pits: loaded {len(pits)} rows')
        
        # ===== PROCESS LAPS FILE =====
        
        # Step 1: Add Team column using get_constructor()
        print(f'\nAdding Team column via get_constructor()...')
        dropped_rows = []  # Track rows that fail constructor mapping
        team_values = []
        
        for idx, row in laps.iterrows():
            try:
                team = get_constructor(row['Driver'], year, int(row['Round']))
                team_values.append(team)
            except ValueError as e:
                team_values.append(None)
                dropped_rows.append({
                    'Driver': row['Driver'],
                    'Round': int(row['Round']),
                    'LapNumber': row['LapNumber'],
                    'Error': str(e)
                })
        
        laps['Team'] = team_values
        laps_before_drop = len(laps)
        
        # Print dropped rows breakdown (Fix 3)
        if dropped_rows:
            print(f'\nBreakdown of {len(dropped_rows)} dropped rows:')
            dropped_df = pd.DataFrame(dropped_rows)
            for _, row in dropped_df.iterrows():
                print(f'  Driver {row["Driver"]:3s} Round {int(row["Round"]):2d}: {row["Error"]}')
        
        # Drop rows with no Team (constructor mapping failed)
        laps = laps[laps['Team'].notna()].reset_index(drop=True)
        print(f'Dropped {len(dropped_rows)} rows where get_constructor() failed')
        print(f'Laps after Team filtering: {len(laps)} rows')
        
        # Step 2: Decode TrackStatus
        print(f'\nDecoding TrackStatus...')
        track_status_decoded = laps['TrackStatus'].apply(decode_track_status)
        decoded_df = pd.DataFrame(list(track_status_decoded))
        
        # Extract sc_active and vsc_active
        laps['sc_active'] = decoded_df['TrackStatus_sc']
        laps['vsc_active'] = decoded_df['TrackStatus_vsc']

        # Step 2.5: Build is_strategic_stop from native pits compounds
        print(f'\nBuilding is_strategic_stop from native pits TyreCompoundOld/TyreCompoundNew...')

        pit_merge_keys = ['Year', 'Round', 'Driver', 'LapNumber']
        required_pits_cols = pit_merge_keys + ['TyreCompoundOld', 'TyreCompoundNew']
        missing_pits_cols = [col for col in required_pits_cols if col not in pits.columns]
        if missing_pits_cols:
            raise ValueError(
                'Missing required pits columns for is_strategic_stop: '
                + ', '.join(missing_pits_cols)
            )

        pit_native = pits[required_pits_cols].copy()
        lap_context = laps[pit_merge_keys + ['sc_active', 'vsc_active']].copy()

        pit_eval = pit_native.merge(
            lap_context,
            on=pit_merge_keys,
            how='left',
            indicator=True
        )

        pit_eval['same_compound'] = (
            pit_eval['TyreCompoundOld'].notna() &
            pit_eval['TyreCompoundNew'].notna() &
            (pit_eval['TyreCompoundOld'] == pit_eval['TyreCompoundNew'])
        )
        with pd.option_context('future.no_silent_downcasting', True):
            sc_active_bool = pit_eval['sc_active'].infer_objects(copy=False).fillna(False).astype(bool)
            vsc_active_bool = pit_eval['vsc_active'].infer_objects(copy=False).fillna(False).astype(bool)
        pit_eval['under_caution'] = (
            sc_active_bool |
            vsc_active_bool
        )
        pit_eval['is_strategic_stop'] = pd.Series(pd.NA, index=pit_eval.index, dtype='boolean')
        same_compound_mask = pit_eval['same_compound']
        pit_eval.loc[same_compound_mask, 'is_strategic_stop'] = (
            pit_eval.loc[same_compound_mask, 'under_caution'].astype('boolean')
        )

        # Assign only on pit-in lap rows; non-pit laps remain NaN.
        laps['is_strategic_stop'] = np.nan
        strategic_assign = pit_eval[pit_merge_keys + ['is_strategic_stop']].drop_duplicates(pit_merge_keys)
        laps = laps.merge(strategic_assign, on=pit_merge_keys, how='left', suffixes=('', '_pit'))
        laps['is_strategic_stop'] = laps['is_strategic_stop_pit']
        laps = laps.drop(columns=['is_strategic_stop_pit'])

        # Keep diagnostics on pits table too.
        pits = pits.merge(
            pit_eval[pit_merge_keys + ['same_compound', 'under_caution', 'is_strategic_stop', '_merge']],
            on=pit_merge_keys,
            how='left'
        )

        total_events_year = len(pit_eval)
        excluded_year = (pit_eval['is_strategic_stop'] == False).sum()
        retained_year = (pit_eval['is_strategic_stop'] == True).sum()
        merge_failed_year = (pit_eval['_merge'] != 'both').sum()

        retained_mask = pit_eval['is_strategic_stop'] == True
        with pd.option_context('future.no_silent_downcasting', True):
            sc_mask = pit_eval['sc_active'].infer_objects(copy=False).fillna(False).astype(bool)
            vsc_mask = pit_eval['vsc_active'].infer_objects(copy=False).fillna(False).astype(bool)

        retained_sc_year = (retained_mask & sc_mask).sum()
        retained_vsc_year = (retained_mask & (~sc_mask) & vsc_mask).sum()
        retained_compound_only_year = (
            retained_mask &
            (~pit_eval['under_caution']) &
            (~pit_eval['same_compound'])
        ).sum()
        retained_under_caution_year = (retained_mask & pit_eval['under_caution']).sum()

        strategic_validation_by_year[year] = {
            'total': int(total_events_year),
            'excluded': int(excluded_year),
            'retained': int(retained_year),
            'retained_sc': int(retained_sc_year),
            'retained_vsc': int(retained_vsc_year),
            'retained_compound_only': int(retained_compound_only_year),
            'retained_under_caution': int(retained_under_caution_year),
            'merge_failed': int(merge_failed_year),
        }
        
        # Step 3: Investigate and handle TyreCompound nulls
        print(f'\nInvestigating TyreCompound nulls...')
        tyre_nulls_before = laps['TyreCompound'].isna().sum()
        print(f'Total TyreCompound nulls before forward-fill: {tyre_nulls_before}')
        
        if year == 2022:
            # Only for 2022, do detailed analysis
            tyre_nulls = laps[laps['TyreCompound'].isna()].copy()
            
            # Analyze null patterns
            pit_in_laps = tyre_nulls[tyre_nulls['PitInTime'].notna()]
            pit_out_laps = tyre_nulls[tyre_nulls['PitOutTime'].notna()]
            inaccurate_laps = tyre_nulls[tyre_nulls['IsAccurate'] == False]
            
            print(f'  - Pit-in laps: {len(pit_in_laps)}')
            print(f'  - Pit-out laps: {len(pit_out_laps)}')
            print(f'  - IsAccurate=False: {len(inaccurate_laps)}')
            
            # Find laps with no obvious explanation
            unexplained = tyre_nulls[
                (tyre_nulls['PitInTime'].isna()) &
                (tyre_nulls['PitOutTime'].isna()) &
                (tyre_nulls['IsAccurate'] != False)
            ]
            print(f'  - No obvious explanation: {len(unexplained)}')
        
        # Forward-fill TyreCompound within stint groups (Driver + Round + Stint)
        print(f'Forward-filling TyreCompound within stint groups...')
        laps['TyreCompound'] = laps.groupby(['Driver', 'Round', 'Stint'])['TyreCompound'].ffill()
        
        remaining_nulls = laps['TyreCompound'].isna().sum()
        print(f'TyreCompound nulls remaining after forward-fill: {remaining_nulls}')
        compound_null_by_year[year] = remaining_nulls
        
        # Step 4: Derive PitDuration_s from consecutive laps (Fix 1)
        print(f'\nDerive PitDuration_s from consecutive laps...')
        
        # Sort laps by Driver, Round, LapNumber for consecutive lookup
        laps_sorted = laps.sort_values(by=['Driver', 'Round', 'LapNumber']).reset_index(drop=True)
        
        # Create a mapping of pit-in laps to pit-out times
        pit_durations = {}  # key: (Driver, Round, LapNumber from pits), value: duration_s
        
        for idx in range(len(laps_sorted) - 1):
            current_row = laps_sorted.iloc[idx]
            next_row = laps_sorted.iloc[idx + 1]
            
            # Check if current lap has PitInTime and next lap is for same driver/round
            if (pd.notna(current_row['PitInTime']) and 
                current_row['Driver'] == next_row['Driver'] and 
                current_row['Round'] == next_row['Round'] and
                pd.notna(next_row['PitOutTime'])):
                
                try:
                    # Convert timedelta strings to seconds
                    pit_in_td = pd.to_timedelta(current_row['PitInTime'])
                    pit_out_td = pd.to_timedelta(next_row['PitOutTime'])
                    duration = (pit_out_td - pit_in_td).total_seconds()
                    
                    # Store with current lap's LapNumber (the pit-in lap)
                    key = (current_row['Driver'], int(current_row['Round']), int(current_row['LapNumber']))
                    pit_durations[key] = duration
                except (TypeError, ValueError):
                    pass
        
        print(f'Found {len(pit_durations)} pit-in laps with corresponding pit-out times')
        
        # ===== PROCESS PITS FILE =====
        
        print(f'\nMatching pit durations to pits file...')
        
        derived_durations = []
        pit_duration_derived = []
        
        for _, pit_row in pits.iterrows():
            driver = pit_row['Driver']
            round_num = int(pit_row['Round'])
            lap_num = int(pit_row['LapNumber'])
            
            key = (driver, round_num, lap_num)
            
            if key in pit_durations:
                derived_durations.append(pit_durations[key])
                pit_duration_derived.append(True)
            else:
                derived_durations.append(None)
                pit_duration_derived.append(False)
        
        pits['PitDuration_s'] = derived_durations
        pits['pit_duration_derived'] = pit_duration_derived
        
        derived_count = sum(pit_duration_derived)
        pit_duration_derived_by_year[year] = derived_count
        print(f'Successfully derived {derived_count} pit durations')
        
        all_laps.append(laps)
        all_pits.append(pits)
    
    # ===== CONCATENATE ALL YEARS =====
    
    print(f'\n{"="*70}')
    print('Concatenating all years...')
    print(f'{"="*70}')
    
    df_laps = pd.concat(all_laps, ignore_index=True)
    df_pits = pd.concat(all_pits, ignore_index=True)
    
    print(f'\nCombined laps: {len(df_laps)} rows')
    print(f'Combined pits: {len(df_pits)} rows')
    
    # ===== ADD COMPOUND_KNOWN FLAG (Fix 2) =====
    
    print(f'\nAdding compound_known flag...')
    df_laps['compound_known'] = df_laps['TyreCompound'].notna()
    
    print(f'Compound_known = False (nulls) per year:')
    for year in [2022, 2023, 2024]:
        year_nulls = compound_null_by_year.get(year, 0)
        year_total = len(df_laps[df_laps['Year'] == year])
        pct = (year_nulls / year_total) * 100 if year_total > 0 else 0
        print(f'  {year}: {year_nulls:4d} nulls ({pct:5.2f}%)')
    
    # ===== ADD MIDFIELD FLAG =====
    
    print(f'\nAdding is_midfield flag...')
    midfield_constructors = set(MIDFIELD_CONSTRUCTORS)
    df_laps['is_midfield'] = df_laps['Team'].isin(midfield_constructors)
    midfield_count = df_laps['is_midfield'].sum()
    big_three_count = (~df_laps['is_midfield']).sum()
    print(f'Midfield rows: {midfield_count:,}')
    print(f'Big Three rows: {big_three_count:,}')
    
    # ===== SAVE BASE DATAFRAME =====
    
    print(f'\n{"="*70}')
    print('Saving to data/processed/base_df.csv...')
    print(f'{"="*70}')
    
    output_path = processed_dir / 'base_df.csv'
    df_laps.to_csv(output_path, index=False)
    
    print(f'\nFinal DataFrame shape: {df_laps.shape}')
    
    # ===== PIT DURATION DERIVATION SUMMARY =====
    
    print(f'\nPit Duration Derivation Summary (Fix 1):')
    total_derived = sum(pit_duration_derived_by_year.values())
    for year in [2022, 2023, 2024]:
        derived_count = pit_duration_derived_by_year.get(year, 0)
        pits_count = len(df_pits[df_pits['Year'] == year])
        pct = (derived_count / pits_count) * 100 if pits_count > 0 else 0
        print(f'  {year}: {derived_count:4d} / {pits_count:4d} pit stops ({pct:5.1f}%)')
    print(f'  Total: {total_derived} pit durations successfully derived')

    # ===== STRATEGIC STOP VALIDATION =====

    print(f'\nStrategic stop validation (native pits compounds):')
    total_events = sum(v['total'] for v in strategic_validation_by_year.values())
    excluded_total = sum(v['excluded'] for v in strategic_validation_by_year.values())
    retained_total = sum(v['retained'] for v in strategic_validation_by_year.values())
    retained_sc_total = sum(v['retained_sc'] for v in strategic_validation_by_year.values())
    retained_vsc_total = sum(v['retained_vsc'] for v in strategic_validation_by_year.values())
    retained_compound_only_total = sum(v['retained_compound_only'] for v in strategic_validation_by_year.values())
    retained_under_caution_total = sum(v['retained_under_caution'] for v in strategic_validation_by_year.values())
    merge_failed_total = sum(v['merge_failed'] for v in strategic_validation_by_year.values())

    print(f'  Total pit stop events identified: {total_events:,}')
    print(f'  is_strategic_stop = False (excluded): {excluded_total:,}')
    print(f'  is_strategic_stop = True (retained): {retained_total:,}')
    print('  Retained stop breakdown:')
    print(f'    Under SC: {retained_sc_total:,}')
    print(f'    Under VSC: {retained_vsc_total:,}')
    print(f'    Compound change alone: {retained_compound_only_total:,}')
    print(f'  Pit stop events failing merge against pit-in lap rows: {merge_failed_total:,}')

    delta_excluded = excluded_total - 368
    delta_retained_caution = retained_under_caution_total - 204
    print(f'  Delta vs expected excluded (368): {delta_excluded:+,}')
    print(f'  Delta vs expected retained-under-caution (204): {delta_retained_caution:+,}')

    # ===== ADDITIONAL DIAGNOSTICS =====

    print(f'\nDiagnostic 2 — Native TyreCompoundOld/TyreCompoundNew audit:')
    for col in ['TyreCompoundOld', 'TyreCompoundNew']:
        null_count = df_pits[col].isna().sum() if col in df_pits.columns else 0
        print(f'  {col}: nulls={null_count:,}')
        if col in df_pits.columns:
            print(f'  {col} value distribution (including NaN):')
            print(df_pits[col].value_counts(dropna=False).to_string())

    print(f'\nDiagnostic 3 — Pit stop clustering during caution periods:')
    current_same_compound_under_caution = df_pits[
        (df_pits['same_compound'] == True) &
        (df_pits['under_caution'] == True)
    ].copy()
    cluster_counts = current_same_compound_under_caution.groupby(['Year', 'Round', 'LapNumber']).size()
    clustered_stops = int((cluster_counts > 1).sum())
    clustered_stop_rows = int((cluster_counts[cluster_counts > 1]).sum())
    total_same_compound_under_caution = len(current_same_compound_under_caution)
    print(f'  Same-compound-under-caution stops: {total_same_compound_under_caution:,}')
    print(f'  Stops on a Year/Round/Lap shared with at least one other pit stop: {clustered_stop_rows:,}')
    print(f'  Distinct clustered Year/Round/Lap events: {clustered_stops:,}')

    print(f'\nDiagnostic 4 — Failed merge investigation:')
    failed_merges = df_pits[df_pits['_merge'] != 'both'][pit_merge_keys].copy()
    print(f'  Failed merge count: {len(failed_merges):,}')
    if len(failed_merges) > 0:
        raw_lap_keys = set(map(tuple, df_laps[pit_merge_keys].itertuples(index=False, name=None)))
        for _, row in failed_merges.iterrows():
            key = (int(row['Year']), int(row['Round']), str(row['Driver']), int(row['LapNumber']))
            reason = 'No matching pit-in lap row in processed laps'
            if key in raw_lap_keys:
                reason = 'Pit-in lap existed in laps CSV but was removed earlier in processing (likely constructor/Team filtering or downstream row drop)'
            print(f"  {int(row['Year'])} R{int(row['Round'])} {row['Driver']} Lap {int(row['LapNumber'])} -> {reason}")
    
    print(f'\nNull summary for core columns:')
    core_cols = ['Year', 'Round', 'Driver', 'Team', 'LapNumber', 'TyreCompound', 
                 'TrackStatus', 'IsAccurate', 'PitInTime', 'PitOutTime', 'sc_active', 'vsc_active', 'compound_known']
    for col in core_cols:
        if col in df_laps.columns:
            null_count = df_laps[col].isna().sum()
            null_pct = (null_count / len(df_laps)) * 100
            print(f'  {col:20s}: {null_count:6d} nulls ({null_pct:5.2f}%)')
    
    print(f'\nProcessing complete!')
    
    return df_laps


def build_tyre_degradation(df):
    """Add tyre degradation rate (deg_rate) to the base DataFrame.
    
    Fits polynomial models to tyre life vs lap time for each compound/circuit/season
    combination. The degradation rate is the first-order coefficient indicating 
    lap time loss per additional lap of tyre age.
    
    Steps:
    1. Filter to clean, eligible laps (no wet/intermediate, SC/VSC, pit events, etc.)
    2. Fit linear and quadratic models per compound/circuit/season group
    3. Select better model and extract first-order coefficient as deg_rate
    4. Assign deg_rate to all rows in each group (even non-eligible ones)
    5. Perform sanity checks and save updated DataFrame
    
    Returns:
        pd.DataFrame: Input DataFrame with new deg_rate column
    """
    print(f'\n{"="*70}')
    print('Building tyre degradation rates...')
    print(f'{"="*70}')
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # ===== STEP 1: CREATE ELIGIBILITY MASK =====
    
    print(f'\nStep 1: Creating eligibility mask for clean laps...')
    
    # Start with all True
    deg_eligible = pd.Series(True, index=df.index)
    
    # Remove rows that don't meet eligibility criteria
    deg_eligible &= (df['compound_known'] == True)
    deg_eligible &= (df['IsAccurate'] == True)
    deg_eligible &= (df['sc_active'] == False)
    deg_eligible &= (df['vsc_active'] == False)
    deg_eligible &= (df['PitInTime'].isna())
    deg_eligible &= (df['PitOutTime'].isna())
    
    # Remove wet compounds (INTERMEDIATE, WET)
    wet_compounds = {'INTERMEDIATE', 'WET'}
    deg_eligible &= ~df['TyreCompound'].isin(wet_compounds)
    
    # Remove outlier laps that exceed 107% of median lap time per Driver/Round
    for (driver, round_num), group in df[deg_eligible].groupby(['Driver', 'Round']):
        median_time = group['LapTime_s'].median()
        threshold = median_time * 1.07
        outlier_indices = group[group['LapTime_s'] > threshold].index
        deg_eligible.loc[outlier_indices] = False
    
    eligible_count = deg_eligible.sum()
    print(f'Eligible laps: {eligible_count:,} / {len(df):,} ({eligible_count/len(df)*100:.1f}%)')
    
    # ===== METHODOLOGY NOTE =====
    # deg_rate is computed on raw LapTime_s and therefore includes fuel load contribution (~0.03 s/kg).
    # Negative values on low-degradation compounds (HARD) and low-deg circuits are expected artefacts
    # of this fuel load effect. deg_rate will be recomputed on fuel-corrected lap times after Feature 5
    # is built. deg_rate_reliable = False flags these rows for exclusion in Feature 2 modeling.
    # ========================
    
    # ===== STEP 2: FIT DEGRADATION CURVES =====
    
    print(f'\nStep 2: Fitting degradation curves per compound/circuit/season...')
    
    deg_rates = {}  # key: (TyreCompound, EventName, Year), value: deg_rate
    group_stats = {}  # for detailed reporting
    linear_count = 0
    quadratic_count = 0
    insufficient_count = 0
    
    # Group by TyreCompound, EventName, Year
    for (compound, event, year), group in df[deg_eligible].groupby(['TyreCompound', 'EventName', 'Year']):
        key = (compound, event, year)
        
        # Need at least 5 points to fit
        if len(group) < 5:
            deg_rates[key] = np.nan
            group_stats[key] = {
                'n_laps': len(group),
                'model': 'insufficient',
                'r2_linear': np.nan,
                'r2_quadratic': np.nan,
                'deg_rate': np.nan
            }
            insufficient_count += 1
            continue
        
        # Prepare data
        x = group['TyreLife'].values.astype(float)
        y = group['LapTime_s'].values.astype(float)
        
        # Skip if any NaN values
        if np.isnan(x).any() or np.isnan(y).any():
            deg_rates[key] = np.nan
            group_stats[key] = {
                'n_laps': len(group),
                'model': 'invalid_data',
                'r2_linear': np.nan,
                'r2_quadratic': np.nan,
                'deg_rate': np.nan
            }
            continue
        
        # Fit linear model (degree 1)
        try:
            coeffs_linear = np.polyfit(x, y, 1)
            y_pred_linear = np.polyval(coeffs_linear, x)
            ss_res_linear = np.sum((y - y_pred_linear) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0
        except (np.linalg.LinAlgError, ValueError):
            r2_linear = np.nan
            coeffs_linear = None
        
        # Fit quadratic model (degree 2)
        try:
            coeffs_quadratic = np.polyfit(x, y, 2)
            y_pred_quadratic = np.polyval(coeffs_quadratic, x)
            ss_res_quadratic = np.sum((y - y_pred_quadratic) ** 2)
            r2_quadratic = 1 - (ss_res_quadratic / ss_tot) if ss_tot > 0 else 0
        except (np.linalg.LinAlgError, ValueError):
            r2_quadratic = np.nan
            coeffs_quadratic = None
        
        # Select model: quadratic if it improves R² by more than 0.02
        if (not np.isnan(r2_quadratic) and not np.isnan(r2_linear) and 
            r2_quadratic > r2_linear + 0.02):
            # Use quadratic: deg_rate is the derivative at mean TyreLife
            # For poly ax² + bx + c, derivative is 2ax + b
            selected_coeffs = coeffs_quadratic
            a = coeffs_quadratic[0]
            b = coeffs_quadratic[1]
            mean_tyre_life = x.mean()
            deg_rate = 2 * a * mean_tyre_life + b  # derivative at mean
            model_type = 'quadratic'
            quadratic_count += 1
        else:
            # Use linear: coefficient of x
            selected_coeffs = coeffs_linear
            if coeffs_linear is not None:
                deg_rate = coeffs_linear[0]  # coefficient of x
            else:
                deg_rate = np.nan
            model_type = 'linear'
            linear_count += 1
        
        deg_rates[key] = deg_rate
        group_stats[key] = {
            'n_laps': len(group),
            'model': model_type,
            'r2_linear': r2_linear,
            'r2_quadratic': r2_quadratic,
            'deg_rate': deg_rate
        }
    
    print(f'Linear models: {linear_count}')
    print(f'Quadratic models: {quadratic_count}')
    print(f'Insufficient data groups: {insufficient_count}')
    
    # ===== STEP 3: ASSIGN DEG_RATE TO ALL ROWS =====
    
    print(f'\nStep 3: Assigning degradation rates to all rows...')
    
    deg_rate_values = []
    
    for _, row in df.iterrows():
        key = (row['TyreCompound'], row['EventName'], row['Year'])
        deg_rate_values.append(deg_rates.get(key, np.nan))
    
    df['deg_rate'] = deg_rate_values
    
    # ===== STEP 3.5: FLAG UNRELIABLE DEG_RATE VALUES =====
    
    print(f'\nStep 3.5: Flagging unreliable deg_rate values...')
    
    # Create deg_rate_reliable column
    # Set to False for: negative values, NaN, or groups with <5 eligible laps
    df['deg_rate_reliable'] = True
    
    # Flag negative deg_rates as unreliable
    df.loc[df['deg_rate'] < 0, 'deg_rate_reliable'] = False
    
    # Flag NaN deg_rates as unreliable
    df.loc[df['deg_rate'].isna(), 'deg_rate_reliable'] = False
    
    # Flag groups with <5 eligible laps as unreliable
    for (compound, event, year), stats in group_stats.items():
        if stats['n_laps'] < 5:
            mask = (df['TyreCompound'] == compound) & (df['EventName'] == event) & (df['Year'] == year)
            df.loc[mask, 'deg_rate_reliable'] = False
    
    # ===== STEP 3.6: CAP EXTREME OUTLIERS =====
    
    print(f'\nStep 3.6: Capping extreme outliers (|deg_rate| > 1.0 s/lap)...')
    
    # Find groups with extreme outliers
    extreme_groups = []
    for (compound, event, year), deg_rate in deg_rates.items():
        if not np.isnan(deg_rate) and abs(deg_rate) > 1.0:
            extreme_groups.append((compound, event, year, deg_rate))
            # Flag and NaN the deg_rate for rows in this group
            mask = (df['TyreCompound'] == compound) & (df['EventName'] == event) & (df['Year'] == year)
            df.loc[mask, 'deg_rate'] = np.nan
            df.loc[mask, 'deg_rate_reliable'] = False
    
    if extreme_groups:
        print(f'Capped {len(extreme_groups)} groups with extreme outliers:')
        for compound, event, year, deg_rate in extreme_groups:
            print(f'  {compound} @ {event} {int(year)}: {deg_rate:.4f} s/lap')
    else:
        print(f'No extreme outliers found.')
    
    # Report counts
    unreliable_count = (~df['deg_rate_reliable']).sum()
    reliable_count = df['deg_rate_reliable'].sum()
    print(f'\ndeg_rate_reliable summary:')
    print(f'  Unreliable: {unreliable_count:,} rows ({unreliable_count/len(df)*100:.1f}%)')
    print(f'  Reliable:   {reliable_count:,} rows ({reliable_count/len(df)*100:.1f}%)')
    
    # ===== STEP 4: SANITY CHECKS =====
    
    print(f'\nStep 4: Sanity checks...')
    
    # Get unique combinations for reporting (avoid repeating same group)
    unique_groups = df[['TyreCompound', 'EventName', 'Year', 'deg_rate']].drop_duplicates().copy()
    unique_valid = unique_groups[unique_groups['deg_rate'].notna()].copy()
    
    # Top 5 highest (should be Soft compounds at high-deg circuits)
    print(f'\nTop 5 highest degradation rates:')
    if len(unique_valid) > 0:
        top5 = unique_valid.nlargest(5, 'deg_rate')
        for idx, row in top5.iterrows():
            print(f'  {row["TyreCompound"]:3s} @ {row["EventName"]:30s} {int(row["Year"])}: {row["deg_rate"]:7.4f} s/lap')
    
    # Top 5 lowest (should be Hard compounds at low-deg circuits)
    print(f'\nTop 5 lowest degradation rates:')
    if len(unique_valid) > 0:
        bot5 = unique_valid.nsmallest(5, 'deg_rate')
        for idx, row in bot5.iterrows():
            print(f'  {row["TyreCompound"]:3s} @ {row["EventName"]:30s} {int(row["Year"])}: {row["deg_rate"]:7.4f} s/lap')
    
    # Check for negative values
    negative_groups = unique_groups[unique_groups['deg_rate'] < 0]
    if len(negative_groups) > 0:
        print(f'\nWARNING: {len(negative_groups)} groups with negative deg_rate (physically implausible):')
        for _, row in negative_groups.head(10).iterrows():
            print(f'  {row["TyreCompound"]} @ {row["EventName"]} {int(row["Year"])}: {row["deg_rate"]:.4f}')
        if len(negative_groups) > 10:
            print(f'  ... and {len(negative_groups) - 10} more')
    else:
        print(f'\nNo negative deg_rate values found (good!)')
    
    # Count NaN deg_rates
    nan_count = df['deg_rate'].isna().sum()
    print(f'\nRows with NaN deg_rate: {nan_count:,} ({nan_count/len(df)*100:.1f}%)')
    
    # Breakdown of reliable vs unreliable by compound
    print(f'\nFinal deg_rate_reliable breakdown by compound:')
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        comp_df = df[df['TyreCompound'] == compound]
        reliable = comp_df['deg_rate_reliable'].sum()
        unreliable = (~comp_df['deg_rate_reliable']).sum()
        print(f'  {compound:8s}: reliable={reliable:6,} ({reliable/len(comp_df)*100:5.1f}%), unreliable={unreliable:6,} ({unreliable/len(comp_df)*100:5.1f}%)')
    
    # ===== SAVE UPDATED DATAFRAME =====
    
    print(f'\n{"="*70}')
    print('Saving updated DataFrame to data/processed/base_df.csv...')
    print(f'{"="*70}')
    
    root = Path(__file__).parent.parent
    output_path = root / 'data' / 'processed' / 'base_df.csv'
    df.to_csv(output_path, index=False)
    
    print(f'\nFinal DataFrame shape: {df.shape}')
    print(f'Columns ({len(df.columns)}): {df.columns.tolist()}')
    
    print(f'\nDegradation rate building complete!')
    
    return df


def build_pit_window_delta(df):
    """Add pit window delta features to the base DataFrame.

    pit_delta_N = pit_loss_s - (deg_rate * N). A negative value means pitting
    now is expected to be faster, while a positive value means staying out is
    still faster. The primary feature is pit_window_delta = pit_delta_3.
    """
    print(f'\n{"="*70}')
    print('Building pit window delta features...')
    print(f'{"="*70}')

    df = df.copy()

    pit_loss_by_event = {
        'Bahrain': 22,
        'Saudi Arabian': 24,
        'Saudi Arabia': 24,
        'Australian': 24,
        'Australia': 24,
        'Azerbaijan': 26,
        'Miami': 23,
        'Monaco': 28,
        'Spanish': 22,
        'Spain': 22,
        'Canadian': 23,
        'Canada': 23,
        'Austrian': 20,
        'Austria': 20,
        'British': 21,
        'Britain': 21,
        'Hungarian': 22,
        'Hungary': 22,
        'Belgian': 19,
        'Belgium': 19,
        'Dutch': 23,
        'Netherlands': 23,
        'Italian': 19,
        'Italy': 19,
        'Singapore': 26,
        'Japanese': 22,
        'Japan': 22,
        'Qatar': 22,
        'United States': 25,
        'Mexico City': 22,
        'Mexico': 22,
        'Brazil': 23,
        'São Paulo': 23,
        'Las Vegas': 24,
        'Chinese Grand Prix': 28,
        'Emilia Romagna Grand Prix': 25,
        'French Grand Prix': 23,
        'Abu Dhabi': 22,
    }
    default_pit_loss = 23
    missing_events = set()

    def lookup_pit_loss(event_name):
        for key, loss in pit_loss_by_event.items():
            if key in event_name:
                return loss
        missing_events.add(event_name)
        return default_pit_loss

    df['pit_loss_s'] = df['EventName'].apply(lookup_pit_loss)

    if missing_events:
        print(f'WARNING: {len(missing_events)} EventName(s) not found in pit loss dictionary. Using default {default_pit_loss}s:')
        for event_name in sorted(missing_events):
            print(f'  {event_name}')

    for n in [1, 3, 5]:
        col = f'pit_delta_{n}'
        df[col] = np.nan

    if 'deg_rate_reliable' not in df.columns:
        raise ValueError('deg_rate_reliable column is required before computing pit window delta features')

    reliable_mask = df['deg_rate_reliable'] == True
    for n in [1, 3, 5]:
        df.loc[reliable_mask, f'pit_delta_{n}'] = (
            df.loc[reliable_mask, 'pit_loss_s'] -
            (df.loc[reliable_mask, 'deg_rate'] * df.loc[reliable_mask, 'TyreLife'] * n)
        )

    df['pit_window_delta'] = df['pit_delta_3']
    df.loc[~reliable_mask, ['pit_delta_1', 'pit_delta_3', 'pit_delta_5', 'pit_window_delta']] = np.nan

    non_null_count = df['pit_window_delta'].notna().sum()
    valid_delta = df['pit_window_delta'].dropna()
    mean_delta = valid_delta.mean() if len(valid_delta) > 0 else np.nan
    median_delta = valid_delta.median() if len(valid_delta) > 0 else np.nan
    min_delta = valid_delta.min() if len(valid_delta) > 0 else np.nan
    max_delta = valid_delta.max() if len(valid_delta) > 0 else np.nan
    negative_proportion = (df['pit_delta_3'] < 0).sum() / len(df) if len(df) > 0 else 0

    print(f'\nPit window delta summary:')
    print(f'  Non-null pit_window_delta rows: {non_null_count:,}')
    print(f'  pit_window_delta distribution: mean={mean_delta:.4f}, median={median_delta:.4f}, min={min_delta:.4f}, max={max_delta:.4f}')
    print(f'  Proportion of laps where pit_delta_3 is negative: {negative_proportion:.2%}')

    print(f'\nSaving updated DataFrame to data/processed/base_df.csv...')
    root = Path(__file__).parent.parent
    output_path = root / 'data' / 'processed' / 'base_df.csv'
    df.to_csv(output_path, index=False)

    print(f'\nFinal DataFrame shape: {df.shape}')
    print(f'Columns ({len(df.columns)}): {df.columns.tolist()}')

    print(f'\nPit window delta building complete!')
    return df


def build_undercut_overcut_threat(df):
    """Add undercut and overcut threat features to the base DataFrame."""
    print(f'\n{"="*70}')
    print('Building undercut/overcut threat features...')
    print(f'{"="*70}')

    df = df.copy()

    if 'Position' not in df.columns:
        raise ValueError('Position column is required to compute threat features')
    if 'LapTime_s' not in df.columns:
        raise ValueError('LapTime_s column is required to compute threat features')
    if 'TyreLife' not in df.columns:
        raise ValueError('TyreLife column is required to compute threat features')

    df['undercut_threat_index'] = 0.0
    df['overcut_threat_index'] = 0.0
    df['undercut_threat_flag'] = False
    df['overcut_threat_flag'] = False

    # Identify cars within the same race and lap
    grouped = df.groupby(['Year', 'Round', 'LapNumber'], sort=False)
    results = []

    for (_, _, _), group in grouped:
        group = group.copy()
        group.sort_values('Position', inplace=True)

        for idx, row in group.iterrows():
            current_pos = row['Position']
            current_lap = row['LapTime_s']
            current_tyre = row['TyreLife']

            behind = group[group['Position'] > current_pos]
            ahead = group[group['Position'] < current_pos]

            undercut_threats = []
            for _, other in behind.iterrows():
                gap = abs(other['LapTime_s'] - current_lap)
                if gap <= 3.0:
                    age_diff = current_tyre - other['TyreLife']
                    if age_diff > 0:
                        gap_floor = max(gap, 0.5)
                        undercut_threats.append(age_diff / gap_floor)

            overcut_threats = []
            for _, other in ahead.iterrows():
                gap = abs(other['LapTime_s'] - current_lap)
                if gap <= 3.0:
                    age_diff = other['TyreLife'] - current_tyre
                    if age_diff > 0:
                        gap_floor = max(gap, 0.5)
                        overcut_threats.append(age_diff / gap_floor)

            undercut_index = max(undercut_threats) if undercut_threats else 0.0
            overcut_index = max(overcut_threats) if overcut_threats else 0.0
            undercut_index = min(undercut_index, 20.0)
            overcut_index = min(overcut_index, 20.0)
            undercut_flag = False
            overcut_flag = False

            results.append((idx, undercut_index, overcut_index, undercut_flag, overcut_flag))

    if results:
        idxs, undercut_vals, overcut_vals, undercut_flags, overcut_flags = zip(*results)
        df.loc[list(idxs), 'undercut_threat_index'] = undercut_vals
        df.loc[list(idxs), 'overcut_threat_index'] = overcut_vals

    non_null_count = df['undercut_threat_index'].notna().sum()

    nonzero_undercut = df.loc[df['undercut_threat_index'] > 0, 'undercut_threat_index']
    nonzero_overcut = df.loc[df['overcut_threat_index'] > 0, 'overcut_threat_index']
    undercut_threshold = float(np.percentile(nonzero_undercut, 75)) if len(nonzero_undercut) > 0 else 1.0
    overcut_threshold = float(np.percentile(nonzero_overcut, 75)) if len(nonzero_overcut) > 0 else 1.0

    if undercut_threshold >= 20.0 and len(nonzero_undercut[nonzero_undercut < 20.0]) > 0:
        undercut_threshold = float(np.percentile(nonzero_undercut[nonzero_undercut < 20.0], 75))
        print('  Note: undercut threshold hit cap; using 75th percentile of values below cap')

    if overcut_threshold >= 20.0 and len(nonzero_overcut[nonzero_overcut < 20.0]) > 0:
        overcut_threshold = float(np.percentile(nonzero_overcut[nonzero_overcut < 20.0], 75))
        print('  Note: overcut threshold hit cap; using 75th percentile of values below cap')

    df['undercut_threat_flag'] = df['undercut_threat_index'] > undercut_threshold
    df['overcut_threat_flag'] = df['overcut_threat_index'] > overcut_threshold

    undercut_flag_pct = df['undercut_threat_flag'].mean() if len(df) > 0 else 0.0
    overcut_flag_pct = df['overcut_threat_flag'].mean() if len(df) > 0 else 0.0

    print(f'\nUndercut/overcut threshold values: undercut={undercut_threshold:.4f}, overcut={overcut_threshold:.4f}')

    street_events = {
        'Monaco Grand Prix',
        'Singapore Grand Prix',
        'Azerbaijan Grand Prix',
        'Miami Grand Prix',
    }
    power_events = {
        'Bahrain Grand Prix',
        'Belgian Grand Prix',
        'Canadian Grand Prix',
        'Austrian Grand Prix',
        'British Grand Prix',
        'Italian Grand Prix',
        'United States Grand Prix',
        'Mexico City Grand Prix',
        'São Paulo Grand Prix',
        'Australian Grand Prix',
    }
    technical_events = {
        'Spanish Grand Prix',
        'Hungarian Grand Prix',
        'Abu Dhabi Grand Prix',
        'Japanese Grand Prix',
    }

    event_to_archetype = {}
    for event in sorted(street_events):
        event_to_archetype[event] = 'Street'
    for event in sorted(power_events):
        event_to_archetype[event] = 'Power'
    for event in sorted(technical_events):
        event_to_archetype[event] = 'Technical'

    print('\nEventName -> archetype mapping dictionary:')
    for event, archetype in sorted(event_to_archetype.items()):
        print(f'  {event} -> {archetype}')

    unique_events = set(df['EventName'].dropna().unique())
    print('\nEventName presence check for Technical assignment targets:')
    for target_event in [
        'Spanish Grand Prix',
        'Hungarian Grand Prix',
        'Abu Dhabi Grand Prix',
        'Japanese Grand Prix',
    ]:
        print(f'  {target_event}: {target_event in unique_events}')

    def map_archetype(event_name):
        return event_to_archetype.get(event_name, 'Other')

    df['circuit_archetype'] = df['EventName'].apply(map_archetype)
    archetype_means = df.groupby('circuit_archetype')['undercut_threat_index'].mean()
    street_mean = archetype_means.get('Street', np.nan)
    power_mean = archetype_means.get('Power', np.nan)

    print(f'\nUndercut/overcut threat summary:')
    print(f'  Non-null undercut_threat_index rows: {non_null_count:,}')
    print(f'  Proportion with undercut_threat_flag=True: {undercut_flag_pct:.2%}')
    print(f'  Proportion with overcut_threat_flag=True: {overcut_flag_pct:.2%}')
    print(f'  Mean undercut_threat_index by archetype: Street={street_mean:.4f}, Power={power_mean:.4f}')

    if not (0.15 <= undercut_flag_pct <= 0.30) or not (0.15 <= overcut_flag_pct <= 0.30):
        nonzero_vals = np.sort(nonzero_undercut.values) if len(nonzero_undercut) > 0 else np.array([])
        if len(nonzero_vals) > 0:
            deciles = np.percentile(nonzero_vals, np.arange(10, 100, 10))
            print(f'\nUndercut threat index deciles (10-90): {", ".join(f"{d:.4f}" for d in deciles)}')

    top5 = df.sort_values('undercut_threat_index', ascending=False).head(5)
    print(f'\nTop 5 highest undercut_threat_index:')
    for _, row in top5.iterrows():
        print(f"  {row['EventName']} | {row['Driver']} | Lap {int(row['LapNumber'])} | {row['undercut_threat_index']:.4f}")

    print(f'\nSaving updated DataFrame to data/processed/base_df.csv...')
    root = Path(__file__).parent.parent
    output_path = root / 'data' / 'processed' / 'base_df.csv'
    df.to_csv(output_path, index=False)

    print(f'\nFinal DataFrame shape: {df.shape}')
    print(f'Columns ({len(df.columns)}): {df.columns.tolist()}')
    print(f'\nUndercut/overcut threat building complete!')
    return df


def build_sc_exposure_rate(df):
    """Add circuit-level SC and VSC exposure priors to the base DataFrame."""
    print(f'\n{"="*70}')
    print('Building safety car exposure rate features...')
    print(f'{"="*70}')

    df = df.copy()

    if 'EventName' not in df.columns:
        raise ValueError('EventName column is required to compute SC exposure features')
    if 'Year' not in df.columns:
        raise ValueError('Year column is required to compute SC exposure features')
    if 'sc_active' not in df.columns:
        raise ValueError('sc_active column is required to compute SC exposure features')
    if 'vsc_active' not in df.columns:
        raise ValueError('vsc_active column is required to compute VSC exposure features')

    # Compute per race metrics using unique lap exposures
    race_metrics = []
    for (event, year), race in df.groupby(['EventName', 'Year'], sort=False):
        lap_groups = race.groupby('LapNumber', sort=False)
        lap_status = lap_groups.agg(
            sc_any=('sc_active', 'any'),
            vsc_any=('vsc_active', 'any')
        )
        total_laps = len(lap_status)
        sc_rate = lap_status['sc_any'].mean() if total_laps > 0 else 0.0
        vsc_rate = lap_status['vsc_any'].mean() if total_laps > 0 else 0.0

        sc_events = ((lap_status['sc_any'].astype(int).shift(fill_value=0) == 0) &
                     (lap_status['sc_any'].astype(int) == 1)).sum()
        vsc_events = ((lap_status['vsc_any'].astype(int).shift(fill_value=0) == 0) &
                      (lap_status['vsc_any'].astype(int) == 1)).sum()

        race_metrics.append({
            'EventName': event,
            'Year': year,
            'sc_rate': sc_rate,
            'vsc_rate': vsc_rate,
            'sc_event_count': sc_events,
            'vsc_event_count': vsc_events,
        })

    race_df = pd.DataFrame(race_metrics)

    circuit_priors = race_df.groupby('EventName', sort=False).agg(
        circuit_sc_rate=('sc_rate', 'mean'),
        circuit_vsc_rate=('vsc_rate', 'mean'),
        circuit_sc_events_per_race=('sc_event_count', 'mean'),
        circuit_vsc_events_per_race=('vsc_event_count', 'mean'),
    ).reset_index()

    # Assign baseline values back to every lap by circuit
    df = df.merge(circuit_priors, on='EventName', how='left')

    median_sc_rate = circuit_priors['circuit_sc_rate'].median()
    df['high_sc_circuit'] = df['circuit_sc_rate'] > median_sc_rate

    # Sanity checks
    top5_sc = circuit_priors.nlargest(5, 'circuit_sc_rate')
    bot5_sc = circuit_priors.nsmallest(5, 'circuit_sc_rate')
    top5_vsc = circuit_priors.nlargest(5, 'circuit_vsc_rate')

    print(f'\nTop 5 circuits by circuit_sc_rate:')
    for _, row in top5_sc.iterrows():
        print(f"  {row['EventName']}: {row['circuit_sc_rate']:.4f}")

    print(f'\nBottom 5 circuits by circuit_sc_rate:')
    for _, row in bot5_sc.iterrows():
        print(f"  {row['EventName']}: {row['circuit_sc_rate']:.4f}")

    print(f'\nTop 5 circuits by circuit_vsc_rate:')
    for _, row in top5_vsc.iterrows():
        print(f"  {row['EventName']}: {row['circuit_vsc_rate']:.4f}")

    high_count = df['high_sc_circuit'].sum()
    low_count = (~df['high_sc_circuit']).sum()
    print(f'\nhigh_sc_circuit rows: {high_count:,}, low_sc_circuit rows: {low_count:,}')

    print(f'\nSaving updated DataFrame to data/processed/base_df.csv...')
    root = Path(__file__).parent.parent
    output_path = root / 'data' / 'processed' / 'base_df.csv'
    df.to_csv(output_path, index=False)

    print(f'\nFinal DataFrame shape: {df.shape}')
    print(f'Columns ({len(df.columns)}): {df.columns.tolist()}')
    print(f'\nSafety car exposure rate building complete!')
    return df


def build_fuel_corrected_pace(df):
    """Add fuel-corrected lap time and pace differential features to the base DataFrame."""
    print(f'\n{"="*70}')
    print('Building fuel-corrected pace features...')
    print(f'{"="*70}')

    df = df.copy()

    required_columns = ['EventName', 'LapNumber', 'LapTime_s', 'TyreLife', 'compound_known',
                        'IsAccurate', 'sc_active', 'vsc_active', 'PitInTime', 'PitOutTime', 'TyreCompound',
                        'deg_rate_reliable']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f'{col} column is required to compute fuel-corrected pace features')

    starting_fuel_by_event = {
        'Monaco Grand Prix': 100,
        'Singapore Grand Prix': 105,
        'Hungarian Grand Prix': 100,
        'Spanish Grand Prix': 102,
        'Abu Dhabi Grand Prix': 100,
        'Bahrain Grand Prix': 102,
        'Saudi Arabian Grand Prix': 103,
        'Australian Grand Prix': 104,
        'Azerbaijan Grand Prix': 100,
        'Miami Grand Prix': 101,
        'Canadian Grand Prix': 100,
        'Austrian Grand Prix': 98,
        'British Grand Prix': 102,
        'Belgian Grand Prix': 104,
        'Dutch Grand Prix': 101,
        'Italian Grand Prix': 98,
        'Japanese Grand Prix': 103,
        'Qatar Grand Prix': 104,
        'United States Grand Prix': 104,
        'Mexico City Grand Prix': 100,
        'São Paulo Grand Prix': 103,
        'Las Vegas Grand Prix': 101,
        'Chinese Grand Prix': 103,
        'Emilia Romagna Grand Prix': 100,
        'French Grand Prix': 102,
    }
    default_starting_fuel = 102
    missing_events = set()

    def lookup_starting_fuel(event_name):
        for key, fuel in starting_fuel_by_event.items():
            if key in event_name:
                return fuel
        missing_events.add(event_name)
        return default_starting_fuel

    df['starting_fuel_kg'] = df['EventName'].apply(lookup_starting_fuel)
    if missing_events:
        print(f'WARNING: {len(missing_events)} EventName(s) not found in starting fuel dictionary. Using default {default_starting_fuel}kg:')
        for event_name in sorted(missing_events):
            print(f'  {event_name}')

    df['fuel_load_kg'] = (df['starting_fuel_kg'] - (df['LapNumber'] - 1) * 1.0).clip(lower=0.0)
    df['fuel_corrected_laptime'] = df['LapTime_s'] - (df['fuel_load_kg'] * 0.03)

    # Build eligibility mask same as build_tyre_degradation()
    deg_eligible = pd.Series(True, index=df.index)
    deg_eligible &= (df['compound_known'] == True)
    deg_eligible &= (df['IsAccurate'] == True)
    deg_eligible &= (df['sc_active'] == False)
    deg_eligible &= (df['vsc_active'] == False)
    deg_eligible &= (df['PitInTime'].isna())
    deg_eligible &= (df['PitOutTime'].isna())

    wet_compounds = {'INTERMEDIATE', 'WET'}
    deg_eligible &= ~df['TyreCompound'].isin(wet_compounds)

    for (driver, round_num), group in df[deg_eligible].groupby(['Driver', 'Round']):
        median_time = group['fuel_corrected_laptime'].median()
        threshold = median_time * 1.07
        outlier_indices = group[group['fuel_corrected_laptime'] > threshold].index
        deg_eligible.loc[outlier_indices] = False

    # Fit deg_rate_corrected using the same logic as build_tyre_degradation().
    deg_rates = {}
    group_stats = {}
    linear_count = 0
    quadratic_count = 0
    insufficient_count = 0

    for (compound, event, year), group in df[deg_eligible].groupby(['TyreCompound', 'EventName', 'Year']):
        key = (compound, event, year)
        if len(group) < 5:
            deg_rates[key] = np.nan
            group_stats[key] = {
                'n_laps': len(group),
                'model': 'insufficient',
                'r2_linear': np.nan,
                'r2_quadratic': np.nan,
                'deg_rate': np.nan,
            }
            insufficient_count += 1
            continue

        x = group['TyreLife'].values.astype(float)
        y = group['fuel_corrected_laptime'].values.astype(float)

        if np.isnan(x).any() or np.isnan(y).any():
            deg_rates[key] = np.nan
            group_stats[key] = {
                'n_laps': len(group),
                'model': 'invalid_data',
                'r2_linear': np.nan,
                'r2_quadratic': np.nan,
                'deg_rate': np.nan,
            }
            continue

        try:
            coeffs_linear = np.polyfit(x, y, 1)
            y_pred_linear = np.polyval(coeffs_linear, x)
            ss_res_linear = np.sum((y - y_pred_linear) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0
        except (np.linalg.LinAlgError, ValueError):
            r2_linear = np.nan
            coeffs_linear = None

        try:
            coeffs_quadratic = np.polyfit(x, y, 2)
            y_pred_quadratic = np.polyval(coeffs_quadratic, x)
            ss_res_quadratic = np.sum((y - y_pred_quadratic) ** 2)
            r2_quadratic = 1 - (ss_res_quadratic / ss_tot) if ss_tot > 0 else 0
        except (np.linalg.LinAlgError, ValueError):
            r2_quadratic = np.nan
            coeffs_quadratic = None

        if (not np.isnan(r2_quadratic) and not np.isnan(r2_linear) and
                r2_quadratic > r2_linear + 0.02):
            a = coeffs_quadratic[0]
            b = coeffs_quadratic[1]
            mean_tyre_life = x.mean()
            deg_rate = 2 * a * mean_tyre_life + b
            model_type = 'quadratic'
            quadratic_count += 1
        else:
            if coeffs_linear is not None:
                deg_rate = coeffs_linear[0]
            else:
                deg_rate = np.nan
            model_type = 'linear'
            linear_count += 1

        deg_rates[key] = deg_rate
        group_stats[key] = {
            'n_laps': len(group),
            'model': model_type,
            'r2_linear': r2_linear,
            'r2_quadratic': r2_quadratic,
            'deg_rate': deg_rate,
        }

    deg_rate_values = []
    for _, row in df.iterrows():
        key = (row['TyreCompound'], row['EventName'], row['Year'])
        deg_rate_values.append(deg_rates.get(key, np.nan))

    df['deg_rate_corrected'] = deg_rate_values
    df['deg_rate_corrected_reliable'] = True
    df.loc[df['deg_rate_corrected'] < 0, 'deg_rate_corrected_reliable'] = False
    df.loc[df['deg_rate_corrected'].isna(), 'deg_rate_corrected_reliable'] = False

    for (compound, event, year), stats in group_stats.items():
        if stats['n_laps'] < 5:
            mask = (
                (df['TyreCompound'] == compound) &
                (df['EventName'] == event) &
                (df['Year'] == year)
            )
            df.loc[mask, 'deg_rate_corrected_reliable'] = False

    original_group_reliability = df.groupby(['TyreCompound', 'EventName', 'Year'])['deg_rate_reliable'].any()
    corrected_group_reliability = df.groupby(['TyreCompound', 'EventName', 'Year'])['deg_rate_corrected_reliable'].any()
    newly_reliable_groups = corrected_group_reliability[corrected_group_reliability & ~original_group_reliability].sum()

    df['pace_differential'] = (
        df['fuel_corrected_laptime'] -
        df.groupby(['Year', 'Round', 'LapNumber'])['fuel_corrected_laptime'].transform('median')
    )

    non_null_count = df['fuel_corrected_laptime'].notna().sum()
    orig_reliable_group_count = int(original_group_reliability.sum())
    corrected_reliable_group_count = int(corrected_group_reliability.sum())

    print(f'\nFuel-corrected pace summary:')
    print(f'  Non-null fuel_corrected_laptime rows: {non_null_count:,}')
    print(f'  Original reliable groups: {orig_reliable_group_count}')
    print(f'  Fuel-corrected reliable groups: {corrected_reliable_group_count}')
    print(f'  Groups newly reliable after fuel correction: {newly_reliable_groups}')

    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        mean_deg = df.loc[df['TyreCompound'] == compound, 'deg_rate_corrected'].mean()
        print(f'  {compound:6s} mean deg_rate_corrected: {mean_deg:.4f}')

    pace_mean = df['pace_differential'].mean()
    pace_median = df['pace_differential'].median()
    print(f'\npace_differential distribution: mean={pace_mean:.4f}, median={pace_median:.4f}')

    print(f'\nSaving updated DataFrame to data/processed/base_df.csv...')
    root = Path(__file__).parent.parent
    output_path = root / 'data' / 'processed' / 'base_df.csv'
    df.to_csv(output_path, index=False)

    print(f'\nFinal DataFrame shape: {df.shape}')
    print(f'Columns ({len(df.columns)}): {df.columns.tolist()}')
    print(f'\nFuel-corrected pace building complete!')
    return df


def build_position_at_stake(df):
    """Add position-at-stake features to the base DataFrame."""
    print(f'\n{"="*70}')
    print('Building position-at-stake features...')
    print(f'{"="*70}')

    df = df.copy()

    required_columns = ['Year', 'Round', 'EventName', 'Driver', 'Team', 'LapNumber',
                        'Position', 'is_midfield', 'deg_rate_reliable']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f'{col} column is required to compute position-at-stake features')

    points_map = {
        1: 25,
        2: 18,
        3: 15,
        4: 12,
        5: 10,
        6: 8,
        7: 6,
        8: 4,
        9: 2,
        10: 1,
    }

    def points_for_position(position):
        if position is None or np.isnan(position):
            return 0
        position_int = int(position)
        return points_map.get(position_int, 0)

    # Derive final classified position per driver in each race
    final_laps = (
        df[df['Position'].notna()]
        .sort_values(['Year', 'Round', 'Driver', 'LapNumber'])
        .groupby(['Year', 'Round', 'Driver'], sort=False)
        .last()
        .reset_index()
    )
    final_laps['Position'] = pd.to_numeric(final_laps['Position'], errors='coerce')
    final_laps['finish_points'] = final_laps['Position'].apply(points_for_position)

    team_points = (
        final_laps
        .groupby(['Year', 'Round', 'Team'], sort=False)
        ['finish_points']
        .sum()
        .reset_index()
        .rename(columns={'finish_points': 'round_points'})
    )

    # Build cumulative constructor standings prior to each race
    standings = {}
    for year, year_points in team_points.groupby('Year', sort=False):
        standings[year] = {}
        current_totals = {}
        for round_num in sorted(year_points['Round'].unique()):
            standings[year][round_num] = current_totals.copy()
            round_points = year_points[year_points['Round'] == round_num]
            for _, row in round_points.iterrows():
                team = row['Team']
                current_totals[team] = current_totals.get(team, 0) + row['round_points']

    def lookup_standing(team, year, round_num):
        return standings.get(year, {}).get(round_num, {}).get(team, 0)

    # Legacy diagnostic (pre-fix behavior): only teams already in standings map received pressure_weight.
    legacy_pressure_lookup = {}
    for year, year_rounds in standings.items():
        for round_num, round_totals in sorted(year_rounds.items()):
            if not round_totals:
                continue
            teams = sorted(round_totals.items(), key=lambda x: (x[1], x[0]))
            points_values = [pts for _, pts in teams]
            team_order = [team for team, _ in teams]
            for idx, team in enumerate(team_order):
                current_points = round_totals[team]
                above_gap = None
                below_gap = None
                if idx < len(points_values) - 1:
                    above_gap = points_values[idx + 1] - current_points
                if idx > 0:
                    below_gap = current_points - points_values[idx - 1]

                gaps = [gap for gap in [above_gap, below_gap] if gap is not None]
                championship_gap = min(gaps) if gaps else 0.0
                legacy_pressure_lookup[(year, round_num, team)] = 1.0 / (1.0 + championship_gap / 10.0)

    legacy_affected = df[
        (df['is_midfield'] == True) &
        (df['Position'].notna()) &
        (~df.apply(
            lambda row: pd.notna(legacy_pressure_lookup.get((row['Year'], row['Round'], row['Team']), np.nan)),
            axis=1
        ))
    ][['Year', 'Round', 'Team']].copy()

    print(f'\nInvestigation — legacy null pressure_weight distribution:')
    print(f'  Affected rows: {len(legacy_affected):,}')
    if len(legacy_affected) > 0:
        print('  By Year:')
        for year, count in legacy_affected.groupby('Year').size().sort_index().items():
            print(f'    {int(year)}: {int(count):,}')
        print('  By Round:')
        for round_num, count in legacy_affected.groupby('Round').size().sort_index().items():
            print(f'    R{int(round_num)}: {int(count):,}')
        print('  By Team:')
        for team, count in legacy_affected.groupby('Team').size().sort_values(ascending=False).items():
            print(f'    {team:12s}: {int(count):,}')

    pressure_lookup = {}
    year_team_universe = {
        year: sorted(year_df['Team'].dropna().unique())
        for year, year_df in df.groupby('Year', sort=False)
    }

    for year, year_rounds in standings.items():
        all_teams_in_year = year_team_universe.get(year, [])
        if not all_teams_in_year:
            continue

        for round_num, round_totals in sorted(year_rounds.items()):
            # Include zero-point teams so Round 1 and pre-scoring teams have defined pressure.
            totals_with_zeros = {
                team: float(round_totals.get(team, 0.0))
                for team in all_teams_in_year
            }

            teams = sorted(totals_with_zeros.items(), key=lambda x: (x[1], x[0]))
            points_values = [pts for _, pts in teams]
            team_order = [team for team, _ in teams]
            for idx, team in enumerate(team_order):
                current_points = totals_with_zeros[team]
                above_gap = None
                below_gap = None
                if idx < len(points_values) - 1:
                    above_gap = points_values[idx + 1] - current_points
                if idx > 0:
                    below_gap = current_points - points_values[idx - 1]

                gaps = [gap for gap in [above_gap, below_gap] if gap is not None]
                championship_gap = min(gaps) if gaps else 0.0
                pressure_lookup[(year, round_num, team)] = 1.0 / (1.0 + championship_gap / 10.0)

    def get_pressure_weight(year, round_num, team, is_midfield):
        if not is_midfield:
            return np.nan
        return pressure_lookup.get((year, round_num, team), np.nan)

    def get_points_delta(position):
        if position is None or np.isnan(position):
            return np.nan
        if position <= 1:
            return 0.0
        current = int(position)
        return points_for_position(current - 1) - points_for_position(current)

    # Positive points_delta means a driver can gain constructor points by moving up one place (overtaking the car ahead).
    df['points_delta'] = df['Position'].apply(get_points_delta)
    df['pressure_weight'] = df.apply(
        lambda row: get_pressure_weight(row['Year'], row['Round'], row['Team'], row['is_midfield']),
        axis=1
    )
    df['position_at_stake'] = df['points_delta'] * df['pressure_weight']

    midfield_df = df[df['is_midfield'] == True].copy()
    mean_by_team = midfield_df.groupby('Team')['position_at_stake'].mean().sort_values(ascending=False)
    print(f'\nMean position_at_stake per midfield team across all laps:')
    for team, mean_val in mean_by_team.items():
        print(f'  {team:12s}: {mean_val:.4f}')

    for team_name in ['McLaren', 'Haas', 'Williams']:
        team_mean = mean_by_team.get(team_name, np.nan)
        print(f'  [Check] {team_name:8s} mean position_at_stake = {team_mean:.4f}')

    points_delta_count = (df['points_delta'] > 0).sum()
    null_position_at_stake = df['position_at_stake'].isna().sum()
    print(f'\nCount of laps where points_delta > 0: {points_delta_count:,}')
    print(f'Count of laps where position_at_stake is null: {null_position_at_stake:,}')

    top5 = df.nlargest(5, 'position_at_stake')[['Year', 'Round', 'Team', 'Driver', 'LapNumber', 'Position', 'position_at_stake']]
    print(f'\nTop 5 highest position_at_stake rows:')
    for _, row in top5.iterrows():
        print(f"  {int(row['Year'])} R{int(row['Round'])} | {row['Team']:12s} | {row['Driver']:3s} | Lap {int(row['LapNumber'])} | P{int(row['Position']) if not np.isnan(row['Position']) else 'NA'} | {row['position_at_stake']:.4f}")

    # ===== VERIFICATION BLOCK =====
    
    print(f'\n{"="*70}')
    print('Position-at-stake verification checks...')
    print(f'{"="*70}')
    
    # Check 1: Year breakdown for McLaren
    print(f'\nCheck 1 — Year breakdown for McLaren:')
    mclaren_df = df[(df['Team'] == 'McLaren') & (df['is_midfield'] == True)]
    mclaren_year_means = {}
    for year in [2022, 2023, 2024]:
        mclaren_year = mclaren_df[mclaren_df['Year'] == year]
        mean_val = mclaren_year['position_at_stake'].mean()
        mclaren_year_means[year] = mean_val
        count = len(mclaren_year)
        print(f'  {year}: mean={mean_val:.4f} (n={count:,} laps)')

    mcl_2022 = mclaren_year_means.get(2022, np.nan)
    mcl_2024 = mclaren_year_means.get(2024, np.nan)
    if pd.notna(mcl_2022) and pd.notna(mcl_2024):
        diff = mcl_2024 - mcl_2022
        pct_change = (diff / abs(mcl_2022) * 100.0) if mcl_2022 != 0 else np.nan
        materially_higher = (diff > 0) and (pd.isna(pct_change) or pct_change >= 10.0)
        print(
            f'  2024 vs 2022 difference: {diff:.4f} '
            f'({pct_change:.1f}% change)'
        )
        print(f'  Materially higher in 2024 vs 2022 (>=10% uplift): {materially_higher}')
    else:
        print('  Unable to compare 2024 vs 2022: one or both yearly means are NaN')
    
    # Check 2: Null count reconciliation
    print(f'\nCheck 2 — Null count reconciliation:')
    baseline_nulls = (df['is_midfield'] == False).sum()
    additional_nulls = (
        (df['is_midfield'] == True) &
        (df['Position'].isna())
    ).sum()
    reconciled_sum = baseline_nulls + additional_nulls
    actual_null_count = df['position_at_stake'].isna().sum()

    print(f'  is_midfield=False rows (expected baseline nulls): {baseline_nulls:,}')
    print(f'  is_midfield=True & Position is null rows (additional nulls): {additional_nulls:,}')
    print(f'  Reconciled sum: {reconciled_sum:,}')
    print(f'  Total null position_at_stake count: {actual_null_count:,}')
    print(f'  Reconciliation exact match (zero discrepancy): {reconciled_sum == actual_null_count}')

    if reconciled_sum == actual_null_count:
        print('  Reconciliation status: OK (component counts sum to total nulls)')
    else:
        discrepancy = actual_null_count - reconciled_sum
        print(f'  Reconciliation status: MISMATCH ({discrepancy:+,} rows)')

        mismatch_mask = (
            df['position_at_stake'].isna() &
            ~((df['is_midfield'] == False) | ((df['is_midfield'] == True) & (df['Position'].isna())))
        )
        mismatch_count = mismatch_mask.sum()
        print(f'  Rows in discrepancy bucket: {mismatch_count:,}')

        midfield_valid_pos_null_pas = (
            (df['is_midfield'] == True) &
            (df['Position'].notna()) &
            (df['position_at_stake'].isna())
        ).sum()
        missing_pressure_weight = (
            (df['is_midfield'] == True) &
            (df['Position'].notna()) &
            (df['pressure_weight'].isna())
        ).sum()
        missing_points_delta = (
            (df['is_midfield'] == True) &
            (df['Position'].notna()) &
            (df['points_delta'].isna())
        ).sum()

        print(f'  Midfield rows with valid Position but null position_at_stake: {midfield_valid_pos_null_pas:,}')
        print(f'  Likely source: missing pressure_weight on midfield valid-position rows: {missing_pressure_weight:,}')
        print(f'  Secondary source check: missing points_delta on midfield valid-position rows: {missing_points_delta:,}')

    round1_midfield_mask = (df['is_midfield'] == True) & (df['Round'] == 1)
    round1_midfield_all_one = (df.loc[round1_midfield_mask, 'pressure_weight'] == 1.0).all()
    print(f'  Round 1 midfield pressure_weight all equal to 1.0: {round1_midfield_all_one}')
    
    # Check 3: Dutch GP context (2023 Round 13, ZHO)
    print(f'\nCheck 3 — Confirm Dutch GP context:')
    r13_2023 = df[(df['Year'] == 2023) & (df['Round'] == 13)]
    if len(r13_2023) > 0:
        event_name = r13_2023['EventName'].iloc[0]
        print(f'  Full EventName for 2023 Round 13: {event_name}')
    else:
        print(f'  No data found for 2023 Round 13')
    
    # Check ZHO laps in top5
    zho_top5_rows = top5[top5['Driver'] == 'ZHO']
    if len(zho_top5_rows) > 0:
        any_sc_or_vsc = False
        print(f'  ZHO laps in Top 5 position_at_stake:')
        for _, row in zho_top5_rows.iterrows():
            year = int(row['Year'])
            round_num = int(row['Round'])
            lap_num = int(row['LapNumber'])
            zho_lap_data = df[(df['Year'] == year) & (df['Round'] == round_num) & 
                              (df['Driver'] == 'ZHO') & (df['LapNumber'] == lap_num)]
            if len(zho_lap_data) > 0:
                sc_active = zho_lap_data['sc_active'].iloc[0]
                vsc_active = zho_lap_data['vsc_active'].iloc[0]
                any_sc_or_vsc = any_sc_or_vsc or bool(sc_active) or bool(vsc_active)
                print(f'    Lap {lap_num}: sc_active={sc_active}, vsc_active={vsc_active}')
        print(f'  Any SC or VSC active on those ZHO Top 5 laps: {any_sc_or_vsc}')
    else:
        print('  No ZHO laps found in Top 5 position_at_stake rows')

    # Check 4: Full-dataset distribution
    print(f'\nCheck 4 — Full-dataset distribution:')
    midfield_nonnull = df[(df['is_midfield'] == True) & (df['position_at_stake'].notna())].copy()
    dist = midfield_nonnull['position_at_stake'].describe()
    print('  position_at_stake describe() for midfield non-null rows:')
    for key in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        print(f'    {key:>5s}: {dist[key]:.6f}')

    q95 = midfield_nonnull['position_at_stake'].quantile(0.95)
    q99 = midfield_nonnull['position_at_stake'].quantile(0.99)
    q999 = midfield_nonnull['position_at_stake'].quantile(0.999)
    print(f'  95th percentile: {q95:.6f}')
    print(f'  99th percentile: {q99:.6f}')
    print(f'  99.9th percentile: {q999:.6f}')

    max_val = midfield_nonnull['position_at_stake'].max()
    max_rows = midfield_nonnull[midfield_nonnull['position_at_stake'] == max_val]
    print(f'  Rows at max value ({max_val:.6f}): {len(max_rows):,}')

    top3_cols = ['Year', 'EventName', 'Driver', 'LapNumber', 'position_at_stake', 'sc_active', 'vsc_active']
    top3_rows = midfield_nonnull.nlargest(3, 'position_at_stake')[top3_cols]
    print('  Top 3 rows by position_at_stake:')
    for _, row in top3_rows.iterrows():
        print(
            f"    {int(row['Year'])} | {row['EventName']} | {row['Driver']} | "
            f"Lap {int(row['LapNumber'])} | {row['position_at_stake']:.6f} | "
            f"sc_active={row['sc_active']} | vsc_active={row['vsc_active']}"
        )

    # Check 5: Archetype breakdown
    print(f'\nCheck 5 — Archetype breakdown:')
    archetype_stats = (
        df[df['is_midfield'] == True]
        .groupby('circuit_archetype', dropna=False)['position_at_stake']
        .agg(['mean', 'std', 'count'])
        .sort_index()
    )

    overall_midfield_mean = midfield_nonnull['position_at_stake'].mean()
    print(f'  Overall midfield mean (non-null): {overall_midfield_mean:.6f}')
    print('  By circuit_archetype (mean, std, count):')
    for archetype, row in archetype_stats.iterrows():
        archetype_label = str(archetype)
        mean_val = row['mean']
        std_val = row['std']
        count_val = int(row['count'])
        if pd.notna(overall_midfield_mean) and overall_midfield_mean != 0 and pd.notna(mean_val):
            rel_diff = abs((mean_val - overall_midfield_mean) / overall_midfield_mean)
            flag = rel_diff > 0.15
        else:
            rel_diff = np.nan
            flag = False
        print(
            f'    {archetype_label:10s}: mean={mean_val:.6f}, std={std_val:.6f}, '
            f'count={count_val:,}, >15% diff={flag}'
        )

    # Check 6: points_delta sign convention worked example
    print(f'\nCheck 6 — points_delta sign convention:')
    example_mask = (
        (df['is_midfield'] == True) &
        (df['Position'].notna()) &
        (df['pressure_weight'].notna()) &
        (df['points_delta'].notna()) &
        (df['points_delta'] > 0)
    )
    if example_mask.any():
        example_row = df.loc[example_mask, ['Team', 'Position', 'points_delta', 'pressure_weight']].iloc[0]
    else:
        fallback_mask = (
            (df['is_midfield'] == True) &
            (df['Position'].notna()) &
            (df['pressure_weight'].notna()) &
            (df['points_delta'].notna())
        )
        example_row = df.loc[fallback_mask, ['Team', 'Position', 'points_delta', 'pressure_weight']].iloc[0]

    print(
        f"  Worked example -> Team={example_row['Team']}, "
        f"Position={int(example_row['Position'])}, "
        f"points_delta={example_row['points_delta']:.6f}, "
        f"pressure_weight={example_row['pressure_weight']:.6f}"
    )

    print(f'\nSaving updated DataFrame to data/processed/base_df.csv...')
    root = Path(__file__).parent.parent
    output_path = root / 'data' / 'processed' / 'base_df.csv'
    df.to_csv(output_path, index=False)

    print(f'\nFinal DataFrame shape: {df.shape}')
    print(f'Columns ({len(df.columns)}): {df.columns.tolist()}')
    print(f'\nPosition-at-stake building complete!')
    return df


if __name__ == '__main__':
    df = build_base_df()
    df['is_strategic_stop'] = df['is_strategic_stop'].astype(pd.BooleanDtype())
    df = build_tyre_degradation(df)
    df = build_pit_window_delta(df)
    df = build_undercut_overcut_threat(df)
    df = build_sc_exposure_rate(df)
    df = build_fuel_corrected_pace(df)
    df = build_position_at_stake(df)
