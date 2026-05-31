"""Utilities for F1 data processing.

Includes:
- TrackStatus decoder and DataFrame applicator
- Lap accuracy filter
- Midfield driver filter (year mapping)
- Circuit archetype mapping and annotator
"""
from typing import Dict, Any
import pandas as pd


def decode_track_status(raw_status: Any) -> Dict[str, bool]:
    """Decode FastF1 raw TrackStatus integer into boolean flags.

    FastF1 encodes TrackStatus as a composite integer where each digit
    represents an active flag (decimal digits):
      1 -> green
      2 -> yellow
      4 -> safety car (SC)
      5 -> red flag
      6 -> virtual safety car (VSC)
      7 -> VSC ending

    The function accepts numbers or strings; missing/NaN -> all False.
    """
    flags = {
        'TrackStatus_green': False,
        'TrackStatus_yellow': False,
        'TrackStatus_sc': False,
        'TrackStatus_red': False,
        'TrackStatus_vsc': False,
        'TrackStatus_vsc_end': False,
    }
    if raw_status is None or (isinstance(raw_status, float) and pd.isna(raw_status)):
        return flags

    s = str(int(raw_status)) if isinstance(raw_status, (int, float)) and not pd.isna(raw_status) else str(raw_status)
    # Check each digit separately
    for ch in set(s):
        if ch == '1':
            flags['TrackStatus_green'] = True
        elif ch == '2':
            flags['TrackStatus_yellow'] = True
        elif ch == '4':
            flags['TrackStatus_sc'] = True
        elif ch == '5':
            flags['TrackStatus_red'] = True
        elif ch == '6':
            flags['TrackStatus_vsc'] = True
        elif ch == '7':
            flags['TrackStatus_vsc_end'] = True

    return flags


def is_sc_or_vsc(raw_status: Any) -> bool:
    """Return True if a Safety Car (SC) or Virtual SC (VSC) is active.

    This treats digits '4' (SC), '6' (VSC) and '7' (VSC ending) as SC/VSC conditions.
    """
    if raw_status is None or (isinstance(raw_status, float) and pd.isna(raw_status)):
        return False
    s = str(int(raw_status)) if isinstance(raw_status, (int, float)) and not pd.isna(raw_status) else str(raw_status)
    return any(ch in s for ch in ('4', '6', '7'))


def apply_decode_to_df(df: pd.DataFrame, track_col: str = 'TrackStatus') -> pd.DataFrame:
    """Apply track status decoding across DataFrame and append boolean columns.

    Adds columns: TrackStatus_green, TrackStatus_yellow, TrackStatus_sc,
    TrackStatus_red, TrackStatus_vsc, TrackStatus_vsc_end, TrackStatus_SCorVSC
    """
    if track_col not in df.columns:
        # Nothing to decode; return original df with columns added as False
        for k in ('TrackStatus_green', 'TrackStatus_yellow', 'TrackStatus_sc', 'TrackStatus_red', 'TrackStatus_vsc', 'TrackStatus_vsc_end', 'TrackStatus_SCorVSC'):
            df[k] = False
        return df

    decoded = df[track_col].apply(decode_track_status)
    # Expand dict into columns
    dec_df = pd.DataFrame(list(decoded))
    df = df.reset_index(drop=True).join(dec_df)
    df['TrackStatus_SCorVSC'] = df.apply(lambda r: bool(r.get('TrackStatus_sc') or r.get('TrackStatus_vsc') or r.get('TrackStatus_vsc_end')), axis=1)
    return df


def filter_accurate_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where `IsAccurate` is True.

    If `IsAccurate` column is missing, return the original DataFrame unchanged.
    """
    if 'IsAccurate' not in df.columns:
        return df.copy()
    return df[df['IsAccurate'] == True].copy()


# Midfield driver mappings (driver codes) per year.
# Big Three driver codes to exclude for all years: VER, PER, HAM, RUS, LEC, SAI
MIDFIELD_DRIVERS = {
    2022: [
        'ALO', 'OCO', 'BOT', 'ZHO', 'GAS', 'TSU', 'ALB', 'MAG', 'HUL', 'STR', 'VET', 'NOR', 'RIC'
    ],
    2023: [
        'ALO', 'OCO', 'BOT', 'ZHO', 'GAS', 'TSU', 'ALB', 'MAG', 'HUL', 'STR', 'VET', 'NOR', 'RIC'
    ],
    2024: [
        'ALO', 'OCO', 'BOT', 'ZHO', 'GAS', 'TSU', 'ALB', 'MAG', 'HUL', 'STR', 'VET', 'NOR', 'RIC'
    ],
}


def filter_midfield_drivers(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Filter DataFrame to midfield drivers for a given year using `MIDFIELD_DRIVERS`.

    If `Driver` column is missing, returns empty DataFrame. Driver codes comparison
    is case-insensitive.
    """
    if 'Driver' not in df.columns:
        return df.iloc[0:0].copy()
    valid = MIDFIELD_DRIVERS.get(year, [])
    valid_upper = {d.upper() for d in valid}
    return df[df['Driver'].astype(str).str.upper().isin(valid_upper)].copy()


# Circuit archetype mapping
CIRCUIT_ARCHETYPES = {
    'Street': ['Monaco', 'Baku', 'Singapore', 'Miami'],
    'Power': ['Monza', 'Spa', 'Silverstone', 'Bahrain'],
    'Technical': ['Barcelona', 'Hungary', 'Abu Dhabi', 'Suzuka'],
}


def add_circuit_archetype(df: pd.DataFrame, event_col: str = 'EventName') -> pd.DataFrame:
    """Add `CircuitArchetype` column based on `EventName`.

    Matches are case-insensitive and substring-based; non-matching events get 'Other'.
    """
    def _find_archetype(name: str) -> str:
        if not isinstance(name, str):
            return 'Other'
        n = name.lower()
        for arche, circuits in CIRCUIT_ARCHETYPES.items():
            for c in circuits:
                if c.lower() in n:
                    return arche
        return 'Other'

    df['CircuitArchetype'] = df.get(event_col, pd.Series([''] * len(df))).apply(_find_archetype)
    return df
