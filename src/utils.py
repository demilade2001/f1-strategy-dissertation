# utils placeholder (left intentionally empty for now)
"""Utilities for F1 data processing.

Includes:
- TrackStatus decoder and DataFrame applicator
- Lap accuracy filter
- Midfield driver filter (year mapping)
- Circuit archetype mapping and annotator
"""
from typing import Dict, Any
import pandas as pd

MIDFIELD_CONSTRUCTORS = [
    'McLaren',
    'Alpine',
    'Aston Martin',
    'AlphaTauri',
    'RB',
    'Sauber',
    'Haas',
    'Williams',
]

TRACK_STATUS_LABELS = {
    0: 'Green',
    1: 'Yellow',
    2: 'Double Yellow',
    3: 'Red',
    4: 'Safety Car',
    5: 'Virtual Safety Car',
    6: 'Formation Lap',
    7: 'VSC Ending',
}

CONSTRUCTOR_MAPPINGS_BY_YEAR = {
    2022: {
        'VER': 'Red Bull',
        'PER': 'Red Bull',
        'LEC': 'Ferrari',
        'SAI': 'Ferrari',
        'HAM': 'Mercedes',
        'RUS': 'Mercedes',
        'NOR': 'McLaren',
        'RIC': 'McLaren',
        'ALO': 'Alpine',
        'STR': 'Aston Martin',
        'OCO': 'Alpine',
        'GAS': 'AlphaTauri',
        'TSU': 'AlphaTauri',
        'DEV': 'AlphaTauri',
        'ZHO': 'Sauber',
        'BOT': 'Sauber',
        'MAG': 'Haas',
        'HUL': 'Haas',
        'ALB': 'Williams',
        'SAR': 'Williams',
        'LAT': 'Williams',
        'MSC': 'Haas',
        'VET': 'Aston Martin',
    },
    2023: {
        'VER': 'Red Bull',
        'PER': 'Red Bull',
        'LEC': 'Ferrari',
        'SAI': 'Ferrari',
        'HAM': 'Mercedes',
        'RUS': 'Mercedes',
        'NOR': 'McLaren',
        'RIC': 'McLaren',
        'ALO': 'Aston Martin',
        'STR': 'Aston Martin',
        'OCO': 'Alpine',
        'GAS': 'Alpine',
        'TSU': 'AlphaTauri',
        'DEV': 'AlphaTauri',
        'ZHO': 'Sauber',
        'BOT': 'Sauber',
        'MAG': 'Haas',
        'HUL': 'Haas',
        'ALB': 'Williams',
        'SAR': 'Williams',
        'VET': 'Aston Martin',
        'PIA': 'McLaren',
    },
    2024: {
        'VER': 'Red Bull',
        'PER': 'Red Bull',
        'LEC': 'Ferrari',
        'SAI': 'Ferrari',
        'HAM': 'Mercedes',
        'RUS': 'Mercedes',
        'NOR': 'McLaren',
        'PIA': 'McLaren',
        'ALO': 'Aston Martin',
        'STR': 'Aston Martin',
        'OCO': 'Alpine',
        'GAS': 'Alpine',
        'TSU': 'RB',
        'RIC': 'RB',
        'ZHO': 'Sauber',
        'BOT': 'Sauber',
        'MAG': 'Haas',
        'HUL': 'Haas',
        'ALB': 'Williams',
        'SAR': 'Williams',
        'BEA': 'Ferrari',
        'COL': 'Williams',
        'DOO': 'Alpine',
    },
}


CANONICAL_CONSTRUCTOR_GROUP = {
    # Keep historical Team values in base_df intact, but allow unified grouping
    # keys for cross-year team-level aggregation in simulation rollups.
    'ALPHATAURI': 'RB',
    'RB': 'RB',
    'ALFA ROMEO': 'Sauber',
    'SAUBER': 'Sauber',
}


def decode_track_status_label(status_int: Any) -> str:
    """Decode a raw FastF1 TrackStatus code to a human-readable label."""
    if status_int is None or (isinstance(status_int, float) and pd.isna(status_int)):
        return 'Unknown'
    try:
        status_code = int(status_int)
    except (TypeError, ValueError):
        return 'Unknown'
    return TRACK_STATUS_LABELS.get(status_code, 'Unknown')


def get_constructor(driver_code: Any, year: int, round_number: int) -> str:
    """Return the constructor name for a given driver code, season, and round."""
    if driver_code is None:
        raise ValueError('driver_code must be provided')
    driver_code_str = str(driver_code).strip().upper()
    if not driver_code_str:
        raise ValueError('driver_code must be a non-empty string')

    # Special case: RIC in 2023 moved to AlphaTauri from Round 12
    if year == 2023 and driver_code_str == 'RIC':
        if round_number >= 12:
            return 'AlphaTauri'
        else:
            raise ValueError(
                f'RIC was not on the grid in 2023 before Round 12 (requested round {round_number})'
            )

    # Special case: LAW in 2023 (rounds 13-17) and 2024 (rounds 19-24)
    if year == 2023 and driver_code_str == 'LAW':
        if 13 <= round_number <= 17:
            return 'AlphaTauri'
        else:
            raise ValueError(
                f'LAW was not on the grid in 2023 during round {round_number}'
            )

    if year == 2024 and driver_code_str == 'LAW':
        if 19 <= round_number <= 24:
            return 'RB'
        else:
            raise ValueError(
                f'LAW was not on the grid in 2024 before Round 19 (requested round {round_number})'
            )

    constructor_map = CONSTRUCTOR_MAPPINGS_BY_YEAR.get(year, {})
    constructor_name = constructor_map.get(driver_code_str)
    if constructor_name:
        return constructor_name

    raise ValueError(
        f'Unknown constructor mapping for driver code {driver_code_str} in {year} round {round_number}'
    )


def canonical_constructor_group(team_name: Any) -> str:
    """Return canonical constructor grouping label used for rollup aggregation.

    This is intentionally separate from get_constructor() so base_df Team remains
    historically accurate by season while team-level summaries can unify renamed
    constructors across seasons (e.g., AlphaTauri/RB).
    """
    if team_name is None:
        return ''
    raw = str(team_name).strip()
    if not raw:
        return ''
    return CANONICAL_CONSTRUCTOR_GROUP.get(raw.upper(), raw)


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
    # Check each digit separately, but treat some common multi-status codes as
    # containing Safety Car / VSC indicators.
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

    if s == '12':
        flags['TrackStatus_sc'] = True

    # Transition codes can include both SC and VSC markers; full SC takes precedence.
    if flags['TrackStatus_sc'] and flags['TrackStatus_vsc']:
        flags['TrackStatus_vsc'] = False

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
    # Use .loc with a boolean mask to ensure a DataFrame is returned (type-checkers may
    # interpret direct indexing as possibly returning a Series).
    mask = df['IsAccurate'] == True
    return df.loc[mask, :].copy()


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
        return df.head(0).copy()
    valid = MIDFIELD_DRIVERS.get(year, [])
    valid_upper = {d.upper() for d in valid}
    mask = df['Driver'].astype(str).str.upper().isin(valid_upper)
    return df.loc[mask, :].copy()


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

    if event_col in df.columns:
        events = df[event_col].astype(str)
    else:
        events = pd.Series([''] * len(df), index=df.index)

    df['CircuitArchetype'] = events.apply(_find_archetype)
    return df
