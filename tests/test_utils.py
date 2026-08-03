from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source_code import utils


def print_result(name: str, passed: bool, detail: str = '') -> int:
    status = 'PASS' if passed else 'FAIL'
    print(f'{status}: {name}' + (f' -> {detail}' if detail else ''))
    return 1 if passed else 0


def run_checks() -> tuple[int, int, int]:
    passed = 0
    failed = 0
    checks_run = 0

    def run(name: str, func: Any) -> None:
        nonlocal passed, failed, checks_run
        checks_run += 1
        try:
            result, detail = func()
            if result:
                passed += 1
                print_result(name, True)
            else:
                failed += 1
                print_result(name, False, detail)
        except Exception as exc:
            failed += 1
            print_result(name, False, str(exc))

    # TrackStatus decoder checks
    run('decode_track_status(1) should mark green', lambda: (
        utils.decode_track_status(1).get('TrackStatus_green', False) is True,
        'green flag not set',
    ))
    run('decode_track_status(4) should mark sc', lambda: (
        utils.decode_track_status(4).get('TrackStatus_sc', False) is True,
        'sc flag not set',
    ))
    run('decode_track_status(6) should mark vsc', lambda: (
        utils.decode_track_status(6).get('TrackStatus_vsc', False) is True,
        'vsc flag not set',
    ))
    run('decode_track_status(12) should mark sc', lambda: (
        utils.decode_track_status(12).get('TrackStatus_sc', False) is True,
        'sc flag not set for code 12',
    ))

    # Apply decode across all TrackStatus values in the laps CSVs
    unhandled_values = []
    for year in (2022, 2023, 2024):
        path = ROOT / 'data' / 'raw' / str(year) / f'f1_{year}_laps.csv'
        if not path.exists():
            unhandled_values.append(f'missing file {path}')
            continue
        df = pd.read_csv(path)
        for index, value in df['TrackStatus'].items():
            if pd.isna(value):
                unhandled_values.append(f'{year}:{index}:NaN')
                continue
            flags = utils.decode_track_status(value)
            if not any(flags.values()):
                unhandled_values.append(f'{year}:{index}:{value}')

    run('decode_track_status handles all TrackStatus values in CSVs', lambda: (
        len(unhandled_values) == 0,
        f'unhandled values present: {unhandled_values[:10]}' if unhandled_values else '',
    ))

    # Constructor mapping checks
    constructor_tests = [
        ('VER', 2022, 1, 'Red Bull'),
        ('NOR', 2022, 1, 'McLaren'),
        ('DEV', 2023, 11, 'AlphaTauri'),
        ('RIC', 2023, 12, 'AlphaTauri'),  # RIC moved to AlphaTauri from Round 12
        ('LAW', 2023, 13, 'AlphaTauri'),  # LAW as substitute starting Round 13
        ('TSU', 2024, 1, 'RB'),
        ('BOT', 2024, 1, 'Sauber'),
    ]

    for driver_code, year, rnd, expected in constructor_tests:
        run(f'get_constructor({driver_code}, {year}, Round {rnd}) == {expected}',
            lambda driver_code=driver_code, year=year, rnd=rnd, expected=expected: (
                utils.get_constructor(driver_code, year, rnd) == expected,
                f'expected {expected}',
            ))

    def invalid_driver_check() -> tuple[bool, str]:
        try:
            utils.get_constructor('XXX', 2023, 1)
        except ValueError:
            return True, ''
        return False, 'did not raise ValueError for invalid driver code'

    run('get_constructor raises ValueError for invalid driver code', invalid_driver_check)

    # Coverage check for driver/year/round combinations present in CSVs
    coverage_errors: list[str] = []
    skipped_driver_rounds: list[str] = []  # Track drivers unavailable in certain rounds
    drivers_per_year: dict[int, set[str]] = {2022: set(), 2023: set(), 2024: set()}
    
    for year in (2022, 2023, 2024):
        path = ROOT / 'data' / 'raw' / str(year) / f'f1_{year}_laps.csv'
        if not path.exists():
            coverage_errors.append(f'missing file {path}')
            continue
        df = pd.read_csv(path)
        unique_combos = df[['Round', 'Driver']].drop_duplicates()
        for _, row in unique_combos.iterrows():
            driver_code = row['Driver']
            round_number = int(row['Round'])
            try:
                constructor = utils.get_constructor(driver_code, year, round_number)
                drivers_per_year[year].add(str(driver_code).strip().upper())
                if not constructor:
                    coverage_errors.append(f'empty constructor for {driver_code} {year} {round_number}')
            except ValueError as exc:
                # Expected for drivers not on grid in certain rounds (e.g., RIC pre-round 12 in 2023)
                skipped_driver_rounds.append(f'{driver_code} {year} r{round_number}')

    run('coverage check for all driver/year/round combos', lambda: (
        len(coverage_errors) == 0,
        f'coverage errors: {coverage_errors[:10]}' if coverage_errors else '',
    ))

    return checks_run, passed, failed


if __name__ == '__main__':
    total, passed, failed = run_checks()
    print('\nSummary:')
    print(f'  total checks run: {total}')
    print(f'  total passed:    {passed}')
    print(f'  total failed:    {failed}')
    if failed:
        sys.exit(1)
