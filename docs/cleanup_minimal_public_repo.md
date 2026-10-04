# Minimal Public Repo Checklist

Scope: non-destructive cleanup with archival moves.
No files were deleted, committed, or pushed.


## Core

- `source_code/` (all current simulation modules and feature/model scripts)
- `diagnostics/` (step-by-step validation and correction scripts used in methodology trail)
- `tests/`
- `README.md`
- `.gitignore`
- `requirements.txt`

## Archived Superseded Scripts


- `archive/superseded_scripts/fetch_f1_2022.py`
  - Reason: overlaps with `fetch_f1_data.py`, which is the generalized fetch entry point.
- `archive/superseded_scripts/simulation_step2f_lock_archetype_races.py`
  - Reason: corrected by `diagnostics/simulation_step2g_lock_archetype_races_corrected.py`.



## Important Git Ignore Note

Current `.gitignore` contains `*.csv`.
That means generated CSV outputs in `data/` are not pushed unless this rule is changed or files are explicitly force-added.

## Archive Location

- `archive/superseded_scripts/`

This keeps provenance without cluttering top-level workflow paths.
