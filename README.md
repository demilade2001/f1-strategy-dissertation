# F1 Pit Stop Strategy Dissertation

This repository contains the data pipeline, feature engineering, and simulation code used for a dissertation project on Formula 1 pit stop strategy modelling. It includes scripts to fetch lap and pit data via FastF1, utilities for data processing, and placeholders for modelling and simulation components.

Project structure:

- data/: raw and processed datasets and outputs
- src/: source code (data fetching, utils, features, models, simulations)
- notebooks/: exploratory analysis and visualisations
- results/: model outputs and evaluation artifacts

See `requirements.txt` for dependencies.

## Pre-Push Cleanup Guidance (No Deletions Applied)

This repository currently contains both core pipeline code and a full dissertation diagnostic trail.
To preserve reproducibility, no files were deleted in this cleanup pass.

Archived superseded scripts:

- `archive/superseded_scripts/fetch_f1_2022.py` (older single-year fetch script).
- `archive/superseded_scripts/simulation_step2f_lock_archetype_races.py` (superseded by corrected step2g script).

Notes:

- `src/simulation/race_model.py` and `src/simulation/race_model_vectorized.py` are both actively relevant and should be retained.
- The current `.gitignore` includes `*.csv`, so CSV artifacts are not pushed unless explicitly unignored.

For a structured keep/archive checklist, see `cleanup_minimal_public_repo.md`.

## Bias-Decomposition Robustness Outcome

- Materiality threshold was added alongside the strict epsilon guard using a p50-derived threshold of 0.88.
- Pilot reference cache was recomputed at n_iterations=10000 for 12 races using 4 parallel workers.
- Observed wall-clock for this full recompute was about 1660 seconds (about 28 minutes), which should be used for simulation run-time budgeting.
- Materiality filtering removed 37 of 45 previously extreme r_b outliers (82 percent).
- The 6 remaining rows with |r_b| > 5 map to exactly 2 driver-races: Belgian Grand Prix 2024 (PIA) and Monaco Grand Prix 2023 (ALO).
- These residual outliers are consistent with the documented race-level X scoping simplification: large whole-race X-versus-B shortfalls propagate across all per-trigger rows for the same race-driver, rather than indicating unexplained Monte Carlo noise.
- Diagnostic note: the outlier-context field caution_lap_count appears to be counting field-wide caution-flagged rows instead of subject-specific laps (for example, Monaco shows 163 for a 78-lap race). This is a cosmetic reporting issue in context printout only and does not affect bias.py r_b calculations.
