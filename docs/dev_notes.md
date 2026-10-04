# Development and Robustness Notes

Internal notes kept for reproducibility and the audit trail. For the project overview, see the main [README](../README.md).

## Repository cleanup

To preserve reproducibility, no files were deleted during cleanup.

Archived superseded scripts:

- `archive/superseded_scripts/fetch_f1_2022.py`: older single-year fetch script.
- `archive/superseded_scripts/simulation_step2f_lock_archetype_races.py`: superseded by the corrected step2g script.

Notes:

- `source_code/simulation/race_model.py` and `source_code/simulation/race_model_vectorized.py` are both in active use and are retained.
- `.gitignore` includes `*.csv`, so CSV artefacts are not pushed unless explicitly unignored.

For the full keep/archive checklist, see [`cleanup_minimal_public_repo.md`](cleanup_minimal_public_repo.md).

## Bias-decomposition robustness

- A materiality threshold was added alongside the strict epsilon guard, using a p50-derived threshold of 0.88.
- The pilot reference cache was recomputed at `n_iterations=10000` for 12 races using 4 parallel workers.
- The full recompute took about 1,660 seconds (about 28 minutes). Use this for simulation run-time budgeting.
- Materiality filtering removed 37 of 45 previously extreme `r_b` outliers (82%).
- The 6 remaining rows with |`r_b`| > 5 map to exactly 2 driver-races: Belgian Grand Prix 2024 (PIA) and Monaco Grand Prix 2023 (ALO).
- These residual outliers are consistent with the documented race-level X scoping simplification. Large whole-race X-versus-B shortfalls propagate across all per-trigger rows for the same driver-race; they do not indicate unexplained Monte Carlo noise.

### Known reporting issue

The outlier-context field `caution_lap_count` appears to count field-wide caution-flagged rows rather than subject-specific laps (for example, Monaco shows 163 for a 78-lap race). This affects the context printout only and does not change the `r_b` calculations in `bias.py`.
