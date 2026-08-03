# Chapter 3 Section Regeneration Notes

## base_df provenance

Most recent EDA pass used:

- `/Users/demi/Downloads/Dissertation Code Project/Code Base/data/processed/base_df.csv`
- Loaded shape: `74536 x 56`
- Notebook provenance line: the EDA notebook prints `Located base_df.csv: /Users/demi/Downloads/Dissertation Code Project/Code Base/data/processed/base_df.csv`

## Fresh degradation rebuild

Rebuilt file used for the regenerated Chapter 3 sections:

- `/Users/demi/Downloads/Dissertation Code Project/Code Base/data/diagnostics/deg_rate_full_peryear_stats_fresh.csv`

Fresh per-year figures regenerated from the rebuilt file:

- `/Users/demi/Downloads/Dissertation Code Project/Code Base/data/eda_outputs_phase1_base_df/figures/tyre_degradation_curves_soft_peryear.png`
- `/Users/demi/Downloads/Dissertation Code Project/Code Base/data/eda_outputs_phase1_base_df/figures/tyre_degradation_curves_medium_peryear.png`
- `/Users/demi/Downloads/Dissertation Code Project/Code Base/data/eda_outputs_phase1_base_df/figures/tyre_degradation_curves_hard_peryear.png`

The rebuild log reports one capped outlier group:

- `SOFT @ Miami Grand Prix 2023: -1.1877 s/lap`

## Section 3.6.2 refreshed figure set

Use the regenerated per-year circuit facets above in place of the older pooled figures.

The fresh build preserves the 12-circuit scope and the same three-figure compound split, but the plots now come from the per-year file rather than the superseded pooled outputs.

## Section 3.6.4 refreshed comparison table

The regeneration script did not find an explicit Section 3.6.4 list in the repository text, so it used the three mandated combinations plus three fallback combinations with the largest absolute old pooled slopes:

| EventName | TyreCompound | Selection source |
| --- | --- | --- |
| Monaco Grand Prix | HARD | section_prompt_mandatory |
| British Grand Prix | SOFT | section_prompt_mandatory |
| Hungarian Grand Prix | HARD | section_prompt_mandatory |
| Japanese Grand Prix | SOFT | fallback_top_abs_old_pooled_slope |
| Monaco Grand Prix | MEDIUM | fallback_top_abs_old_pooled_slope |
| Azerbaijan Grand Prix | SOFT | fallback_top_abs_old_pooled_slope |

Hungarian Grand Prix / HARD remains the explicit anchor example in the regenerated table:

| Year | Old pooled n | Old pooled slope | New per-year n | New per-year slope | Low sample |
| --- | --- | --- | --- | --- | --- |
| 2022 | 1901 | -0.022748 | 230 | -0.004143 | False |
| 2023 | 1901 | -0.022748 | 21 | 0.009520 | True |
| 2024 | 1901 | -0.022748 | 808 | 0.032389 | False |

## Reference log

- `/Users/demi/Downloads/Dissertation Code Project/Code Base/data/diagnostics/chapter3_degradation_regeneration_run.log`