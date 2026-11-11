# Phase 2 Robustness Study Report

Study date: 2026-11-10  
Protocol: [`PHASE2_PROTOCOL.md`](PHASE2_PROTOCOL.md)  
Dataset: `de_lu_2023`  
Evidence level: `internal_proxy`

## Executive result

Phase 2 strengthens the internal evidence but does not establish external
validity. Seasonal quantile remains the highest-F1 Phase 1 operating point,
but its advantage over robust MAD is not resolved by the paired 168-hour block
bootstrap. Robust MAD is substantially more stable across chronological folds.
Conformal MAD is preferred in most high-false-negative-cost scenarios because
it misses no proxy positives, but it incurs a much larger review burden.
Isolation Forest is resilient to moderate Gaussian perturbation and vulnerable
to feature missingness.

All negative controls reject circularly shifted label alignment after Holm
adjustment. This supports genuine timestamp-specific association, but it does
not solve the circularity created by deriving both labels and several features
from forecast error.

## 1. Paired model comparisons

The Phase 1 seasonal-quantile detector is the reference. Differences are
candidate minus reference; intervals use 500 paired moving-block resamples.

| Candidate | F1 difference (95% CI) | PR-AUC difference (95% CI) | Interpretation |
|---|---:|---:|---|
| Robust MAD | -0.052 (-0.189, 0.071) | 0.000 (-0.018, 0.033) | No resolved difference |
| Conformal MAD | -0.086 (-0.177, 0.006) | -0.009 (-0.026, 0.027) | No resolved difference |
| Isolation Forest | -0.253 (-0.345, -0.178) | -0.318 (-0.438, -0.243) | Consistently worse |
| Rolling z-score | -0.543 (-0.727, -0.404) | -0.214 (-0.348, -0.114) | Consistently worse |

The correct conclusion is not that seasonal quantile definitively beats robust
MAD. Their paired intervals overlap zero, and robust MAD has the better
temporal-transfer profile.

## 2. Threshold-policy sensitivity

Changing from validation-label-optimized thresholds to validation-score
quantiles changes the ranking and operating burden.

| Model | Best tested validation quantile | Holdout F1 | Precision | Recall | False alerts / 30 d |
|---|---:|---:|---:|---:|---:|
| Robust MAD | 0.975 | 0.847 | 0.839 | 0.855 | 7.56 |
| Seasonal quantile | 0.975 | 0.754 | 0.607 | 0.994 | 29.68 |
| Rolling z-score | 0.950 | 0.708 | 0.726 | 0.691 | 12.04 |
| Isolation Forest | 0.975 | 0.599 | 0.525 | 0.697 | 29.12 |

These values describe the tested grid; they are not a second opportunity to
select a production threshold on the holdout. The result demonstrates that
headline F1 and model ordering depend materially on the decision policy.

## 3. Temporal transfer

| Model | F1 mean ± SD | Worst-fold F1 | Mean PR-AUC | Mean false alerts / 30 d | Threshold CV |
|---|---:|---:|---:|---:|---:|
| Robust MAD | **0.879 ± 0.134** | **0.683** | **0.984** | **2.66** | 0.122 |
| Rolling z-score | 0.632 ± 0.044 | 0.587 | 0.758 | 17.23 | **0.091** |
| Seasonal quantile | 0.625 ± 0.424 | 0.000 | 0.965 | 4.90 | 0.580 |
| Isolation Forest | 0.552 ± 0.092 | 0.460 | 0.693 | 39.33 | 2.216 |

Seasonal quantile has excellent ranking quality but produced no alerts in one
fold after validation-threshold transfer. This is a calibration failure, not a
reason to delete the fold. Robust MAD offers the strongest combined accuracy,
stability, and alert-burden result on the current proxy benchmark.

## 4. Negative controls

For every model and both F1 and PR-AUC, the observed metric exceeded all 500
sampled circular-shift null values. The finite-sample empirical p-value was
0.001996 for each test and the Holm-adjusted value was 0.01996.

This indicates the scores and decisions are timestamp-aligned with the proxy
events beyond what is explained by prevalence and contiguous event structure.
It does **not** prove operational event detection because the null test retains
the project's construct-coupled label definition.

## 5. Feature-corruption robustness

Isolation Forest was evaluated with its fitted model and Phase 1 threshold
unchanged.

| Corruption | Level | Mean F1 | 95% repetition range | Mean PR-AUC |
|---|---:|---:|---:|---:|
| None | 0% | 0.583 | 0.583–0.583 | 0.638 |
| Gaussian noise | 20% of fit SD | 0.579 | 0.560–0.597 | 0.615 |
| Missing cells, median imputation | 5% | 0.554 | 0.508–0.607 | 0.606 |
| Missing cells, median imputation | 10% | 0.495 | 0.467–0.539 | 0.557 |
| Missing cells, median imputation | 20% | 0.343 | 0.301–0.391 | 0.495 |

At 20% missingness, mean F1 falls by approximately 41% relative to the
uncorrupted holdout. Operational deployment should therefore monitor per-field
missingness and refuse or downgrade scores beyond a validated coverage limit.
The small improvement under low Gaussian noise is not interpreted as a benefit;
it is compatible with threshold movement and sampling variation.

## 6. Scenario utility

Conformal MAD is best in 11 of the 12 declared scenarios because its full
proxy-label recall avoids false-negative penalties. Seasonal quantile is best
only when false positives cost one unit and false negatives also cost one unit.

This is a sensitivity analysis, not an economic result. The coefficients are
dimensionless assumptions and omit market price, reserve activation, operator
capacity, alert clustering, and intervention effectiveness. Real deployment
requires costs estimated with grid and market stakeholders.

## 7. External-validity status

The dataset catalog contains one active dataset: Germany/Luxembourg 2023 with
an internal rolling forecast-error proxy. Consequently:

- cross-year replication is not available;
- cross-zone replication is not available;
- independently governed outcomes are not available; and
- causal or operational-benefit claims remain unsupported.

The catalog and runner are ready for additional datasets. A valid external
replication should add at least one different zone or year and one external
proxy or adjudicated outcome, retain the same protocol, and report every active
dataset rather than selecting favorable replications.

## Reproduction and artifacts

```powershell
python run_phase2_studies.py
```

The complete outputs are under `results/phase2/de_lu_2023/`:

- `paired_comparisons.csv` — paired block-bootstrap differences;
- `threshold_sensitivity.csv` — validation-score-quantile policies;
- `negative_controls.csv` — circular-shift null tests and Holm correction;
- `utility_scenarios.csv` — transparent decision assumptions;
- `stress_raw.csv` and `stress_summary.csv` — seeded corruption experiments;
- `rolling_folds.csv` and `threshold_stability.csv` — temporal transfer;
- four publication-ready diagnostic figures; and
- `dataset_manifest.json` — provenance, evidence level, split, and thresholds.

The root `phase2_manifest.json` records the Git state, data hash, seed, study
settings, and claim boundary. `cross_dataset_benchmark.csv` will expand
automatically as active catalog entries are added.
