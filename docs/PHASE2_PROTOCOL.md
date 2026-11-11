# Phase 2 Protocol: Robustness and External-Validity Readiness

Protocol version: 1.0 (2026-11-10)

Phase 2 begins after the Phase 1 holdout results are fixed. It does not retune
the Phase 1 models. Its purpose is to test whether those conclusions survive
paired uncertainty analysis, operating-policy changes, temporal transfer,
controlled feature corruption, and negative controls. It also establishes the
dataset contract required for genuine cross-zone and cross-year replication.

## Research questions

1. **Paired superiority:** Is any model's holdout F1 or PR-AUC consistently
   different from the Phase 1 seasonal-quantile reference when identical
   serially dependent blocks are resampled?
2. **Threshold dependence:** Do model rankings change when thresholds are set
   by validation-score quantiles rather than validation labels?
3. **Decision dependence:** Which detector is preferred under different
   explicit false-positive and false-negative cost assumptions?
4. **Robustness:** How quickly does Isolation Forest performance degrade under
   holdout-only feature missingness and measurement noise?
5. **Negative control:** Does observed performance exceed a null distribution
   formed by circularly shifting the label sequence while preserving its event
   clustering?
6. **Temporal transfer:** How variable are thresholds and F1 across expanding
   chronological folds?
7. **External validity:** Are conclusions replicated on at least two active
   datasets representing different years or bidding zones, with at least one
   independently sourced outcome?

## Locked analyses

### Paired moving-block comparisons

The seasonal-quantile detector is the reference because it had the highest
Phase 1 point and event F1. Candidate-minus-reference differences are reported
for F1 and PR-AUC using the same 168-hour block indices for both models. Five
hundred resamples produce percentile 95% intervals and a bootstrap win
fraction. The win fraction is descriptive, not a posterior probability.

### Threshold sensitivity

For each non-conformal score, thresholds are the 90th, 95th, 97.5th, 99th,
99.5th, and 99.9th percentiles of validation scores. Holdout recall, F1,
PR-AUC, MCC, and false alerts per 30 days are recorded. This analysis uses no
holdout labels to choose a threshold.

### Decision scenarios

Scenario utility is

\[
U = TP - c_{FP}FP - c_{FN}FN,
\]

where the true-positive benefit is fixed to one, `c_FP` is in
{0.01, 0.1, 0.5, 1}, and `c_FN` is in {1, 5, 10}. These are dimensionless
sensitivity assumptions, not estimated market costs. A detector is not called
economically beneficial from this analysis.

### Controlled corruption

Isolation Forest is scored after modifying only the holdout features. Random
missing cells are replaced with fit-period medians. Gaussian noise is scaled
by fit-period feature standard deviations. Levels are 0%, 1%, 5%, 10%, and
20%, with 20 seeded repetitions. Labels, fit data, validation thresholds, and
all other model parameters remain unchanged.

### Negative controls

The holdout label vector is circularly shifted by at least 168 hours. Five
hundred seeded shifts preserve positive prevalence and local event clustering
while breaking timestamp alignment. Empirical upper-tail p-values compare the
observed F1 and PR-AUC with their shifted-label null distributions. Holm
adjustment controls family-wise error across all reported model/metric tests.

This control can detect suspicious timestamp alignment; it cannot remove the
deeper construct coupling between forecast-error features and proxy labels.

### Temporal transfer

Four expanding-window folds repeat fit, validation-only threshold selection,
and later testing. Per-model threshold ranges, threshold coefficient of
variation, mean/standard deviation/minimum F1, PR-AUC, and alert burden are
reported. A zero-alert fold is retained as a result, not repaired post hoc.

## Dataset and evidence contract

`config/study_datasets.csv` registers every study dataset with a unique ID,
feature path, optional outcome path, zone, declared period, label type,
evidence level, and active flag. Allowed evidence levels are:

- `internal_proxy`: labels constructed from the analyzed feature family;
- `external_proxy`: independently sourced but rule-derived outcomes; and
- `adjudicated_outcome`: events reviewed under a documented annotation policy.

Label types are also constrained: `rolling_forecast_error_proxy` for internal
proxy evidence, `operational_stress_proxy` for imbalance-price/volume parquet
outcomes, and `adjudicated_event` for validated `timestamp,is_event` CSV files.
External evidence levels are rejected unless an outcome path is supplied.

The runner processes every active dataset and writes a cross-dataset benchmark.
External validity is not claimed unless at least two substantively distinct
datasets are active and at least one uses independently governed outcomes.

At protocol creation only `de_lu_2023` is active, with an internal proxy label.
Therefore Phase 2 can produce internal robustness evidence but cannot satisfy
the external-validity criterion.

## Outputs and interpretation rules

The command is:

```powershell
python run_phase2_studies.py
```

Artifacts are written under `results/phase2/<dataset_id>/`, with a root
cross-dataset table and manifest. Every conclusion must report:

- the label evidence level;
- whether the paired interval crosses zero;
- sensitivity to threshold and cost assumptions;
- worst-fold performance, not only mean performance;
- corruption degradation relative to the unmodified holdout; and
- negative-control results after Holm adjustment.

No result from this phase establishes causal benefit, ahead-of-time prediction,
or grid reliability improvement. Those claims require prospective evaluation,
real operational costs, and independently governed outcomes.

## Methodological basis

The design follows the time-series anomaly-evaluation concerns described by
Wu and Keogh (2021), the metric taxonomy of Sørbø and Ruocco (2024), and the
large-scale evaluation emphasis of TSB-AD (2024). Exact references and links
are listed in `docs/SCIENTIFIC_PROTOCOL.md` and the project README.
