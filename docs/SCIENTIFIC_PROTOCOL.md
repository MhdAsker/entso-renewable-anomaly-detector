# Scientific Validation Protocol

Protocol version: 1.0 (2026-11-10)

This document fixes the first scientific-validation experiment before reading
its new result artifacts. It distinguishes an engineering benchmark from a
claim of independently validated grid-event detection.

## Research questions and hypotheses

1. **RQ1 — baseline competitiveness.** Does a multivariate Isolation Forest
   improve holdout event F1 over simple causal statistical baselines?
2. **RQ2 — seasonal conditioning.** Does an hour-by-month conditional quantile
   improve calibration relative to an unconditional score?
3. **RQ3 — feature contribution.** Do ramp, trailing-context, calendar, and
   cyclic groups incrementally improve Isolation Forest holdout F1 and PR-AUC?
4. **RQ4 — operating burden.** Which method achieves useful recall without an
   excessive number of false alerts or false events per 30 days?

The corresponding directional hypotheses are that Isolation Forest benefits
from non-error features, seasonal conditioning reduces false-alert burden, and
a label-free conformal operating point trades some recall for a more explicit
false-alarm control target. Rejection or mixed evidence is a valid result.

## Data and evidence levels

The checked-in experiment uses the Germany/Luxembourg 2023 feature extract.
The default target, `is_anomaly`, is a causal rolling forecast-error proxy. It
is not ground truth and is construct-coupled to several model inputs.

An optional independent outcome path accepts ENTSO-E imbalance prices and
volumes. The operational-stress label is the union of absolute price and
volume values exceeding their 99th percentiles estimated strictly in the fit
period. Missing outcome hours remain unknown and are excluded, never converted
to normal observations. Coverage and thresholds are written to the manifest.

At protocol creation, the configured ENTSO-E credential returned HTTP 401 for
the imbalance endpoint. Therefore no independent-outcome result is claimed in
the current study. The default run is evidence level **proxy benchmark only**.

## Locked split and leakage controls

- First 60%: unsupervised model fitting and seasonal reference estimation.
- Next 10%: decision-threshold calibration.
- Final 30%: untouched chronological holdout.
- Rolling and reference features are strictly trailing.
- All scaler and Isolation Forest parameters are fitted on the first split.
- Label-informed thresholds maximize F1 only on validation observations.
- Conformal MAD uses validation scores without labels and a fixed alpha of
  0.05; its exchangeability assumption is explicitly treated as approximate
  for seasonal time-series data.
- No model or threshold may be selected after inspecting final holdout results.

The existing expanding-window backtest remains a complementary temporal
robustness check. It must not be used to tune the final holdout after results
are observed.

## Comparators

1. causal rolling z-score;
2. causal rolling median/MAD;
3. seasonal month-hour 99th-quantile score;
4. Isolation Forest using all registered features; and
5. split-conformal p-values applied to the MAD score.

The cumulative Isolation Forest ablation adds: signed and normalized errors,
ramps, trailing statistics, calendar indicators, then cyclic encodings.

## Outcomes and metrics

Primary descriptive metric: event-level F1 with one-to-one overlap matching
and no point adjustment. Secondary metrics are event precision/recall,
detection delay, point precision/recall/F1, PR-AUC, ROC-AUC, Matthews
correlation coefficient, false alerts per 30 days, and false events per 30
days. Results are also stratified by season.

Uncertainty uses a 95% moving-block bootstrap with 168-hour blocks and 500
resamples. This preserves local serial dependence better than an IID bootstrap.
Intervals are descriptive; the study does not turn overlapping or
non-overlapping intervals into uncorrected significance claims.

## Reproduction

```powershell
python run_scientific_study.py
```

With authorized independent outcomes:

```powershell
python fetch_operational_outcomes.py --start 20230101 --end 20240101
python run_scientific_study.py `
  --outcomes data/external/operational_outcomes.parquet
```

Each run writes a feature-file SHA-256, Git revision, exact temporal bounds,
label provenance, package versions, bootstrap settings, tables, and figures to
`results/scientific/`.

## Interpretation rules

- Strong proxy agreement is not evidence of operational utility.
- ROC-AUC alone is insufficient under class imbalance; PR-AUC and alert burden
  must be reported beside it.
- Event metrics are reported without expanding a single detected point across
  an entire true event.
- Seasonal subgroup results are exploratory because subgroup sizes are small.
- A method is not called superior unless the direction is consistent across
  primary point/event metrics, uncertainty intervals, and rolling-origin
  checks.
- External validity requires multiple years, multiple bidding zones, and an
  independently governed outcome or adjudicated incident set.

## Methodological basis

- Wu and Keogh, “Current Time Series Anomaly Detection Benchmarks are Flawed
  and are Creating the Illusion of Progress,” *IEEE TKDE* (2021),
  <https://doi.org/10.1109/TKDE.2021.3112126>.
- Sørbø and Ruocco, “Navigating the Metric Maze: A Taxonomy of Evaluation
  Metrics for Anomaly Detection in Time Series,” *Data Mining and Knowledge
  Discovery* (2024), <https://doi.org/10.1007/s10618-023-00988-8>.
- Xu and Boström, “Conformalized Time Series Anomaly Thresholding with Latent
  Space Features,” *PMLR* (2026),
  <https://proceedings.mlr.press/v329/xu26a.html>.
- Liu and Paparrizos, “The Elephant in the Room: Towards A Reliable Time-Series
  Anomaly Detection Benchmark,” *NeurIPS Datasets and Benchmarks* (2024),
  <https://papers.nips.cc/paper_files/paper/2024/hash/c3f3c690b7a99fba16d0efd35cb83b2c-Abstract-Datasets_and_Benchmarks_Track.html>.

The protocol intentionally avoids claiming conformance to any single benchmark
standard; it adopts the shared principles of chronological evaluation,
transparent thresholding, strong simple baselines, and event-aware reporting.
