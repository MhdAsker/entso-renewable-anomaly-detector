# Improvement plan

## Current architecture

The project downloads German wind and solar actual/forecast data from ENTSO-E,
creates hourly forecast-error and calendar features, generates proxy anomaly
labels from unusually large absolute errors, and compares a rolling z-score,
Isolation Forest, and LSTM autoencoder.

## Main drawbacks found

1. **The old benchmark was in-sample.** Isolation Forest was fitted and scored
   on the same year. The LSTM was trained on rows selected using the proxy
   labels and then evaluated against those labels. Reported metrics therefore
   overstated generalisation.
2. **The labels are proxy labels, not ground truth.** They are generated from
   the same forecast-error family used as model input. Metrics measure agreement
   with that rule, not verified grid incidents or operational value.
3. **Two causal/alignment bugs affected signals.** Rolling label and baseline
   windows included the current observation; LSTM windows were off by one. The
   ramp-error feature also compared wind actuals with total-renewable forecasts.
4. **Data ingestion is brittle.** Capacity constants are hard-coded, missing
   sources can silently become zeros in aggregate columns, and a single annual
   request is vulnerable to API gaps and schema changes.
5. **Production concerns are absent.** There is no scheduled incremental fetch,
   drift tracking, alert policy, model registry, or observability.

## Three phases

### Phase 1 — benchmark integrity and foundations (implemented)

- chronological 70/30 train/holdout evaluation;
- no label-assisted selection of LSTM training rows;
- causal rolling windows and corrected ramp/LSTM alignment;
- deterministic seeds, CLI configuration, schema validation, safer metrics;
- dependency bounds and regression tests;
- full-year fetch boundary and clearer API-key failure.

### Phase 2 — data and modelling quality (implemented)

- fetch in retryable monthly chunks and retain source-quality flags;
- load installed capacity by date instead of using constants;
- add cyclical time encoding and an optional weather/load/cross-border context contract;
- use rolling-origin backtests across seasons and tune only on validation folds;
- compare a robust MAD baseline and calibrate all alert thresholds;
- support independently sourced event labels through a validated CSV contract.

Actual weather and independently sourced incident records are not bundled with
the repository. They must be supplied with provenance through `data/external/`;
the project deliberately does not invent them.

### Phase 3 — operationalisation (implemented)

- incremental pipeline with idempotent storage and data contracts;
- versioned model artifacts plus reproducible metadata;
- near-real-time scoring API or scheduled batch job;
- alert deduplication, severity levels and operator feedback loop;
- dashboards for data freshness, alert rate, drift and delayed outcome metrics.

The implementation is a dependency-light scheduled batch rather than a
long-running API. It provides atomic timestamp upserts, versioned/checksummed
artifacts, persisted scoring state, alert episode deduplication and severity,
operator feedback, plus JSON and HTML monitoring output. See `docs/OPERATIONS.md`.

## Interpretation warning

Even after Phase 2, the bundled precision/recall/F1 results are agreement scores against a
statistical proxy. They must not be presented as validated detection of grid
incidents until an independent outcome-label dataset is supplied.
