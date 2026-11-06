# Operations guide

## Components

- `run_detector.py` trains, calibrates, evaluates, and registers an immutable
  model version under `models/registry/<run_id>/`.
- `run_operational.py` is an idempotent scheduled batch. It rebuilds causal
  features from the current raw store, scores only unseen timestamps, merges
  alert episodes, and refreshes monitoring output.
- `record_feedback.py` records true-positive, false-positive, or uncertain
  operator outcomes against an alert event.

Runtime state is stored under `runtime/` and written atomically. The model
registry and runtime state are intentionally excluded from Git.

## Registering models

New training runs register automatically:

```powershell
python run_detector.py --run-id germany-2023-v1
```

To migrate already trained calibrated artifacts without retraining:

```powershell
python register_current.py --run-id germany-2023-v1
```

Registry loading verifies SHA-256 checksums before scoring.

## Running the batch

```powershell
python run_operational.py
```

The first run scores the most recent 24 hours. Later runs use
`runtime/state.json` and score only timestamps newer than `last_scored_at`.
Repeating a run with unchanged data produces zero new scores.

To schedule hourly with Windows Task Scheduler, create a task whose program is
the environment's `python.exe`, arguments are `run_operational.py`, and start-in
directory is the repository root. Fetch or append new raw data before this step.
Use the same service account so `.env`, registry, and runtime paths remain stable.

## Outputs

- `runtime/features.parquet`: idempotent engineered-feature store;
- `runtime/scores.parquet`: model-versioned hourly scores;
- `runtime/alerts.csv`: deduplicated alert episodes and feedback;
- `results/monitoring/status.json`: machine-readable health status;
- `results/monitoring/dashboard.html`: freshness, alert-rate, and drift page.

Freshness is `critical` when the newest raw timestamp is more than six hours
old. The bundled 2023 dataset therefore reports critical freshness by design.
Calendar features are displayed as informational drift because seasonal movement
is expected; operational signal features retain warning and critical states.

## Feedback

```powershell
python record_feedback.py EVENT_ID true_positive --notes "Confirmed reserve activation"
```

Allowed labels are `true_positive`, `false_positive`, and `uncertain`. Feedback
is retained when later batches upsert the same alert episode.

## Recovery and model rollback

- Delete `runtime/state.json` to rescore the latest window without deleting raw data.
- Pass `--model-version <run_id>` to pin or roll back a registered version.
- Do not edit registry artifacts. Register a new version when models, thresholds,
  or feature contracts change.
