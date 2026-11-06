"""Idempotent one-shot operational batch: features, scores, alerts, monitoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from alerts import build_alert_events, upsert_alerts
from features import build_features
from monitoring import build_monitoring_report
from registry import DEFAULT_REGISTRY
from scoring import OperationalScorer
from storage import atomic_upsert_parquet, atomic_upsert_scores, read_state, write_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "entso_raw.parquet")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "runtime")
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model-version", default="latest")
    parser.add_argument("--initial-hours", type=int, default=24)
    parser.add_argument("--dedup-hours", type=int, default=3)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    raw = pd.read_parquet(args.raw)
    staged_path = args.state_dir / "staged_features.parquet"
    incoming_features = build_features(raw, output_path=staged_path)
    feature_store = atomic_upsert_parquet(
        args.state_dir / "features.parquet", incoming_features
    )

    scorer = OperationalScorer(args.model_version, args.registry_dir)
    all_scored = scorer.score(feature_store)
    state_path = args.state_dir / "state.json"
    state = read_state(state_path)
    if (
        state.get("last_scored_at")
        and state.get("model_version") == scorer.manifest["run_id"]
    ):
        cutoff = pd.Timestamp(state["last_scored_at"])
        new_scores = all_scored[pd.to_datetime(all_scored["timestamp"]) > cutoff]
    else:
        cutoff = feature_store.index.max() - pd.Timedelta(hours=args.initial_hours - 1)
        new_scores = all_scored[pd.to_datetime(all_scored["timestamp"]) >= cutoff]

    score_path = args.state_dir / "scores.parquet"
    if not new_scores.empty:
        stored_scores = atomic_upsert_scores(score_path, new_scores)
        incoming_alerts = build_alert_events(
            new_scores, scorer.manifest["score_reference"], args.dedup_hours
        )
        alerts = upsert_alerts(
            args.state_dir / "alerts.csv", incoming_alerts, args.dedup_hours
        )
        state["last_scored_at"] = pd.Timestamp(new_scores["timestamp"].max()).isoformat()
        state["model_version"] = scorer.manifest["run_id"]
        write_state(state_path, state)
    else:
        stored_scores = pd.read_parquet(score_path) if score_path.exists() else new_scores
        alert_path = args.state_dir / "alerts.csv"
        alerts = pd.read_csv(alert_path) if alert_path.exists() else pd.DataFrame()

    report = build_monitoring_report(
        feature_store,
        stored_scores,
        alerts,
        scorer.manifest,
        ROOT / "results" / "monitoring",
    )
    print(
        f"Operational batch complete | new scores={len(new_scores)} | "
        f"alerts={len(alerts)} | freshness={report['freshness_hours']:.1f}h"
    )
    return report


if __name__ == "__main__":
    run(parse_args())
