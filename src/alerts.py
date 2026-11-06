"""Alert severity, event deduplication, persistence, and feedback helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _severity(model: str, score: float, threshold: float, reference: dict) -> str:
    model_reference = reference.get(model, {})
    medium = max(threshold, float(model_reference.get("p95", threshold)))
    high = max(medium, float(model_reference.get("p99", medium)))
    critical = max(high, float(model_reference.get("max", high)))
    if score >= critical:
        return "critical"
    if score >= high:
        return "high"
    if score >= medium:
        return "medium"
    return "low"


def build_alert_events(
    scores: pd.DataFrame, score_reference: dict, dedup_hours: int = 3
) -> pd.DataFrame:
    anomalies = scores[scores["is_anomaly"] == 1].copy()
    columns = [
        "event_id", "model", "model_version", "start_time", "end_time",
        "severity", "peak_score", "threshold", "hours", "status",
        "operator_label", "notes",
    ]
    if anomalies.empty:
        return pd.DataFrame(columns=columns)
    anomalies["timestamp"] = pd.to_datetime(anomalies["timestamp"])
    events = []
    for model, group in anomalies.sort_values("timestamp").groupby("model"):
        group = group.copy()
        group["new_event"] = (
            group["timestamp"].diff() > pd.Timedelta(hours=dedup_hours)
        ).fillna(True)
        for _, event in group.groupby(group["new_event"].cumsum()):
            peak = event.loc[event["score"].idxmax()]
            start = event["timestamp"].min()
            end = event["timestamp"].max()
            severity = _severity(
                model, float(peak["score"]), float(peak["threshold"]), score_reference
            )
            key = f"{peak['model_version']}|{model}|{start.isoformat()}"
            events.append(
                {
                    "event_id": hashlib.sha1(key.encode()).hexdigest()[:16],
                    "model": model,
                    "model_version": peak["model_version"],
                    "start_time": start,
                    "end_time": end,
                    "severity": severity,
                    "peak_score": float(peak["score"]),
                    "threshold": float(peak["threshold"]),
                    "hours": len(event),
                    "status": "new",
                    "operator_label": "",
                    "notes": "",
                }
            )
    return pd.DataFrame(events, columns=columns).sort_values("start_time")


def upsert_alerts(
    path: str | Path, incoming: pd.DataFrame, dedup_hours: int = 3
) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        current = pd.read_csv(path, parse_dates=["start_time", "end_time"])
        combined = current.copy()
        for _, event in incoming.iterrows():
            candidates = combined[
                (combined["model"] == event["model"])
                & (combined["model_version"] == event["model_version"])
                & (pd.to_datetime(combined["end_time"]) <= pd.Timestamp(event["start_time"]))
                & (
                    pd.Timestamp(event["start_time"]) - pd.to_datetime(combined["end_time"])
                    <= pd.Timedelta(hours=dedup_hours)
                )
            ]
            if candidates.empty:
                combined = pd.concat([combined, event.to_frame().T], ignore_index=True)
                continue
            index = candidates.index[-1]
            combined.loc[index, "end_time"] = max(
                pd.Timestamp(combined.loc[index, "end_time"]), pd.Timestamp(event["end_time"])
            )
            combined.loc[index, "hours"] = int(combined.loc[index, "hours"]) + int(event["hours"])
            if float(event["peak_score"]) > float(combined.loc[index, "peak_score"]):
                combined.loc[index, "peak_score"] = event["peak_score"]
            if SEVERITY_RANK[event["severity"]] > SEVERITY_RANK[combined.loc[index, "severity"]]:
                combined.loc[index, "severity"] = event["severity"]
    else:
        combined = incoming.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)
    return combined


def record_feedback(
    path: str | Path, event_id: str, operator_label: str, notes: str = ""
) -> None:
    allowed = {"true_positive", "false_positive", "uncertain"}
    if operator_label not in allowed:
        raise ValueError(f"operator_label must be one of {sorted(allowed)}")
    path = Path(path)
    alerts = pd.read_csv(path)
    alerts["operator_label"] = alerts["operator_label"].fillna("").astype("object")
    alerts["notes"] = alerts["notes"].fillna("").astype("object")
    match = alerts["event_id"] == event_id
    if not match.any():
        raise KeyError(f"Unknown event_id: {event_id}")
    alerts.loc[match, "operator_label"] = operator_label
    alerts.loc[match, "notes"] = notes
    alerts.loc[match, "status"] = "reviewed"
    temporary = path.with_suffix(path.suffix + ".tmp")
    alerts.to_csv(temporary, index=False)
    temporary.replace(path)
