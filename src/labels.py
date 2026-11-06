"""Optional ingestion of independently sourced event labels."""

from pathlib import Path

import pandas as pd


def load_event_labels(path: str | Path) -> pd.Series:
    """Load `timestamp,is_event` CSV labels and return a UTC-naive hourly series."""
    labels = pd.read_csv(path)
    required = {"timestamp", "is_event"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Event-label file is missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(labels["timestamp"], utc=True).dt.tz_localize(None)
    if timestamps.duplicated().any():
        raise ValueError("Event-label timestamps must be unique")
    values = pd.to_numeric(labels["is_event"], errors="raise").astype(int)
    if not values.isin([0, 1]).all():
        raise ValueError("is_event must contain only 0 and 1")
    return pd.Series(values.to_numpy(), index=timestamps, name="event_label")


def attach_event_labels(features: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    result = features.copy()
    labels = load_event_labels(path)
    result["event_label"] = labels.reindex(result.index).fillna(0).astype(int)
    return result
