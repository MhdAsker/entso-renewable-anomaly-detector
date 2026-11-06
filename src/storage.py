"""Atomic, idempotent local storage helpers for scheduled batch runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def validate_time_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Data must use a DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate timestamps are not allowed")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted")


def atomic_upsert_parquet(path: str | Path, incoming: pd.DataFrame) -> pd.DataFrame:
    """Upsert by timestamp, keeping incoming rows, and atomically replace the file."""
    path = Path(path)
    validate_time_frame(incoming)
    if path.exists():
        current = pd.read_parquet(path)
        validate_time_frame(current)
        if set(current.columns) != set(incoming.columns):
            raise ValueError("Incoming parquet schema does not match existing schema")
        combined = pd.concat([current, incoming[current.columns]])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = incoming.copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_parquet(temporary)
    temporary.replace(path)
    return combined


def atomic_upsert_scores(path: str | Path, incoming: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "model", "model_version"}
    if not required.issubset(incoming.columns):
        raise ValueError(f"Scores require columns: {sorted(required)}")
    path = Path(path)
    current = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([current, incoming], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.drop_duplicates(
        ["timestamp", "model", "model_version"], keep="last"
    ).sort_values(["timestamp", "model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(path)
    return combined.reset_index(drop=True)


def read_state(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_state(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
