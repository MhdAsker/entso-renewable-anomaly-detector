"""Contract for optional weather, load, and cross-border context data."""

from pathlib import Path

import pandas as pd

OPTIONAL_CONTEXT_COLS = [
    "temperature_c",
    "wind_speed_mps",
    "irradiance_wm2",
    "actual_load_mw",
    "forecast_load_mw",
    "net_import_mw",
]


def attach_context(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Join an hourly `timestamp` CSV and retain explicit missingness indicators."""
    context = pd.read_csv(path)
    if "timestamp" not in context:
        raise ValueError("Context CSV must contain a timestamp column")
    timestamps = pd.to_datetime(context.pop("timestamp"), utc=True).dt.tz_localize(None)
    context.index = timestamps
    if context.index.duplicated().any():
        raise ValueError("Context timestamps must be unique")
    available = [column for column in OPTIONAL_CONTEXT_COLS if column in context]
    if not available:
        raise ValueError(f"Context CSV must contain one of: {OPTIONAL_CONTEXT_COLS}")
    result = frame.join(context[available], how="left", rsuffix="_context")
    for column in available:
        result[f"{column}_missing"] = result[column].isna().astype(int)
        result[column] = pd.to_numeric(result[column], errors="coerce").ffill(limit=6)
    return result
