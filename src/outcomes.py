"""Independent operational-outcome ingestion and stress-proxy construction."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

logger = logging.getLogger(__name__)


def _safe_error(exc: Exception) -> str:
    """Remove API tokens from HTTP exception text before logging."""
    return re.sub(
        r"securityToken=[^&\s]+",
        "securityToken=<redacted>",
        str(exc),
        flags=re.IGNORECASE,
    )


def _flatten_column(column) -> str:
    if isinstance(column, tuple):
        column = "_".join(str(part) for part in column if str(part) != "")
    return re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")


def _hourly_frame(value, prefix: str) -> pd.DataFrame:
    if value is None or getattr(value, "empty", True):
        return pd.DataFrame()
    frame = value.to_frame() if isinstance(value, pd.Series) else value.copy()
    frame.columns = [f"{prefix}_{_flatten_column(column)}" for column in frame.columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    return frame.resample("h").mean()


class OperationalOutcomeFetcher:
    """Fetch imbalance prices and volumes without converting missing data to normal."""

    def __init__(self, max_attempts: int = 3, retry_delay: float = 1.0):
        load_dotenv(".env")
        api_key = os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            raise RuntimeError("ENTSOE_API_KEY is not set")
        self.client = EntsoePandasClient(api_key=api_key)
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

    def _query(self, method, *args, **kwargs):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return method(*args, **kwargs)
            except Exception as exc:
                if attempt == self.max_attempts:
                    logger.error("%s failed: %s", method.__name__, _safe_error(exc))
                    return pd.DataFrame()
                time.sleep(self.retry_delay * 2 ** (attempt - 1))

    def fetch(
        self, start: str, end: str, zone: str = "DE_LU"
    ) -> tuple[pd.DataFrame, dict]:
        start_ts = pd.Timestamp(start, tz="Europe/Brussels")
        end_ts = pd.Timestamp(end, tz="Europe/Brussels")
        if start_ts >= end_ts:
            raise ValueError("Outcome start must be earlier than end")
        chunks = []
        cursor = start_ts
        failures = []
        while cursor < end_ts:
            chunk_end = min((cursor + pd.offsets.MonthBegin(1)).normalize(), end_ts)
            price = self._query(
                self.client.query_imbalance_prices, zone, start=cursor, end=chunk_end
            )
            volume = self._query(
                self.client.query_imbalance_volumes, zone, start=cursor, end=chunk_end
            )
            price_frame = _hourly_frame(price, "imbalance_price")
            volume_frame = _hourly_frame(volume, "imbalance_volume")
            if price_frame.empty and volume_frame.empty:
                failures.append(f"{cursor.date()}:{chunk_end.date()}")
            chunk = pd.concat([price_frame, volume_frame], axis=1)
            expected = pd.date_range(
                cursor.tz_convert("UTC").tz_localize(None),
                chunk_end.tz_convert("UTC").tz_localize(None),
                freq="h",
                inclusive="left",
            )
            chunks.append(chunk.reindex(expected))
            cursor = chunk_end
        result = pd.concat(chunks).sort_index()
        result = result[~result.index.duplicated(keep="last")]
        observed = result.notna().any(axis=1) if len(result.columns) else pd.Series(False, index=result.index)
        coverage = {
            "zone": zone,
            "start": str(result.index.min()),
            "end": str(result.index.max()),
            "expected_hours": len(result),
            "observed_hours": int(observed.sum()),
            "coverage_fraction": float(observed.mean()) if len(observed) else 0.0,
            "columns": list(result.columns),
            "failed_chunks": failures,
        }
        return result, coverage


def build_operational_stress_label(
    outcomes: pd.DataFrame,
    calibration_end: pd.Timestamp,
    quantile: float = 0.99,
) -> tuple[pd.Series, dict]:
    """Create a documented operational proxy using calibration-period tails."""
    price_cols = [column for column in outcomes if column.startswith("imbalance_price_")]
    volume_cols = [column for column in outcomes if column.startswith("imbalance_volume_")]
    if not price_cols and not volume_cols:
        raise ValueError("No imbalance price or volume columns are available")
    calibration = outcomes.loc[:calibration_end]
    components = {}
    thresholds = {}
    usable_columns = []
    unavailable_components = []
    if price_cols:
        price_score = outcomes[price_cols].abs().max(axis=1, skipna=True)
        price_calibration = calibration[price_cols].abs().max(axis=1, skipna=True).dropna()
        if price_calibration.empty:
            unavailable_components.append("price")
        else:
            thresholds["price_abs_quantile"] = float(price_calibration.quantile(quantile))
            components["price"] = price_score >= thresholds["price_abs_quantile"]
            usable_columns.extend(price_cols)
    if volume_cols:
        volume_score = outcomes[volume_cols].abs().max(axis=1, skipna=True)
        volume_calibration = calibration[volume_cols].abs().max(axis=1, skipna=True).dropna()
        if volume_calibration.empty:
            unavailable_components.append("volume")
        else:
            thresholds["volume_abs_quantile"] = float(volume_calibration.quantile(quantile))
            components["volume"] = volume_score >= thresholds["volume_abs_quantile"]
            usable_columns.extend(volume_cols)
    if not components:
        raise ValueError("Calibration period has no observed operational outcomes")
    observed = outcomes[usable_columns].notna().any(axis=1)
    label = pd.Series(pd.NA, index=outcomes.index, dtype="Int64", name="operational_stress_label")
    combined = pd.concat(components, axis=1).any(axis=1)
    label.loc[observed] = combined.loc[observed].astype(int)
    metadata = {
        "definition": "price OR volume above calibration-period absolute quantile",
        "quantile": quantile,
        "calibration_end": str(calibration_end),
        "thresholds": thresholds,
        "unavailable_components": unavailable_components,
        "observed_hours": int(observed.sum()),
        "positive_hours": int(label.fillna(0).sum()),
    }
    return label, metadata


def save_outcomes(
    frame: pd.DataFrame, coverage: dict, output_path: str | Path
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path)
    output_path.with_suffix(".coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
