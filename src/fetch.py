"""Resilient, chunked ENTSO-E data ingestion with row-level quality flags."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SOURCE_COLUMNS = [
    "actual_wind_onshore_mw",
    "actual_wind_offshore_mw",
    "actual_solar_mw",
    "forecast_wind_onshore_mw",
    "forecast_wind_offshore_mw",
    "forecast_solar_mw",
]


class ENTSOFetcher:
    def __init__(self, max_attempts: int = 3, retry_delay: float = 1.0):
        load_dotenv()
        api_key = os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            raise RuntimeError("ENTSOE_API_KEY is not set; copy .env.example to .env")
        self.client = EntsoePandasClient(api_key=api_key)
        self.zone = "DE_LU"
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        logger.info("ENTSO-E client ready")

    def _fetch_with_retry(self, func, *args, **kwargs):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if attempt == self.max_attempts:
                    logger.error("%s failed after %d attempts: %s", func.__name__, attempt, exc)
                    return pd.DataFrame()
                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "%s attempt %d/%d failed; retrying in %.1fs: %s",
                    func.__name__, attempt, self.max_attempts, delay, exc,
                )
                time.sleep(delay)

    @staticmethod
    def _hourly_series(obj, name: str) -> pd.Series:
        if obj is None or getattr(obj, "empty", True):
            return pd.Series(dtype=float, name=name)
        if isinstance(obj, pd.DataFrame):
            preferred = [c for c in obj.columns if "actual aggregated" in str(c).lower()]
            obj = obj[preferred[0] if preferred else obj.select_dtypes("number").columns[0]]
        series = obj.copy()
        if series.index.tz is not None:
            series.index = series.index.tz_convert("UTC").tz_localize(None)
        series = pd.to_numeric(series, errors="coerce").resample("h").mean()
        series.name = name
        return series

    @staticmethod
    def _forecast_series(frame: pd.DataFrame, fragment: str, name: str) -> pd.Series:
        if frame is None or getattr(frame, "empty", True):
            return pd.Series(dtype=float, name=name)
        matches = [c for c in frame.columns if fragment.lower() in str(c).lower()]
        if not matches:
            logger.warning("Forecast column matching '%s' not found", fragment)
            return pd.Series(dtype=float, name=name)
        return ENTSOFetcher._hourly_series(frame[matches[0]], name)

    def _fetch_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        query_generation = self.client.query_generation
        actual_wind_on = self._fetch_with_retry(
            query_generation, self.zone, start=start, end=end, psr_type="B19"
        )
        actual_wind_off = self._fetch_with_retry(
            query_generation, self.zone, start=start, end=end, psr_type="B18"
        )
        actual_solar = self._fetch_with_retry(
            query_generation, self.zone, start=start, end=end, psr_type="B16"
        )
        forecast = self._fetch_with_retry(
            self.client.query_wind_and_solar_forecast, self.zone, start=start, end=end
        )

        parts = [
            self._hourly_series(actual_wind_on, "actual_wind_onshore_mw"),
            self._hourly_series(actual_wind_off, "actual_wind_offshore_mw"),
            self._hourly_series(actual_solar, "actual_solar_mw"),
            self._forecast_series(forecast, "Wind Onshore", "forecast_wind_onshore_mw"),
            self._forecast_series(forecast, "Wind Offshore", "forecast_wind_offshore_mw"),
            self._forecast_series(forecast, "Solar", "forecast_solar_mw"),
        ]
        non_empty = [part for part in parts if not part.empty]
        frame = pd.concat(non_empty, axis=1) if non_empty else pd.DataFrame()
        expected = pd.date_range(
            start.tz_convert("UTC").tz_localize(None),
            end.tz_convert("UTC").tz_localize(None),
            freq="h",
            inclusive="left",
        )
        frame = frame.reindex(expected)
        for column in SOURCE_COLUMNS:
            if column not in frame:
                frame[column] = float("nan")
        return frame[SOURCE_COLUMNS]

    def fetch_all(self, start: str = "20230101", end: str = "20240101") -> pd.DataFrame:
        start_ts = pd.Timestamp(start, tz="Europe/Brussels")
        end_ts = pd.Timestamp(end, tz="Europe/Brussels")
        if end_ts <= start_ts:
            raise ValueError("end must be later than start")

        chunks = []
        cursor = start_ts
        while cursor < end_ts:
            next_month = (cursor + pd.offsets.MonthBegin(1)).normalize()
            chunk_end = min(next_month, end_ts)
            logger.info("Fetching chunk %s to %s", cursor.date(), chunk_end.date())
            chunks.append(self._fetch_range(cursor, chunk_end))
            cursor = chunk_end

        df = pd.concat(chunks).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        missing_before = df[SOURCE_COLUMNS].isna()
        df[SOURCE_COLUMNS] = df[SOURCE_COLUMNS].ffill(limit=2)
        imputed = missing_before & df[SOURCE_COLUMNS].notna()
        df["quality_source_missing_count"] = missing_before.sum(axis=1)
        df["quality_imputed_count"] = imputed.sum(axis=1)
        df["quality_unresolved_count"] = df[SOURCE_COLUMNS].isna().sum(axis=1)
        df["quality_is_complete"] = (df["quality_unresolved_count"] == 0).astype(int)

        df["actual_wind_total_mw"] = df[
            ["actual_wind_onshore_mw", "actual_wind_offshore_mw"]
        ].sum(axis=1, min_count=2)
        df["forecast_wind_total_mw"] = df[
            ["forecast_wind_onshore_mw", "forecast_wind_offshore_mw"]
        ].sum(axis=1, min_count=2)
        df["actual_renewable_mw"] = df[
            ["actual_wind_total_mw", "actual_solar_mw"]
        ].sum(axis=1, min_count=2)
        df["forecast_renewable_mw"] = df[
            ["forecast_wind_total_mw", "forecast_solar_mw"]
        ].sum(axis=1, min_count=2)

        output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "entso_raw.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        logger.info(
            "Fetched %d hourly rows | complete %.1f%% | unresolved source values %d",
            len(df), 100 * df["quality_is_complete"].mean(),
            int(df["quality_unresolved_count"].sum()),
        )
        return df


if __name__ == "__main__":
    result = ENTSOFetcher().fetch_all()
    print(result.shape)
    print(result.filter(like="quality_").describe())
