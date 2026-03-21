import os
import logging
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ENTSOFetcher:
    INSTALLED_WIND_MW = 66000   # ~66 GW installed wind capacity DE 2023
    INSTALLED_SOLAR_MW = 73000  # ~73 GW installed solar capacity DE 2023

    def __init__(self):
        load_dotenv()
        self.client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
        self.country = "DE"
        self.zone = "DE_LU"
        self.installed_wind_mw = self.INSTALLED_WIND_MW
        self.installed_solar_mw = self.INSTALLED_SOLAR_MW
        logger.info("ENTSO-E client ready")

    def _safe_fetch(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", func.__name__, exc)
            return pd.DataFrame()

    def fetch_all(self, start: str = "20230101", end: str = "20231231") -> pd.DataFrame:
        start_ts = pd.Timestamp(start, tz="Europe/Brussels")
        end_ts = pd.Timestamp(end, tz="Europe/Brussels")

        # ── 1. Actual wind onshore (B19) ──────────────────────────────────────
        raw_wind_on = self._safe_fetch(
            self.client.query_generation,
            self.country,
            start=start_ts,
            end=end_ts,
            psr_type="B19",
        )

        # ── 2. Actual wind offshore (B18) ─────────────────────────────────────
        raw_wind_off = self._safe_fetch(
            self.client.query_generation,
            self.country,
            start=start_ts,
            end=end_ts,
            psr_type="B18",
        )

        # ── 3. Actual solar (B16) ─────────────────────────────────────────────
        raw_solar = self._safe_fetch(
            self.client.query_generation,
            self.country,
            start=start_ts,
            end=end_ts,
            psr_type="B16",
        )

        # ── 4. Wind + solar forecast (day-ahead) ──────────────────────────────
        raw_forecast = self._safe_fetch(
            self.client.query_wind_and_solar_forecast,
            self.zone,
            start=start_ts,
            end=end_ts,
        )

        # ── Helper: normalise to hourly UTC Series/DataFrame ──────────────────
        def to_hourly(obj, name=None):
            if obj is None or (isinstance(obj, pd.DataFrame) and obj.empty):
                return pd.Series(dtype=float, name=name)
            if isinstance(obj, pd.DataFrame):
                # query_generation may return MultiIndex columns; take first numeric col
                obj = obj.iloc[:, 0]
            obj = obj.copy()
            obj.index = obj.index.tz_convert("UTC").tz_localize(None)
            return obj.resample("h").mean()

        wind_on_h = to_hourly(raw_wind_on, "actual_wind_onshore_mw")
        wind_off_h = to_hourly(raw_wind_off, "actual_wind_offshore_mw")
        solar_h = to_hourly(raw_solar, "actual_solar_mw")

        # ── Forecast columns ──────────────────────────────────────────────────
        def forecast_series(df, col_fragment, out_name):
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                return pd.Series(dtype=float, name=out_name)
            match = [c for c in df.columns if col_fragment.lower() in c.lower()]
            if not match:
                logger.warning("Forecast column matching '%s' not found", col_fragment)
                return pd.Series(dtype=float, name=out_name)
            s = df[match[0]].copy()
            s.index = s.index.tz_convert("UTC").tz_localize(None)
            s = s.resample("h").mean()
            s.name = out_name
            return s

        fc_wind_on_h = forecast_series(raw_forecast, "Wind Onshore", "forecast_wind_onshore_mw")
        fc_wind_off_h = forecast_series(raw_forecast, "Wind Offshore", "forecast_wind_offshore_mw")
        fc_solar_h = forecast_series(raw_forecast, "Solar", "forecast_solar_mw")

        # ── Rename actuals ────────────────────────────────────────────────────
        wind_on_h.name = "actual_wind_onshore_mw"
        wind_off_h.name = "actual_wind_offshore_mw"
        solar_h.name = "actual_solar_mw"

        # ── Merge on index (outer join) ───────────────────────────────────────
        parts = [
            wind_on_h,
            wind_off_h,
            solar_h,
            fc_wind_on_h,
            fc_wind_off_h,
            fc_solar_h,
        ]
        # Filter out empty series before concat to avoid all-NaN columns
        non_empty = [p for p in parts if not p.empty]
        if not non_empty:
            logger.warning("All fetches returned empty data; returning empty DataFrame")
            return pd.DataFrame()

        df = pd.concat(non_empty, axis=1, join="outer")

        # Ensure all expected columns exist
        for col in [
            "actual_wind_onshore_mw",
            "actual_wind_offshore_mw",
            "actual_solar_mw",
            "forecast_wind_onshore_mw",
            "forecast_wind_offshore_mw",
            "forecast_solar_mw",
        ]:
            if col not in df.columns:
                df[col] = float("nan")

        # Forward-fill max 2 hours
        df = df.ffill(limit=2)

        # ── Computed columns ──────────────────────────────────────────────────
        df["actual_wind_total_mw"] = (
            df["actual_wind_onshore_mw"].fillna(0)
            + df["actual_wind_offshore_mw"].fillna(0)
        )
        df["forecast_wind_total_mw"] = (
            df["forecast_wind_onshore_mw"].fillna(0)
            + df["forecast_wind_offshore_mw"].fillna(0)
        )
        df["actual_renewable_mw"] = (
            df["actual_wind_total_mw"] + df["actual_solar_mw"].fillna(0)
        )
        df["forecast_renewable_mw"] = (
            df["forecast_wind_total_mw"] + df["forecast_solar_mw"].fillna(0)
        )

        # Drop rows where ALL actual columns are NaN
        actual_cols = ["actual_wind_onshore_mw", "actual_wind_offshore_mw", "actual_solar_mw"]
        df = df.dropna(subset=actual_cols, how="all")

        # ── Save ──────────────────────────────────────────────────────────────
        out_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "raw", "entso_raw.parquet"
        )
        df.to_parquet(out_path)

        wind_mean = df["actual_wind_total_mw"].mean()
        solar_mean = df["actual_solar_mw"].mean()
        logger.info(
            "Fetched %d hourly rows | Wind: %.0f MW avg | Solar: %.0f MW avg",
            len(df),
            wind_mean,
            solar_mean,
        )

        return df


if __name__ == "__main__":
    fetcher = ENTSOFetcher()
    df = fetcher.fetch_all("20230101", "20231231")
    print(df.head())
    print(df.shape)
    print(df.isnull().sum())
