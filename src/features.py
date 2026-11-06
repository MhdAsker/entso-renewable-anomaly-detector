import logging
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np

from capacity import DEFAULT_CAPACITY_PATH, capacities_for_index
from context import OPTIONAL_CONTEXT_COLS, attach_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "wind_error_mw",
    "solar_error_mw",
    "total_error_mw",
    "wind_error_norm",
    "solar_error_norm",
    "total_error_norm",
    "wind_ramp_mw",
    "solar_ramp_mw",
    "ramp_error_mw",
    "error_roll_mean_24h",
    "error_roll_std_24h",
    "error_roll_mean_168h",
    "abs_error_roll_max_24h",
    "hour",
    "month",
    "season",
    "is_night",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "week_sin",
    "week_cos",
    "year_sin",
    "year_cos",
]


def feature_columns_for(df: pd.DataFrame) -> list[str]:
    """Return core features plus any available optional context features."""
    columns = list(FEATURE_COLS)
    for column in OPTIONAL_CONTEXT_COLS:
        if column in df and df[column].notna().any():
            columns.extend([column, f"{column}_missing"])
    return columns

REQUIRED_INPUT_COLS = [
    "actual_wind_total_mw",
    "forecast_wind_total_mw",
    "actual_solar_mw",
    "forecast_solar_mw",
    "actual_renewable_mw",
    "forecast_renewable_mw",
]

_SEASON_MAP = {
    12: 0, 1: 0, 2: 0,   # winter
     3: 1, 4: 1, 5: 1,   # spring
     6: 2, 7: 2, 8: 2,   # summer
     9: 3, 10: 3, 11: 3, # autumn
}


def _validate_input(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_INPUT_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {', '.join(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input must use a pandas DatetimeIndex")
    if df.index.has_duplicates:
        raise ValueError("Input index contains duplicate timestamps")


def build_features(
    df: pd.DataFrame,
    output_path: str | os.PathLike | None = None,
    capacity_path: str | os.PathLike = DEFAULT_CAPACITY_PATH,
    context_path: str | os.PathLike | None = None,
) -> pd.DataFrame:
    _validate_input(df)
    df = df.copy()
    df = df.sort_index()
    if context_path is not None:
        df = attach_context(df, context_path)

    # ── 1. Forecast errors ────────────────────────────────────────────────────
    df["wind_error_mw"] = df["actual_wind_total_mw"] - df["forecast_wind_total_mw"]
    df["solar_error_mw"] = df["actual_solar_mw"] - df["forecast_solar_mw"]
    df["total_error_mw"] = df["actual_renewable_mw"] - df["forecast_renewable_mw"]

    # ── 2. Normalised errors ──────────────────────────────────────────────────
    capacity = capacities_for_index(df.index, capacity_path)
    df["installed_wind_mw"] = capacity["wind_mw"]
    df["installed_solar_mw"] = capacity["solar_mw"]
    df["installed_total_mw"] = capacity.sum(axis=1)
    df["wind_error_norm"] = df["wind_error_mw"] / df["installed_wind_mw"]
    df["solar_error_norm"] = df["solar_error_mw"] / df["installed_solar_mw"]
    df["total_error_norm"] = df["total_error_mw"] / df["installed_total_mw"]

    # ── 3. Absolute errors (for labeling) ────────────────────────────────────
    df["wind_abs_error"] = df["wind_error_mw"].abs()
    df["solar_abs_error"] = df["solar_error_mw"].abs()
    df["total_abs_error"] = df["total_error_mw"].abs()

    # ── 4. Ramp rates ─────────────────────────────────────────────────────────
    df["wind_ramp_mw"] = df["actual_wind_total_mw"].diff(1)
    df["solar_ramp_mw"] = df["actual_solar_mw"].diff(1)
    df["actual_ramp_mw"] = df["actual_renewable_mw"].diff(1)
    df["forecast_ramp_mw"] = df["forecast_renewable_mw"].diff(1)
    df["ramp_error_mw"] = df["actual_ramp_mw"] - df["forecast_ramp_mw"]

    # ── 5. Rolling statistics (shifted by 1 to avoid leakage) ────────────────
    df["error_roll_mean_24h"] = (
        df["total_error_mw"].rolling(24, min_periods=12).mean().shift(1)
    )
    df["error_roll_std_24h"] = (
        df["total_error_mw"].rolling(24, min_periods=12).std().shift(1)
    )
    df["error_roll_mean_168h"] = (
        df["total_error_mw"].rolling(168, min_periods=12).mean().shift(1)
    )
    df["abs_error_roll_max_24h"] = (
        df["total_abs_error"].rolling(24, min_periods=12).max().shift(1)
    )

    # ── 6. Time features ──────────────────────────────────────────────────────
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["season"] = df["month"].map(_SEASON_MAP)
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] >= 22)).astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    hour_angle = 2 * np.pi * df["hour"] / 24
    week_angle = 2 * np.pi * df.index.dayofweek / 7
    year_angle = 2 * np.pi * (df.index.dayofyear - 1) / 365.25
    df["hour_sin"] = np.sin(hour_angle)
    df["hour_cos"] = np.cos(hour_angle)
    df["week_sin"] = np.sin(week_angle)
    df["week_cos"] = np.cos(week_angle)
    df["year_sin"] = np.sin(year_angle)
    df["year_cos"] = np.cos(year_angle)

    # ── 7. Anomaly labels (rolling z-score on total_abs_error) ───────────────
    history = df["total_abs_error"].shift(1)
    df["rolling_mean"] = history.rolling(window=720, min_periods=168).mean()
    df["rolling_std"] = history.rolling(window=720, min_periods=168).std()
    df["z_score"] = (df["total_abs_error"] - df["rolling_mean"]) / df["rolling_std"]
    df["is_anomaly"] = (df["z_score"] > 2.0).astype(int)

    n_anomalies = df["is_anomaly"].sum()
    pct = 100 * n_anomalies / len(df)
    logger.info("Anomaly labels: %d hours flagged (%.1f%% of total)", n_anomalies, pct)

    # ── 8. Drop NaN rows created by rolling / diff ───────────────────────────
    n_before = len(df)
    model_features = feature_columns_for(df)
    df = df.dropna(subset=model_features + ["is_anomaly", "z_score"])
    n_dropped = n_before - len(df)
    logger.info(
        "Dropped %d rows due to NaN | Remaining: %d", n_dropped, len(df)
    )

    # ── 9. Save ───────────────────────────────────────────────────────────────
    out_path = output_path or os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "processed", "features.parquet"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_parquet(out_path)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    n_final = len(df)
    n_anomalies_final = int(df["is_anomaly"].sum())
    pct_final = 100 * n_anomalies_final / n_final
    date_start = df.index.min()
    date_end = df.index.max()
    mean_wind_err = df["wind_error_mw"].mean()
    mean_solar_err = df["solar_error_mw"].mean()
    max_total_err = df["total_abs_error"].max()
    max_ts = df["total_abs_error"].idxmax()

    print("=" * 44)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 44)
    print(f"Total rows      : {n_final}")
    print(f"Feature columns : {len(model_features)}")
    print(f"Anomaly hours   : {n_anomalies_final} ({pct_final:.1f}%)")
    print(f"Date range      : {date_start} → {date_end}")
    print(f"Mean wind error : {mean_wind_err:.0f} MW")
    print(f"Mean solar error: {mean_solar_err:.0f} MW")
    print(f"Max total error : {max_total_err:.0f} MW (at {max_ts})")
    print("=" * 44)

    return df


if __name__ == "__main__":
    raw = pd.read_parquet("data/raw/entso_raw.parquet")
    df = build_features(raw)
    print(df[["total_error_mw", "is_anomaly", "z_score"]].describe())
