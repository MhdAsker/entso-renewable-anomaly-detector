import logging
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

INSTALLED_WIND_MW = 66_000
INSTALLED_SOLAR_MW = 73_000
INSTALLED_TOTAL_MW = 139_000

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
]

_SEASON_MAP = {
    12: 0, 1: 0, 2: 0,   # winter
     3: 1, 4: 1, 5: 1,   # spring
     6: 2, 7: 2, 8: 2,   # summer
     9: 3, 10: 3, 11: 3, # autumn
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 1. Forecast errors ────────────────────────────────────────────────────
    df["wind_error_mw"] = df["actual_wind_total_mw"] - df["forecast_wind_total_mw"]
    df["solar_error_mw"] = df["actual_solar_mw"] - df["forecast_solar_mw"]
    df["total_error_mw"] = df["actual_renewable_mw"] - df["forecast_renewable_mw"]

    # ── 2. Normalised errors ──────────────────────────────────────────────────
    df["wind_error_norm"] = df["wind_error_mw"] / INSTALLED_WIND_MW
    df["solar_error_norm"] = df["solar_error_mw"] / INSTALLED_SOLAR_MW
    df["total_error_norm"] = df["total_error_mw"] / INSTALLED_TOTAL_MW

    # ── 3. Absolute errors (for labeling) ────────────────────────────────────
    df["wind_abs_error"] = df["wind_error_mw"].abs()
    df["solar_abs_error"] = df["solar_error_mw"].abs()
    df["total_abs_error"] = df["total_error_mw"].abs()

    # ── 4. Ramp rates ─────────────────────────────────────────────────────────
    df["wind_ramp_mw"] = df["actual_wind_total_mw"].diff(1)
    df["solar_ramp_mw"] = df["actual_solar_mw"].diff(1)
    df["forecast_ramp_mw"] = df["forecast_renewable_mw"].diff(1)
    df["ramp_error_mw"] = df["wind_ramp_mw"] - df["forecast_ramp_mw"]

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

    # ── 7. Anomaly labels (rolling z-score on total_abs_error) ───────────────
    df["rolling_mean"] = (
        df["total_abs_error"].rolling(window=720, min_periods=168).mean()
    )
    df["rolling_std"] = (
        df["total_abs_error"].rolling(window=720, min_periods=168).std()
    )
    df["z_score"] = (df["total_abs_error"] - df["rolling_mean"]) / df["rolling_std"]
    df["is_anomaly"] = (df["z_score"] > 2.0).astype(int)

    n_anomalies = df["is_anomaly"].sum()
    pct = 100 * n_anomalies / len(df)
    logger.info("Anomaly labels: %d hours flagged (%.1f%% of total)", n_anomalies, pct)

    # ── 8. Drop NaN rows created by rolling / diff ───────────────────────────
    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLS + ["is_anomaly", "z_score"])
    n_dropped = n_before - len(df)
    logger.info(
        "Dropped %d rows due to NaN | Remaining: %d", n_dropped, len(df)
    )

    # ── 9. Save ───────────────────────────────────────────────────────────────
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "processed", "features.parquet",
    )
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
    print(f"Feature columns : {len(FEATURE_COLS)}")
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
