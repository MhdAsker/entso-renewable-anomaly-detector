"""Date-effective installed-capacity lookup used for normalisation."""

from pathlib import Path

import pandas as pd

DEFAULT_CAPACITY_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "capacity_history.csv"
)


def load_capacity_history(path: str | Path = DEFAULT_CAPACITY_PATH) -> pd.DataFrame:
    capacity = pd.read_csv(path, parse_dates=["effective_date"])
    required = {"effective_date", "wind_mw", "solar_mw"}
    missing = required - set(capacity.columns)
    if missing:
        raise ValueError(f"Capacity history is missing columns: {sorted(missing)}")
    if capacity.empty or capacity["effective_date"].duplicated().any():
        raise ValueError("Capacity history must have unique effective dates")
    if (capacity[["wind_mw", "solar_mw"]] <= 0).any().any():
        raise ValueError("Installed capacities must be positive")
    return capacity.sort_values("effective_date").reset_index(drop=True)


def capacities_for_index(
    index: pd.DatetimeIndex, path: str | Path = DEFAULT_CAPACITY_PATH
) -> pd.DataFrame:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    capacity = load_capacity_history(path)
    timestamps = pd.DataFrame({"timestamp": index, "_order": range(len(index))})
    timestamps["timestamp"] = timestamps["timestamp"].astype("datetime64[ns]")
    capacity["effective_date"] = capacity["effective_date"].astype("datetime64[ns]")
    merged = pd.merge_asof(
        timestamps.sort_values("timestamp"),
        capacity,
        left_on="timestamp",
        right_on="effective_date",
        direction="backward",
    ).sort_values("_order")
    if merged[["wind_mw", "solar_mw"]].isna().any().any():
        first = capacity["effective_date"].min().date()
        raise ValueError(f"Capacity history starts after requested data (first: {first})")
    merged.index = index
    return merged[["wind_mw", "solar_mw"]]
