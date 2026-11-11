"""Dataset catalog and evidence-contract validation for scientific studies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_CATALOG_COLUMNS = {
    "dataset_id",
    "feature_path",
    "outcome_path",
    "zone",
    "period_start",
    "period_end",
    "label_type",
    "evidence_level",
    "active",
}
EVIDENCE_LEVELS = {"internal_proxy", "external_proxy", "adjudicated_outcome"}
LABEL_TYPES = {
    "rolling_forecast_error_proxy",
    "operational_stress_proxy",
    "adjudicated_event",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    feature_path: Path
    outcome_path: Path | None
    zone: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    label_type: str
    evidence_level: str


def load_dataset_catalog(path: str | Path, project_root: str | Path) -> list[DatasetSpec]:
    """Load active dataset specifications and resolve paths from the project root."""
    path = Path(path)
    project_root = Path(project_root)
    catalog = pd.read_csv(path, keep_default_na=False)
    missing = REQUIRED_CATALOG_COLUMNS - set(catalog)
    if missing:
        raise ValueError(f"Dataset catalog is missing columns: {sorted(missing)}")
    if catalog["dataset_id"].duplicated().any():
        raise ValueError("Dataset identifiers must be unique")
    active = catalog["active"].astype(str).str.lower().isin({"1", "true", "yes"})
    catalog = catalog.loc[active]
    if catalog.empty:
        raise ValueError("Dataset catalog has no active datasets")

    specifications = []
    for row in catalog.itertuples(index=False):
        if row.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError(
                f"Unsupported evidence level for {row.dataset_id}: {row.evidence_level}"
            )
        if row.label_type not in LABEL_TYPES:
            raise ValueError(
                f"Unsupported label type for {row.dataset_id}: {row.label_type}"
            )
        if row.evidence_level != "internal_proxy" and not row.outcome_path:
            raise ValueError(
                f"{row.dataset_id}: external evidence requires an outcome_path"
            )
        if row.evidence_level == "internal_proxy" and row.label_type != "rolling_forecast_error_proxy":
            raise ValueError(
                f"{row.dataset_id}: internal_proxy requires rolling_forecast_error_proxy"
            )
        if row.evidence_level == "adjudicated_outcome" and row.label_type != "adjudicated_event":
            raise ValueError(
                f"{row.dataset_id}: adjudicated_outcome requires adjudicated_event labels"
            )
        start = pd.Timestamp(row.period_start)
        end = pd.Timestamp(row.period_end)
        if start >= end:
            raise ValueError(f"Invalid study period for {row.dataset_id}")
        feature_path = project_root / row.feature_path
        outcome_path = project_root / row.outcome_path if row.outcome_path else None
        specifications.append(
            DatasetSpec(
                dataset_id=row.dataset_id,
                feature_path=feature_path,
                outcome_path=outcome_path,
                zone=row.zone,
                period_start=start,
                period_end=end,
                label_type=row.label_type,
                evidence_level=row.evidence_level,
            )
        )
    return specifications


def validate_dataset_period(frame: pd.DataFrame, specification: DatasetSpec) -> None:
    """Ensure the stored timestamps agree with their catalog declaration."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Study features must use a DatetimeIndex")
    if frame.empty:
        raise ValueError("Study feature data is empty")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Study timestamps must be sorted and unique")
    lower = specification.period_start
    upper = specification.period_end + pd.Timedelta(days=1)
    if frame.index.min() < lower or frame.index.max() >= upper:
        raise ValueError(
            f"Dataset {specification.dataset_id} falls outside its declared period"
        )
