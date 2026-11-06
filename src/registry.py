"""Versioned model registry with reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "models" / "registry"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def register_run(
    *,
    feature_path: str | Path,
    feature_cols: list[str],
    fit_frame: pd.DataFrame,
    thresholds: dict[str, float],
    score_reference: dict[str, dict[str, float]],
    metrics: list[dict],
    split: dict[str, str],
    label_column: str,
    run_id: str | None = None,
    registry_dir: str | Path = DEFAULT_REGISTRY,
) -> Path:
    registry_dir = Path(registry_dir)
    created_at = datetime.now(timezone.utc)
    run_id = run_id or created_at.strftime("%Y%m%dT%H%M%SZ")
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id may contain only letters, numbers, '-' and '_'")
    run_dir = registry_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Registry run already exists: {run_id}")
    run_dir.mkdir(parents=True)

    artifacts = {
        "isolation_forest": ROOT / "models" / "iforest.pkl",
        "lstm_autoencoder": ROOT / "models" / "lstm_autoencoder.pt",
    }
    artifact_manifest = {}
    for name, source in artifacts.items():
        destination = run_dir / source.name
        shutil.copy2(source, destination)
        artifact_manifest[name] = {
            "file": destination.name,
            "sha256": file_sha256(destination),
        }

    feature_stats = {}
    for column in feature_cols:
        values = fit_frame[column].to_numpy(dtype=float)
        feature_stats[column] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p01": float(np.quantile(values, 0.01)),
            "p99": float(np.quantile(values, 0.99)),
        }

    packages = {}
    for package in ["pandas", "numpy", "scikit-learn", "torch", "joblib"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "unknown"

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "git_revision": _git_revision(),
        "feature_file_sha256": file_sha256(feature_path),
        "feature_columns": feature_cols,
        "feature_stats": feature_stats,
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "score_reference": score_reference,
        "metrics": metrics,
        "split": split,
        "label_column": label_column,
        "packages": packages,
        "artifacts": artifact_manifest,
    }
    _atomic_json(run_dir / "manifest.json", manifest)
    _atomic_json(registry_dir / "latest.json", {"run_id": run_id})
    return run_dir


def resolve_run(
    version: str = "latest", registry_dir: str | Path = DEFAULT_REGISTRY
) -> tuple[Path, dict]:
    registry_dir = Path(registry_dir)
    if version == "latest":
        pointer = json.loads((registry_dir / "latest.json").read_text(encoding="utf-8"))
        version = pointer["run_id"]
    run_dir = registry_dir / version
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"].values():
        path = run_dir / artifact["file"]
        if file_sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"Artifact checksum mismatch: {path}")
    return run_dir, manifest
