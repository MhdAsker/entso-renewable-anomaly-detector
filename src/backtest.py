"""Expanding-window model backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestFold:
    fold: int
    fit: slice
    validation: slice
    test: slice


def expanding_window_folds(
    n_rows: int,
    initial_fit_fraction: float = 0.40,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.15,
) -> list[BacktestFold]:
    """Return non-overlapping tests with an expanding historical fit window."""
    if n_rows < 100:
        raise ValueError("At least 100 rows are required for backtesting")
    if min(initial_fit_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Fold fractions must be positive")
    fit_size = int(n_rows * initial_fit_fraction)
    validation_size = int(n_rows * validation_fraction)
    test_size = int(n_rows * test_fraction)
    folds = []
    fold_number = 1
    while fit_size + validation_size < n_rows:
        validation_end = fit_size + validation_size
        test_end = min(validation_end + test_size, n_rows)
        if test_end <= validation_end:
            break
        folds.append(
            BacktestFold(
                fold=fold_number,
                fit=slice(0, fit_size),
                validation=slice(fit_size, validation_end),
                test=slice(validation_end, test_end),
            )
        )
        fit_size += test_size
        fold_number += 1
    return folds
