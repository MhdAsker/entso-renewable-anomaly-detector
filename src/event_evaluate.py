"""Event-level evaluation without point adjustment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binary_events(values: np.ndarray, index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    values = np.asarray(values, dtype=int)
    if len(values) != len(index):
        raise ValueError("values and index must have equal length")
    events = []
    start = None
    for position, value in enumerate(values):
        if position and start is not None and index[position] - index[position - 1] > pd.Timedelta(hours=1):
            events.append((pd.Timestamp(start), pd.Timestamp(index[position - 1])))
            start = None
        if value and start is None:
            start = index[position]
        if start is not None and (not value or position == len(values) - 1):
            end_position = position if value and position == len(values) - 1 else position - 1
            events.append((pd.Timestamp(start), pd.Timestamp(index[end_position])))
            start = None
    return events


def compute_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: pd.DatetimeIndex,
    tolerance_hours: int = 0,
) -> dict:
    """One-to-one overlap matching of contiguous true and predicted episodes."""
    true_events = binary_events(y_true, index)
    predicted_events = binary_events(y_pred, index)
    tolerance = pd.Timedelta(hours=tolerance_hours)
    unmatched = set(range(len(predicted_events)))
    delays = []
    matched = 0
    for true_start, true_end in true_events:
        candidates = []
        for pred_index in unmatched:
            pred_start, pred_end = predicted_events[pred_index]
            if pred_end >= true_start - tolerance and pred_start <= true_end + tolerance:
                overlap = min(pred_end, true_end) - max(pred_start, true_start)
                candidates.append((overlap, pred_index))
        if not candidates:
            continue
        _, best = max(candidates, key=lambda item: item[0])
        unmatched.remove(best)
        matched += 1
        delay = (predicted_events[best][0] - true_start).total_seconds() / 3600
        delays.append(max(0.0, delay))

    event_precision = matched / len(predicted_events) if predicted_events else 0.0
    event_recall = matched / len(true_events) if true_events else 0.0
    denominator = event_precision + event_recall
    event_f1 = 2 * event_precision * event_recall / denominator if denominator else 0.0
    duration_days = max((index[-1] - index[0]).total_seconds() / 86400, 1 / 24)
    return {
        "n_true_events": len(true_events),
        "n_predicted_events": len(predicted_events),
        "matched_events": matched,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "mean_detection_delay_hours": float(np.mean(delays)) if delays else float("nan"),
        "median_detection_delay_hours": float(np.median(delays)) if delays else float("nan"),
        "false_events_per_30d": len(unmatched) * 30 / duration_days,
    }
