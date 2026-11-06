"""Generate machine-readable status and a dependency-free HTML monitoring page."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CALENDAR_FEATURES = {
    "hour", "month", "season", "is_night", "is_weekend",
    "hour_sin", "hour_cos", "week_sin", "week_cos", "year_sin", "year_cos",
}


def build_monitoring_report(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    alerts: pd.DataFrame,
    manifest: dict,
    output_dir: str | Path,
    recent_hours: int = 168,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = pd.Timestamp(features.index.max())
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    freshness_hours = max(0.0, (now - latest).total_seconds() / 3600)
    freshness_status = "ok" if freshness_hours <= 2 else "warning" if freshness_hours <= 6 else "critical"

    recent_features = features.tail(recent_hours)
    drift = []
    for column in manifest["feature_columns"]:
        reference = manifest["feature_stats"][column]
        scale = max(float(reference["std"]), np.finfo(float).eps)
        mean_shift = abs(float(recent_features[column].mean()) - float(reference["mean"])) / scale
        std_ratio = float(recent_features[column].std()) / scale
        status = (
            "informational"
            if column in CALENDAR_FEATURES
            else "critical" if mean_shift >= 2 else "warning" if mean_shift >= 1 else "ok"
        )
        drift.append(
            {
                "feature": column,
                "mean_shift_sigma": mean_shift,
                "std_ratio": std_ratio,
                "status": status,
            }
        )

    recent_cutoff = latest - pd.Timedelta(hours=recent_hours)
    recent_scores = scores[pd.to_datetime(scores["timestamp"]) >= recent_cutoff]
    alert_rates = (
        recent_scores.groupby("model")["is_anomaly"].mean().fillna(0).to_dict()
        if not recent_scores.empty else {}
    )
    severity_counts = alerts["severity"].value_counts().to_dict() if not alerts.empty else {}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": manifest["run_id"],
        "latest_data_timestamp": latest.isoformat(),
        "freshness_hours": freshness_hours,
        "freshness_status": freshness_status,
        "recent_window_hours": recent_hours,
        "alert_rates": {key: float(value) for key, value in alert_rates.items()},
        "severity_counts": {key: int(value) for key, value in severity_counts.items()},
        "drift": drift,
    }
    (output_dir / "status.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    drift_rows = "".join(
        f"<tr><td>{html.escape(item['feature'])}</td>"
        f"<td>{item['mean_shift_sigma']:.2f}</td><td>{item['std_ratio']:.2f}</td>"
        f"<td class='{item['status']}'>{item['status']}</td></tr>"
        for item in sorted(drift, key=lambda item: item["mean_shift_sigma"], reverse=True)
    )
    rate_rows = "".join(
        f"<tr><td>{html.escape(model)}</td><td>{rate:.2%}</td></tr>"
        for model, rate in sorted(alert_rates.items())
    ) or "<tr><td colspan='2'>No recent scores</td></tr>"
    dashboard = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Renewable anomaly monitor</title><style>
body{{font-family:system-ui;margin:2rem;max-width:1100px}} table{{border-collapse:collapse;width:100%}}
th,td{{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}} .ok{{color:#087830}}
.warning{{color:#a15c00}} .critical{{color:#b00020;font-weight:bold}} .informational{{color:#555}} .cards{{display:flex;gap:1rem}}
.card{{padding:1rem;border:1px solid #ddd;border-radius:.5rem;flex:1}}</style></head><body>
<h1>Renewable anomaly monitor</h1><div class='cards'><div class='card'><b>Model</b><br>{html.escape(manifest['run_id'])}</div>
<div class='card'><b>Latest data</b><br>{latest}</div><div class='card'><b>Freshness</b><br>
<span class='{freshness_status}'>{freshness_hours:.1f} h</span></div></div>
<h2>Recent alert rate</h2><table><tr><th>Model</th><th>Rate</th></tr>{rate_rows}</table>
<h2>Feature drift</h2><table><tr><th>Feature</th><th>Mean shift (σ)</th><th>Std ratio</th><th>Status</th></tr>
{drift_rows}</table></body></html>"""
    (output_dir / "dashboard.html").write_text(dashboard, encoding="utf-8")
    return report
