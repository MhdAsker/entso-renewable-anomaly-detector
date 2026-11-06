# Optional external data contracts

`context.csv` may contain an hourly `timestamp` plus any of:
`temperature_c`, `wind_speed_mps`, `irradiance_wm2`, `actual_load_mw`,
`forecast_load_mw`, and `net_import_mw`. Missingness indicators are retained.

`event_labels.csv` must contain `timestamp,is_event`, where `is_event` is 0 or 1.
Attach it with `labels.attach_event_labels`, then run benchmarks with
`--label-column event_label`. Sources and label definitions should be recorded
alongside the file; proxy labels must not be mixed with independent outcomes.
