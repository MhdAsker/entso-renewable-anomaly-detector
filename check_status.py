import os
import sys

# Force UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── helpers ───────────────────────────────────────────────────────────────────
OK  = "[OK] "
ERR = "[ERR]"
WRN = "[WRN]"

def section(title):
    print(f"\n{'═'*52}")
    print(f"  {title}")
    print(f"{'═'*52}")

# ══════════════════════════════════════════════════════
# 1. FOLDER STRUCTURE
# ══════════════════════════════════════════════════════
section("1. FOLDER STRUCTURE")
folders = [
    "data/raw",
    "data/processed",
    "models",
    "results/metrics",
    "results/plots",
    "src/models",
]
for f in folders:
    exists = os.path.isdir(f)
    print(f"  {OK if exists else ERR}  {f}/")

# ══════════════════════════════════════════════════════
# 2. FILE STATUS
# ══════════════════════════════════════════════════════
section("2. FILE STATUS")

import importlib

def try_import(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None

pd = try_import("pandas")
joblib_mod = try_import("joblib")
torch_mod  = try_import("torch")

# ── entso_raw.parquet ─────────────────────────────────
print("\n  data/raw/entso_raw.parquet")
raw_path = "data/raw/entso_raw.parquet"
raw_ok = False
if os.path.exists(raw_path) and pd:
    raw_ok = True
    df_raw = pd.read_parquet(raw_path)
    print(f"    {OK} EXISTS")
    print(f"       Shape      : {df_raw.shape}")
    print(f"       Columns    : {list(df_raw.columns)}")
    print(f"       Date range : {df_raw.index.min()} → {df_raw.index.max()}")
    print(f"       Null counts:")
    for col, cnt in df_raw.isnull().sum().items():
        print(f"         {col}: {cnt}")
    key_cols = [c for c in ["actual_wind_total_mw","actual_solar_mw","forecast_renewable_mw"] if c in df_raw.columns]
    print(f"       Key column means:")
    for col in key_cols:
        print(f"         {col}: {df_raw[col].mean():.1f} MW")
else:
    print(f"    {ERR} MISSING — run: python src/fetch.py")

# ── features.parquet ──────────────────────────────────
print("\n  data/processed/features.parquet")
feat_path = "data/processed/features.parquet"
feat_ok = False
df_feat = None
if os.path.exists(feat_path) and pd:
    feat_ok = True
    df_feat = pd.read_parquet(feat_path)
    n_total  = len(df_feat)
    n_anom   = int(df_feat["is_anomaly"].sum()) if "is_anomaly" in df_feat.columns else 0
    n_normal = n_total - n_anom
    print(f"    {OK} EXISTS")
    print(f"       Shape   : {df_feat.shape}")
    print(f"       Columns : {list(df_feat.columns)}")
    print(f"       Normal  hours : {n_normal} ({100*n_normal/n_total:.1f}%)")
    print(f"       Anomaly hours : {n_anom}  ({100*n_anom/n_total:.1f}%)")
else:
    print(f"    {ERR} MISSING — run: python src/features.py")

# ── iforest.pkl ───────────────────────────────────────
print("\n  models/iforest.pkl")
iforest_ok = False
if os.path.exists("models/iforest.pkl") and joblib_mod:
    iforest_ok = True
    artifact = joblib_mod.load("models/iforest.pkl")
    mdl = artifact.get("model") if isinstance(artifact, dict) else artifact
    print(f"    {OK} EXISTS")
    print(f"       Type         : {type(mdl).__name__}")
    print(f"       n_estimators : {getattr(mdl, 'n_estimators', 'N/A')}")
else:
    print(f"    {ERR} MISSING")

# ── lstm_autoencoder.pt ───────────────────────────────
print("\n  models/lstm_autoencoder.pt")
lstm_ok = False
if os.path.exists("models/lstm_autoencoder.pt"):
    lstm_ok = True
    size_mb = os.path.getsize("models/lstm_autoencoder.pt") / 1_048_576
    print(f"    {OK} EXISTS  ({size_mb:.2f} MB)")
else:
    print(f"    {ERR} MISSING")

# ── anomaly_results.csv ───────────────────────────────
print("\n  results/metrics/anomaly_results.csv")
csv_path = "results/metrics/anomaly_results.csv"
csv_ok = False
df_csv = None
if os.path.exists(csv_path) and pd:
    csv_ok = True
    df_csv = pd.read_csv(csv_path)
    print(f"    {OK} EXISTS")
    print(f"       {df_csv.to_string(index=False)}")
else:
    print(f"    {ERR} MISSING")

# ── plots ─────────────────────────────────────────────
print("\n  results/plots/")
expected_plots = [
    "forecast_error_overview.png",
    "anomaly_comparison.png",
    "confusion_matrices.png",
    "lstm_reconstruction_error.png",
    "seasonal_anomaly_heatmap.png",
]
found_plots = []
if os.path.isdir("results/plots"):
    found_plots = [f for f in os.listdir("results/plots") if f.endswith(".png")]
print(f"    Found {len(found_plots)}/5 expected PNGs:")
for p in expected_plots:
    flag = OK if p in found_plots else ERR
    print(f"      {flag}  {p}")
plots_ok = len(found_plots) >= 5

# ══════════════════════════════════════════════════════
# 3. IMPORTS CHECK
# ══════════════════════════════════════════════════════
section("3. IMPORTS CHECK")

import_map = {
    "entsoe":      "entsoe",
    "pandas":      "pandas",
    "numpy":       "numpy",
    "sklearn":     "scikit-learn",
    "torch":       "torch",
    "matplotlib":  "matplotlib",
    "seaborn":     "seaborn",
    "scipy":       "scipy",
    "joblib":      "joblib",
}
for mod_name, pkg_name in import_map.items():
    try:
        mod = importlib.import_module(mod_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {OK}  {pkg_name} {version}")
    except ImportError as e:
        print(f"  {ERR}  {pkg_name} NOT INSTALLED  ({e})")

# ══════════════════════════════════════════════════════
# 4. ENV CHECK
# ══════════════════════════════════════════════════════
section("4. ENV CHECK")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.environ.get("ENTSOE_API_KEY", "")
if api_key:
    masked = api_key[:8] + "…" if len(api_key) >= 8 else api_key
    print(f"  {OK}  ENTSOE_API_KEY loaded (first 8 chars: {masked})")
else:
    print(f"  {ERR}  ENTSOE_API_KEY missing — add to .env")

# ══════════════════════════════════════════════════════
# 5. QUICK DATA SANITY CHECK
# ══════════════════════════════════════════════════════
section("5. QUICK DATA SANITY CHECK")
if df_feat is not None and pd:
    n_rows = len(df_feat)

    # >10% nulls
    null_pct = df_feat.isnull().mean()
    high_null = null_pct[null_pct > 0.10]
    if high_null.empty:
        print(f"  {OK}  No column has >10% nulls")
    else:
        for col, pct in high_null.items():
            print(f"  {WRN} Column '{col}' has {pct*100:.1f}% nulls")

    # is_anomaly exists
    if "is_anomaly" in df_feat.columns:
        print(f"  {OK}  is_anomaly column exists")
    else:
        print(f"  {ERR}  is_anomaly column MISSING")

    # Anomaly rate
    anom_rate = df_feat["is_anomaly"].mean() * 100 if "is_anomaly" in df_feat.columns else 0
    if 1.0 <= anom_rate <= 20.0:
        print(f"  {OK}  Anomaly rate {anom_rate:.1f}% — reasonable (1–20%)")
    else:
        print(f"  {WRN} Anomaly rate {anom_rate:.1f}% — suspicious (expected 1–20%)")

    # Date range covers full year
    idx_min = df_feat.index.min()
    idx_max = df_feat.index.max()
    span_days = (idx_max - idx_min).days
    if span_days >= 300:
        print(f"  {OK}  Date range {idx_min.date()} → {idx_max.date()} ({span_days} days)")
    else:
        print(f"  {ERR}  Date range only {span_days} days — expected ~365")

    # total_error_mw has both signs
    if "total_error_mw" in df_feat.columns:
        has_pos = (df_feat["total_error_mw"] > 0).any()
        has_neg = (df_feat["total_error_mw"] < 0).any()
        if has_pos and has_neg:
            print(f"  {OK}  total_error_mw has both positive and negative values")
        else:
            print(f"  {ERR}  total_error_mw is one-sided — check feature engineering")
else:
    print(f"  {WRN} Skipped — features.parquet not available")

# ══════════════════════════════════════════════════════
# 6. PIPELINE STATUS SUMMARY
# ══════════════════════════════════════════════════════
section("PIPELINE STATUS — entso-renewable-anomaly-detector")

steps = [
    (raw_ok,     "Step 1: Raw data fetched",    "entso_raw.parquet",     "python src/fetch.py"),
    (feat_ok,    "Step 2: Features engineered", "features.parquet",      "python src/features.py"),
    (True,       "Step 3: Z-Score model",       "in-memory only",        None),
    (iforest_ok, "Step 4: Isolation Forest",    "iforest.pkl",           "python run_detector.py"),
    (lstm_ok,    "Step 5: LSTM Autoencoder",    "lstm_autoencoder.pt",   "python run_detector.py"),
    (csv_ok,     "Step 6: Results saved",       "anomaly_results.csv",   "python run_detector.py"),
    (plots_ok,   f"Step 7: Plots generated",    f"{len(found_plots)}/5 PNGs found", "python run_detector.py"),
]

next_action = None
for ok, label, detail, cmd in steps:
    flag = OK if ok else ERR
    print(f"  [{flag}] {label:<35} ({detail})")
    if not ok and next_action is None and cmd:
        next_action = cmd

print()
if next_action:
    print(f"  NEXT ACTION: {next_action}")
else:
    print(f"  {OK}  ALL STEPS COMPLETE — pipeline is fully built")

# ══════════════════════════════════════════════════════
# 7. BENCHMARK RESULTS (if available)
# ══════════════════════════════════════════════════════
if csv_ok and df_csv is not None and pd:
    section("BENCHMARK RESULTS")
    df_sorted = df_csv.sort_values("f1", ascending=False)
    print(df_sorted.to_string(index=False))
    best = df_sorted.iloc[0]
    baseline_rows = df_csv[df_csv["model_name"].str.contains("Z-Score", case=False, na=False)]
    if not baseline_rows.empty:
        baseline_f1 = baseline_rows.iloc[0]["f1"]
        improvement = 100 * (best["f1"] - baseline_f1) / baseline_f1 if baseline_f1 > 0 else float("nan")
        print(f"\n  Best F1    : {best['model_name']}  (F1={best['f1']:.2f})")
        print(f"  vs baseline: +{improvement:.1f}% over Z-Score")
    print(f"\n  Largest event caught: (run run_detector.py for timestamp detail)")
