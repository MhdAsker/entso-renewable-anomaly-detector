import os

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "plots"
)
os.makedirs(_PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def _save(name: str) -> None:
    path = os.path.join(_PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── 1 ─────────────────────────────────────────────────────────────────────────
def plot_forecast_error_overview(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))

    error = df["total_error_mw"]
    mu = error.mean()
    sigma = error.std()

    ax.plot(df.index, error, color="#2c7bb6", linewidth=0.6, alpha=0.9, label="Forecast error")

    # Shade anomaly regions
    anomaly = df["is_anomaly"].astype(bool)
    in_block = False
    block_start = None
    for ts, flag in zip(df.index, anomaly):
        if flag and not in_block:
            block_start = ts
            in_block = True
        elif not flag and in_block:
            ax.axvspan(block_start, ts, color="red", alpha=0.18)
            in_block = False
    if in_block:
        ax.axvspan(block_start, df.index[-1], color="red", alpha=0.18)

    # ±2σ lines
    ax.axhline(mu + 2 * sigma, color="#d73027", linewidth=1.0, linestyle="--", label="+2σ")
    ax.axhline(mu - 2 * sigma, color="#d73027", linewidth=1.0, linestyle="--", label="−2σ")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_xlabel("2023")
    ax.set_ylabel("Error (MW)")
    ax.set_title("German Renewable Generation Forecast Error 2023", fontsize=13, fontweight="bold")

    red_patch = mpatches.Patch(color="red", alpha=0.3, label="Anomaly region")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [red_patch], labels + ["Anomaly region"], loc="upper right", fontsize=9)

    _save("forecast_error_overview.png")


# ── 2 ─────────────────────────────────────────────────────────────────────────
def plot_anomaly_comparison(df: pd.DataFrame, predictions_dict: dict) -> None:
    model_names = list(predictions_dict.keys())
    n_rows = 1 + len(model_names)

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3 * n_rows), sharex=True)

    # Row 1 — actual error
    axes[0].plot(df.index, df["total_error_mw"], color="#2c7bb6", linewidth=0.5)
    axes[0].set_ylabel("Error (MW)")
    axes[0].set_title("Actual Forecast Error", fontsize=10)

    arrays = np.stack(list(predictions_dict.values()), axis=0)  # (n_models, T)
    all_agree = (arrays.sum(axis=0) == len(model_names)).astype(int)

    colors = ["#fc8d59", "#e34a33", "#b30000"]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        ax = axes[i + 1]
        preds = np.asarray(preds)

        # Normal flags
        normal_flag = np.where(preds == 1, 0.6, np.nan)
        ax.fill_between(df.index, 0, normal_flag, step="mid",
                        color=colors[i % len(colors)], alpha=0.5, label="Anomaly")

        # All-model agreement — dark red overlay
        agree_flag = np.where((preds == 1) & (all_agree == 1), 0.9, np.nan)
        ax.fill_between(df.index, 0, agree_flag, step="mid",
                        color="#67000d", alpha=0.85, label="All models agree")

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    axes[-1].set_xlabel("2023")
    fig.suptitle("Anomaly Flags by Model (dark red = all 3 agree)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("anomaly_comparison.png")


# ── 3 ─────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(y_true: np.ndarray, predictions_dict: dict) -> None:
    from sklearn.metrics import confusion_matrix

    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_true, np.asarray(preds))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds",
            ax=ax,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticklabels(["Normal", "Anomaly"], rotation=0)

    fig.suptitle("Confusion Matrices vs Ground-Truth Labels", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("confusion_matrices.png")


# ── 4 ─────────────────────────────────────────────────────────────────────────
def plot_reconstruction_error(lstm_scores: np.ndarray, df: pd.DataFrame, threshold: float) -> None:
    fig, ax = plt.subplots(figsize=(16, 4))

    scores = np.asarray(lstm_scores)
    above = scores > threshold

    ax.scatter(df.index[~above], scores[~above], s=2, color="#2c7bb6", alpha=0.5, label="Normal")
    ax.scatter(df.index[above], scores[above], s=4, color="#d73027", alpha=0.8, label="Anomaly")
    ax.axhline(threshold, color="#d73027", linewidth=1.2, linestyle="--",
               label=f"Threshold ({threshold:.4f})")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.set_xlabel("2023")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("LSTM Autoencoder — Reconstruction Error Over Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=3)

    _save("lstm_reconstruction_error.png")


# ── 5 ─────────────────────────────────────────────────────────────────────────
def plot_seasonal_anomaly_pattern(df: pd.DataFrame, predictions_dict: dict) -> None:
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for ax, (name, preds) in zip(axes, predictions_dict.items()):
        tmp = df[["is_anomaly"]].copy()
        tmp["pred"] = np.asarray(preds)
        tmp["hour"] = df.index.hour
        tmp["month"] = df.index.month

        pivot = (
            tmp.groupby(["hour", "month"])["pred"]
            .mean()
            .unstack(level="month")
            .reindex(columns=range(1, 13))
        )

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=pivot.values.max(),
            cbar_kws={"label": "Anomaly rate"},
            linewidths=0.0,
        )
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Hour of day")
        ax.set_xticklabels(month_labels, rotation=45, ha="right")
        ax.invert_yaxis()

    fig.suptitle(
        "Seasonal Anomaly Pattern — Hour × Month Heatmap",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _save("seasonal_anomaly_heatmap.png")
