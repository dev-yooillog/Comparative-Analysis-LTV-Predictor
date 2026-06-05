import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict

def rmse(y_true: np.ndarray, y_pred: np.ndarray, events: np.ndarray = None) -> float:
    if events is not None:
        mask = events == 1
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def concordance_index(durations: np.ndarray, pred_scores: np.ndarray, events: np.ndarray) -> float:
    durations = np.asarray(durations, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    events = np.asarray(events, dtype=np.float64)

    event_idx = np.where(events == 1)[0]
    concordant = 0
    total = 0

    for i in event_idx:
        mask = durations > durations[i]
        if mask.sum() == 0:
            continue
        score_i = pred_scores[i]
        scores_j = pred_scores[mask]
        concordant += (score_i > scores_j).sum() + 0.5 * (score_i == scores_j).sum()
        total += len(scores_j)

    return concordant / total if total > 0 else 0.5

def evaluate_model(model_name: str, durations_true: np.ndarray, pred_durations: np.ndarray, events: np.ndarray) -> Dict:
    hazard_scores = -pred_durations
    r = rmse(durations_true, pred_durations, events)
    c = concordance_index(durations_true, hazard_scores, events)
    return {"model": model_name, "rmse": r, "c_index": c}

def setup_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.dpi": 120,
    })

def plot_comparison(results: list, save_path: str = None):
    setup_plot_style()
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#4C72B0", "#DD8452"]

    axes[0].bar(df["model"], df["rmse"], color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("RMSE (lower is better)")
    for bar, val in zip(axes[0].patches, df["rmse"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}", ha="center", va="bottom")

    axes[1].bar(df["model"], df["c_index"], color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("C-index (higher is better)")
    axes[1].set_ylim(0, 1)
    for bar, val in zip(axes[1].patches, df["c_index"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_training_curve(train_losses, val_losses, save_path: str = None):
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_title("DeepSurv Training Curve")
    ax.legend()
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_rfm_distribution(rfm: pd.DataFrame, save_path: str = None):
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    data_cols = [("recency_days", "Recency"), ("frequency", "Frequency"), ("monetary", "Monetary")]

    for ax, (col, title) in zip(axes, data_cols):
        clip = rfm[col].quantile(0.99)
        ax.hist(rfm[col].clip(upper=clip), bins=50, edgecolor="white")
        ax.set_title(title)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def plot_predicted_vs_actual(dur_true, pred_bgnbd, pred_deepsurv, events, save_path=None):
    setup_plot_style()
    mask = events == 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, preds, title in zip(axes, [pred_bgnbd[mask], pred_deepsurv[mask]], ["BG/NBD", "DeepSurv"]):
        ax.scatter(dur_true[mask], preds, alpha=0.15, s=10)
        lim = max(dur_true[mask].max(), preds.max())
        ax.plot([0, lim], [0, lim], "r--")
        ax.set_title(f"{title}: Pred vs Actual")

    plt.tight_layout()
    if save_path: plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()