"""Per-model neuron summary figures (layer distribution, boxplots, scatter)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import ALL_LAYERS, AUC_HIGH, AUC_LOW
from ..data.activations import load_activation_column


def _apply_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")


def _layer_stats(neurons_df: pd.DataFrame, disc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer in ALL_LAYERS:
        lyr = neurons_df[neurons_df["layer"] == layer]
        lyr_disc = lyr[lyr["discriminative"]]
        rows.append({
            "layer": layer,
            "pct_discriminative": len(lyr_disc) / len(lyr) * 100 if len(lyr) else 0,
            "ai_preferring": (lyr_disc["auc"] > AUC_HIGH).sum(),
            "human_preferring": (lyr_disc["auc"] < AUC_LOW).sum(),
        })
    return pd.DataFrame(rows)


def fig_layer_distribution(
    neurons_df: pd.DataFrame, disc_df: pd.DataFrame, model: str
) -> plt.Figure:
    _apply_style()
    layer_df = _layer_stats(neurons_df, disc_df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = layer_df["layer"]

    mean_pct = layer_df["pct_discriminative"].mean()
    colors = ["#e74c3c" if p >= mean_pct else "#3498db" for p in layer_df["pct_discriminative"]]
    ax1.bar(x, layer_df["pct_discriminative"], 0.6, color=colors, alpha=0.8,
            edgecolor="black", linewidth=0.5)
    ax1.axhline(mean_pct, color="black", linestyle="--", linewidth=2,
                label=f"Mean ({mean_pct:.1f}%)")
    ax1.set_xlabel("BERT Layer", fontsize=12, fontweight="bold")
    ax1.set_ylabel("% Discriminative", fontsize=12, fontweight="bold")
    ax1.set_title("Discriminative % per Layer", fontsize=13, fontweight="bold")
    ax1.set_xticks(ALL_LAYERS)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ylim = ax1.get_ylim()[1]
    for label, pos in [("Early", 2.5), ("Middle", 6.5), ("Late", 10.5)]:
        ax1.text(pos, ylim * 0.95, label, ha="center", fontsize=10,
                 fontstyle="italic", color="gray")

    ratio = layer_df["ai_preferring"] / layer_df["human_preferring"].replace(0, 1)
    colors2 = ["#e74c3c" if r > 1 else "#27ae60" for r in ratio]
    ax2.bar(x, ratio, 0.6, color=colors2, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax2.axhline(1, color="black", linestyle="-", linewidth=2, label="Equal (1:1)")
    ax2.set_xlabel("BERT Layer", fontsize=12, fontweight="bold")
    ax2.set_ylabel("AI : Human Ratio", fontsize=12, fontweight="bold")
    ax2.set_title("AI vs Human Preference Ratio", fontsize=13, fontweight="bold")
    ax2.set_xticks(ALL_LAYERS)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Model: {model}", fontsize=11, color="gray")
    fig.tight_layout()
    return fig


def fig_activation_boxplots(
    disc_df: pd.DataFrame, labels: np.ndarray, results_path: Path, model: str
) -> plt.Figure:
    _apply_style()
    if disc_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No discriminative neurons found",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle(f"Top 10 Discriminative Neurons — {model}", fontsize=14, fontweight="bold")
        return fig

    top10 = disc_df.nlargest(10, "auc_deviation")
    ai_mask, human_mask = labels == 1, labels == 0

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    for ax, (_, neuron) in zip(axes.flatten(), top10.iterrows()):
        layer, n_idx = int(neuron["layer"]), int(neuron["neuron_idx"])
        auc = neuron["auc"]
        acts = load_activation_column(results_path, layer, n_idx)

        bp = ax.boxplot([acts[ai_mask], acts[human_mask]],
                        tick_labels=["AI", "Human"], patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor("#e74c3c")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("#27ae60")
        bp["boxes"][1].set_alpha(0.7)
        ax.set_title(f"L{layer}:N{n_idx}\nAUC={auc:.3f} ({'AI' if auc > 0.5 else 'Human'})",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Top 10 Discriminative Neurons — {model}", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def fig_activation_scatter(disc_df: pd.DataFrame, model: str) -> plt.Figure:
    _apply_style()
    if disc_df.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, "No discriminative neurons found",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(f"AI vs Human Mean Activation — {model}", fontsize=13, fontweight="bold")
        return fig

    fig, ax = plt.subplots(figsize=(8, 8))
    ai_mask = disc_df["auc"] > AUC_HIGH
    hu_mask = disc_df["auc"] < AUC_LOW

    ax.scatter(disc_df.loc[ai_mask, "ai_median"], disc_df.loc[ai_mask, "human_median"],
               c="#e74c3c", alpha=0.6, s=40, label=f"AI-preferring (n={ai_mask.sum()})")
    ax.scatter(disc_df.loc[hu_mask, "ai_median"], disc_df.loc[hu_mask, "human_median"],
               c="#27ae60", alpha=0.6, s=40, label=f"Human-preferring (n={hu_mask.sum()})")

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=2, label="Equal activation")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean Activation on AI Samples", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Activation on Human Samples", fontsize=12, fontweight="bold")
    ax.set_title(f"AI vs Human Mean Activation — {model}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    ax.text(0.95, 0.05, "Above line: Higher for Human\nBelow line: Higher for AI",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    fig.tight_layout()
    return fig
