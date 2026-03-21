"""Silhouette bars, dendrograms, merge-gap charts for hierarchical methods."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

from .constants import CLUSTER_PALETTE


def fig_silhouette(k_range, silhouette_scores: dict, optimal_k: int, model: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(k_range, [silhouette_scores[k] for k in k_range],
           color="steelblue", edgecolor="black")
    ax.axhline(silhouette_scores[optimal_k], color="red", linestyle="--", linewidth=2,
               alpha=0.5, label=f"Best (K={optimal_k})")
    ax.set_xlabel("Number of Clusters", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title(f"Silhouette Score vs K — {model}", fontsize=14, fontweight="bold")
    ax.set_xticks(list(k_range))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def fig_dendrogram(Z: np.ndarray, optimal_k: int, model: str) -> plt.Figure:
    colors = CLUSTER_PALETTE[:optimal_k]
    set_link_color_palette(colors)
    cut_height = Z[-(optimal_k - 1), 2]

    fig, ax = plt.subplots(figsize=(15, 7))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=cut_height,
               above_threshold_color="#888888")
    ax.axhline(y=cut_height, color="red", linestyle="--", linewidth=2.5,
               label=f"Cut at K={optimal_k}", zorder=10)
    ax.set_title(f"Hierarchical Clustering Dendrogram — {model}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Neuron Index", fontsize=12)
    ax.set_ylabel("Distance (Ward)", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    return fig


def fig_merge_distance_gaps(Z: np.ndarray, model: str, n_last: int = 15) -> plt.Figure:
    n_merges = min(n_last, len(Z))
    distances = Z[-n_merges:, 2]
    gaps = np.diff(distances)
    k_values = np.arange(n_merges, 1, -1)

    best_idx = int(np.argmax(gaps))
    suggested_k = k_values[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if i == best_idx else "steelblue" for i in range(len(gaps))]
    ax.bar(k_values, gaps, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("K  (number of clusters if cut here)", fontsize=12)
    ax.set_ylabel("Merge distance gap", fontsize=12)
    ax.set_title(f"Merge Distance Gaps — {model}", fontsize=14, fontweight="bold")
    ax.invert_xaxis()
    ax.annotate(f"Suggested K = {suggested_k}",
                xy=(suggested_k, gaps[best_idx]),
                xytext=(suggested_k + 1.5, gaps[best_idx] * 0.85),
                fontsize=11, fontweight="bold", color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
