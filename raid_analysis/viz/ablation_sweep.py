"""Ablation sweep curve: accuracy drop vs number of ablated neurons."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def fig_ablation_sweep(
    ablation_sweep_data: list[dict[str, Any]],
    baseline_accuracy: float | None = None,
    title: str = "Ablation Sweep: Selected vs Random Neurons",
) -> plt.Figure:
    """Plot accuracy drop for selected vs random neuron ablation.

    Args:
        ablation_sweep_data: List of dicts from :class:`AblationEvaluator`,
            each with keys ``k``, ``selected_drop``, ``random_drop_mean``,
            ``random_drop_std``.
        baseline_accuracy: If provided, annotated on the plot.
        title: Figure title.

    Returns:
        Matplotlib ``Figure``.
    """
    ks = [d["k"] for d in ablation_sweep_data]
    sel_drops = [d["selected_drop"] for d in ablation_sweep_data]
    rand_means = [d["random_drop_mean"] for d in ablation_sweep_data]
    rand_stds = [d["random_drop_std"] for d in ablation_sweep_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        ks, sel_drops, "o-", color="#e41a1c", linewidth=2,
        markersize=7, label="Selected neurons",
    )

    ax.errorbar(
        ks, rand_means, yerr=rand_stds,
        fmt="s--", color="#999999", capsize=4, linewidth=1.5,
        markersize=5, label="Random neurons (mean ± std)",
    )

    ax.fill_between(
        ks,
        np.array(rand_means) - np.array(rand_stds),
        np.array(rand_means) + np.array(rand_stds),
        alpha=0.15, color="#999999",
    )

    if baseline_accuracy is not None:
        ax.annotate(
            f"Baseline acc: {baseline_accuracy:.4f}",
            xy=(0.02, 0.98), xycoords="axes fraction",
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )

    ax.set_xlabel("Number of neurons ablated (k)", fontsize=12)
    ax.set_ylabel("Accuracy drop", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig
