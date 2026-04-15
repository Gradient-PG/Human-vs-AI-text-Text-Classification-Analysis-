"""Accuracy-vs-sparsity curve with knee point annotation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def fig_accuracy_vs_sparsity(
    sweep_curve: list[dict],
    knee_idx: int | None = None,
    title: str = "Accuracy vs Sparsity (L1 Probe Sweep)",
) -> plt.Figure:
    """Plot validation accuracy vs nonzero neuron count with knee annotation.

    Args:
        sweep_curve: List of dicts with keys ``C``, ``val_accuracy``,
            ``n_nonzero`` — as produced by ``SparseProbeSelector`` in
            ``SelectionResult.metadata["sweep_curve"]``.
        knee_idx: Index into *sweep_curve* for the detected knee.
        title: Figure title.

    Returns:
        Matplotlib ``Figure``.
    """
    n_nonzero = np.array([p["n_nonzero"] for p in sweep_curve])
    accuracy = np.array([p["val_accuracy"] for p in sweep_curve])
    c_values = [p["C"] for p in sweep_curve]

    order = np.argsort(n_nonzero)
    n_nonzero = n_nonzero[order]
    accuracy = accuracy[order]
    c_values_sorted = [c_values[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        n_nonzero, accuracy,
        "o-", color="#377eb8", linewidth=2,
        markersize=7, label="L1 probe val accuracy",
    )

    if knee_idx is not None:
        knee_sorted = int(np.where(order == knee_idx)[0][0])
        ax.axvline(
            n_nonzero[knee_sorted], color="#e41a1c",
            linestyle="--", alpha=0.7, label="Knee point",
        )
        ax.scatter(
            [n_nonzero[knee_sorted]], [accuracy[knee_sorted]],
            color="#e41a1c", s=150, zorder=5, edgecolors="black",
        )
        ax.annotate(
            f"C={c_values_sorted[knee_sorted]}",
            xy=(n_nonzero[knee_sorted], accuracy[knee_sorted]),
            xytext=(15, -15), textcoords="offset points",
            fontsize=10, color="#e41a1c",
            arrowprops=dict(arrowstyle="->", color="#e41a1c"),
        )

    for i, c_val in enumerate(c_values_sorted):
        ax.annotate(
            f"{c_val}",
            xy=(n_nonzero[i], accuracy[i]),
            xytext=(0, 10), textcoords="offset points",
            fontsize=7, ha="center", alpha=0.6,
        )

    ax.set_xlabel("Number of nonzero neurons", fontsize=12)
    ax.set_ylabel("Validation accuracy (L1 probe)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig
