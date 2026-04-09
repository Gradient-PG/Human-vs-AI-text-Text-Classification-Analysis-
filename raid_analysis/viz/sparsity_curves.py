"""Accuracy-vs-sparsity curve with knee point annotation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..experiments.exp_sparse_probe import SweepPoint


def fig_accuracy_vs_sparsity(
    sweep_points: list[SweepPoint],
    knee_idx: int | None = None,
    title: str = "Accuracy vs Sparsity (L1 Probe Sweep)",
) -> plt.Figure:
    """Plot mean accuracy vs mean nonzero neuron count with error bars.

    Args:
        sweep_points: List of :class:`SweepPoint` from the sweep experiment.
        knee_idx: Index into *sweep_points* for the detected knee. If
            provided, the knee point is highlighted.
        title: Figure title.

    Returns:
        Matplotlib ``Figure``.
    """
    n_nonzero = np.array([sp.mean_n_nonzero for sp in sweep_points])
    accuracy = np.array([sp.mean_accuracy for sp in sweep_points])
    acc_std = np.array([sp.std_accuracy for sp in sweep_points])
    c_values = [sp.C for sp in sweep_points]

    order = np.argsort(n_nonzero)
    n_nonzero = n_nonzero[order]
    accuracy = accuracy[order]
    acc_std = acc_std[order]
    c_values_sorted = [c_values[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(
        n_nonzero, accuracy, yerr=acc_std,
        fmt="o-", color="#377eb8", capsize=4, linewidth=2,
        markersize=7, label="L2 eval probe accuracy",
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
    ax.set_ylabel("Accuracy (L2 eval probe, test fold)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig
