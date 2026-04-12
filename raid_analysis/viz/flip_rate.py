"""Flip rate bar chart for activation patching results."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def fig_flip_rate_by_domain(
    flip_rate_by_domain: dict[str, dict[str, Any]],
    overall_flip_rate: float | None = None,
    title: str = "Patching Flip Rate by Domain",
) -> plt.Figure:
    """Bar chart of human->AI flip rate per domain with error bars.

    Args:
        flip_rate_by_domain: ``{domain_name: {"flip_rate_mean": ...,
            "flip_rate_std": ..., "n_pairs_per_shuffle": ...}}``.
        overall_flip_rate: If provided, drawn as a horizontal reference line.
        title: Figure title.

    Returns:
        Matplotlib ``Figure``.
    """
    domains = sorted(flip_rate_by_domain.keys())
    means = [flip_rate_by_domain[d]["flip_rate_mean"] for d in domains]
    stds = [flip_rate_by_domain[d].get("flip_rate_std", 0.0) for d in domains]
    n_pairs = [flip_rate_by_domain[d].get("n_pairs_per_shuffle", 0) for d in domains]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(domains))
    bars = ax.bar(
        x, means, yerr=stds, capsize=4,
        color="#4daf4a", edgecolor="black", linewidth=0.5,
    )

    for i, (bar, n) in enumerate(zip(bars, n_pairs)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.01,
            f"n={n}", ha="center", va="bottom", fontsize=8,
        )

    if overall_flip_rate is not None:
        ax.axhline(
            overall_flip_rate, color="#e41a1c", linestyle="--",
            linewidth=1.5, label=f"Overall: {overall_flip_rate:.3f}",
        )
        ax.legend(fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_xlabel("Domain", fontsize=12)
    ax.set_ylabel("Flip rate (human \u2192 AI)", fontsize=12)
    ax.set_title(title, fontsize=14)
    max_val = max(m + s for m, s in zip(means, stds)) if means else 1.0
    ax.set_ylim(0, min(1.0, max_val * 1.3))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    return fig
