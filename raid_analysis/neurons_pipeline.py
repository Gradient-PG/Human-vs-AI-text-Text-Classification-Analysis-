"""Per-model neuron statistics, summary figures, and 2D embedding of all neurons (no clustering)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_full_neuron_matrix, load_stats
from .dim_reduction import get_dim_reduction
from .figures_embedding import fig_embedding_all_neurons
from .figures_neurons import (
    fig_activation_boxplots,
    fig_activation_scatter,
    fig_layer_distribution,
)
from .io import save_figure, write_text
from .summaries import neuron_summary_text


def run_neurons_analysis(
    model: str,
    results_path: Path,
    neurons_out: Path,
    neuron_viz: str = "pca",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Write layer/box/scatter figures, neurons_summary.txt, and neuron space embedding plot.

    Returns:
        neurons_df, disc_df, labels, X_all (full activation matrix).
    """
    neurons_out.mkdir(parents=True, exist_ok=True)
    figures_path = neurons_out / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    neurons_df, disc_df, labels = load_stats(results_path)
    print(f"    Neurons total: {len(neurons_df):,}  |  discriminative: {len(disc_df):,}")

    save_figure(
        fig_layer_distribution(neurons_df, disc_df, model),
        figures_path / "figure1_layer_distribution.png",
    )
    save_figure(
        fig_activation_boxplots(disc_df, labels, results_path, model),
        figures_path / "figure2_activation_boxplots.png",
    )
    save_figure(
        fig_activation_scatter(disc_df, model),
        figures_path / "figure3_activation_scatter.png",
    )
    write_text(
        neurons_out / "neurons_summary.txt",
        neuron_summary_text(neurons_df, disc_df, results_path, model),
    )

    X_all = build_full_neuron_matrix(neurons_df, results_path)
    print(f"    {neuron_viz.upper()} 2D embedding for neuron-space overview ...")
    dim_red = get_dim_reduction(neuron_viz)
    spec = dim_red.spec
    embedding = dim_red.fit_transform(X_all)
    save_figure(
        fig_embedding_all_neurons(
            embedding,
            neurons_df,
            model,
            axis_x=spec.axis_x,
            axis_y=spec.axis_y,
            embed_name=spec.name,
        ),
        figures_path / f"neuron_{spec.slug}_all.png",
    )

    return neurons_df, disc_df, labels, X_all
