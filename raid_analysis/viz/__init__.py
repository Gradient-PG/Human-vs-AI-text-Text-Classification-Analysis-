"""Visualization and figure generation."""

from .dim_reduction import (
    DimReductionSpec,
    DimReductionStrategy,
    PCAReduction,
    UMAPReduction,
    get_dim_reduction,
)
from .embedding import fig_embedding_all_neurons, fig_full_space_clustering
from .hierarchy import fig_dendrogram, fig_merge_distance_gaps, fig_silhouette
from .neurons import fig_activation_boxplots, fig_activation_scatter, fig_layer_distribution
from .sparsity_curves import fig_accuracy_vs_sparsity

__all__ = [
    "DimReductionSpec",
    "DimReductionStrategy",
    "PCAReduction",
    "UMAPReduction",
    "get_dim_reduction",
    "fig_embedding_all_neurons",
    "fig_full_space_clustering",
    "fig_dendrogram",
    "fig_merge_distance_gaps",
    "fig_silhouette",
    "fig_activation_boxplots",
    "fig_activation_scatter",
    "fig_layer_distribution",
    "fig_accuracy_vs_sparsity",
]
