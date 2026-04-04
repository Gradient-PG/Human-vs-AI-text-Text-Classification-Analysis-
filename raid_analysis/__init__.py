"""RAID multi-model neuron analysis: embedding strategies, clustering, figures."""

from .clustering import (
    CLUSTERING_STRATEGY_IDS,
    ClusteringResult,
    ClusteringStrategy,
    HDBSCANStrategy,
    KMeansSilhouetteStrategy,
    WardGapStrategy,
    WardSilhouetteStrategy,
    clustering_strategies_from_names,
    default_clustering_strategies,
    run_clustering_strategies,
)
from .constants import ALL_LAYERS, ALPHA, AUC_HIGH, AUC_LOW
from .data import build_full_neuron_matrix, load_activation_column, load_stats
from .dim_reduction import PCAReduction, UMAPReduction, get_dim_reduction
from .io import save_figure, write_text
from .run_analysis import analyze_raid_model
from .summaries import (
    clustering_summary_text,
    exemplar_text,
    neuron_summary_text,
)
from .traits import (
    add_derived_neuron_columns,
    build_trait_matrix,
    sanitize_trait_matrix,
    traits_matrix_to_csv,
)

analyze_model = analyze_raid_model  # backward-compatible name

__all__ = [
    # Constants
    "ALL_LAYERS",
    "ALPHA",
    "AUC_HIGH",
    "AUC_LOW",
    # Data loading
    "build_full_neuron_matrix",
    "load_activation_column",
    "load_stats",
    # Clustering
    "CLUSTERING_STRATEGY_IDS",
    "ClusteringResult",
    "ClusteringStrategy",
    "HDBSCANStrategy",
    "KMeansSilhouetteStrategy",
    "WardGapStrategy",
    "WardSilhouetteStrategy",
    "clustering_strategies_from_names",
    "default_clustering_strategies",
    "run_clustering_strategies",
    # Dimensionality reduction
    "PCAReduction",
    "UMAPReduction",
    "get_dim_reduction",
    # Orchestration
    "analyze_model",
    "analyze_raid_model",
    # IO
    "save_figure",
    "write_text",
    # Summaries
    "clustering_summary_text",
    "exemplar_text",
    "neuron_summary_text",
    # Traits
    "add_derived_neuron_columns",
    "build_trait_matrix",
    "sanitize_trait_matrix",
    "traits_matrix_to_csv",
]
