"""RAID multi-model neuron analysis: data loading, clustering, visualization, reports."""

from .constants import ALL_LAYERS, ALPHA, AUC_HIGH, AUC_LOW

from .data import (
    build_full_neuron_matrix,
    load_activation_column,
    load_activations,
    load_activations_for_model,
    load_stats,
    compute_neuron_statistics,
    identify_discriminative_neurons,
    add_derived_neuron_columns,
    build_trait_matrix,
    sanitize_trait_matrix,
    traits_matrix_to_csv,
)

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
    run_clustering_analysis,
)

from .viz import (
    PCAReduction,
    UMAPReduction,
    get_dim_reduction,
)

from .io import save_figure, write_text

from .reports import (
    neuron_summary_text,
    clustering_summary_text,
    exemplar_text,
    assign_auc_preference_groups,
)

from .run_analysis import analyze_raid_model

analyze_model = analyze_raid_model

__all__ = [
    # Constants
    "ALL_LAYERS",
    "ALPHA",
    "AUC_HIGH",
    "AUC_LOW",
    # Data loading
    "build_full_neuron_matrix",
    "load_activation_column",
    "load_activations",
    "load_activations_for_model",
    "load_stats",
    # Neuron statistics
    "compute_neuron_statistics",
    "identify_discriminative_neurons",
    # Traits
    "add_derived_neuron_columns",
    "build_trait_matrix",
    "sanitize_trait_matrix",
    "traits_matrix_to_csv",
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
    "run_clustering_analysis",
    "run_clustering_strategies",
    # Dimensionality reduction
    "PCAReduction",
    "UMAPReduction",
    "get_dim_reduction",
    # IO
    "save_figure",
    "write_text",
    # Reports
    "assign_auc_preference_groups",
    "clustering_summary_text",
    "exemplar_text",
    "neuron_summary_text",
    # Orchestration
    "analyze_model",
    "analyze_raid_model",
]
