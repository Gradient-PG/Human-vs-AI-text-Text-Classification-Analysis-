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
from .dim_reduction import PCAReduction, UMAPReduction, get_dim_reduction
from .run_analysis import analyze_raid_model

analyze_model = analyze_raid_model  # backward-compatible name

__all__ = [
    "CLUSTERING_STRATEGY_IDS",
    "ClusteringResult",
    "ClusteringStrategy",
    "HDBSCANStrategy",
    "KMeansSilhouetteStrategy",
    "PCAReduction",
    "UMAPReduction",
    "WardGapStrategy",
    "WardSilhouetteStrategy",
    "analyze_model",
    "analyze_raid_model",
    "clustering_strategies_from_names",
    "default_clustering_strategies",
    "get_dim_reduction",
    "run_clustering_strategies",
]
