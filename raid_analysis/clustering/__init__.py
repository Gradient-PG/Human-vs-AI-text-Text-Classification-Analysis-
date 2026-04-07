"""Self-contained clustering module (optional feature)."""

from .strategies import (
    CLUSTERING_STRATEGY_IDS,
    ClusteringResult,
    ClusteringStrategy,
    HDBSCANStrategy,
    KMeansSilhouetteStrategy,
    WardGapStrategy,
    WardLinkageContext,
    WardSilhouetteStrategy,
    clustering_strategies_from_names,
    default_clustering_strategies,
    run_clustering_strategies,
)
from .pipeline import run_clustering_analysis

__all__ = [
    "CLUSTERING_STRATEGY_IDS",
    "ClusteringResult",
    "ClusteringStrategy",
    "HDBSCANStrategy",
    "KMeansSilhouetteStrategy",
    "WardGapStrategy",
    "WardLinkageContext",
    "WardSilhouetteStrategy",
    "clustering_strategies_from_names",
    "default_clustering_strategies",
    "run_clustering_analysis",
    "run_clustering_strategies",
]
