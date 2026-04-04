"""Clustering strategies on a feature matrix (e.g. full neuron activations or traits)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringResult:
    labels: np.ndarray
    method_display_name: str
    file_tag: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WardLinkageContext:
    """Shared Ward linkage matrix for multiple hierarchical cuts."""

    Z: np.ndarray

    @staticmethod
    def from_embedding(embedding: np.ndarray) -> "WardLinkageContext":
        return WardLinkageContext(Z=linkage(embedding, method="ward"))


class ClusteringStrategy(ABC):
    @abstractmethod
    def run(
        self,
        embedding: np.ndarray,
        ward: WardLinkageContext | None = None,
    ) -> ClusteringResult:
        ...


class WardSilhouetteStrategy(ClusteringStrategy):
    def __init__(self, k_range: Iterable[int] | None = None):
        self.k_range = list(k_range) if k_range is not None else list(range(2, 9))

    def run(
        self,
        embedding: np.ndarray,
        ward: WardLinkageContext | None = None,
    ) -> ClusteringResult:
        if ward is None:
            ward = WardLinkageContext.from_embedding(embedding)
        Z = ward.Z
        sil_all = {
            k: silhouette_score(
                embedding,
                fcluster(Z, t=k, criterion="maxclust"),
                metric="euclidean",
            )
            for k in self.k_range
        }
        opt_k = max(sil_all, key=sil_all.get)
        labels = fcluster(Z, t=opt_k, criterion="maxclust")
        header = [
            f"K (Silhouette):               {opt_k}  (score: {sil_all[opt_k]:.4f})",
            "",
            "Silhouette scores per K:",
        ] + [
            f"  K={k}: {sil_all[k]:.4f}{' <-- optimal' if k == opt_k else ''}"
            for k in self.k_range
        ]
        return ClusteringResult(
            labels=labels,
            method_display_name="Ward (Silhouette)",
            file_tag="ward_silhouette",
            extra={
                "Z": Z,
                "silhouette": sil_all,
                "opt_k": opt_k,
                "k_range": self.k_range,
                "summary_header_lines": header,
            },
        )


class WardGapStrategy(ClusteringStrategy):
    def __init__(self, n_last: int = 15):
        self.n_last = n_last

    def _merge_gap_optimal_k(self, Z: np.ndarray) -> int:
        n_merges = min(self.n_last, len(Z))
        distances = Z[-n_merges:, 2]
        gaps = np.diff(distances)
        k_values = np.arange(n_merges, 1, -1)
        return int(k_values[int(np.argmax(gaps))])

    def run(
        self,
        embedding: np.ndarray,
        ward: WardLinkageContext | None = None,
    ) -> ClusteringResult:
        if ward is None:
            ward = WardLinkageContext.from_embedding(embedding)
        Z = ward.Z
        opt_k = self._merge_gap_optimal_k(Z)
        labels = fcluster(Z, t=opt_k, criterion="maxclust")
        return ClusteringResult(
            labels=labels,
            method_display_name="Ward (Merge Gap)",
            file_tag="ward_gap",
            extra={
                "Z": Z,
                "opt_k": opt_k,
                "summary_header_lines": [f"K (largest merge gap):        {opt_k}"],
            },
        )


class KMeansSilhouetteStrategy(ClusteringStrategy):
    def __init__(self, k_range: Iterable[int] | None = None, n_init: int = 10, random_state: int = 42):
        self.k_range = list(k_range) if k_range is not None else list(range(2, 9))
        self.n_init = n_init
        self.random_state = random_state

    def run(
        self,
        embedding: np.ndarray,
        ward: WardLinkageContext | None = None,
    ) -> ClusteringResult:
        km_sil = {}
        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=self.n_init)
            lbl = km.fit_predict(embedding)
            km_sil[k] = silhouette_score(embedding, lbl, metric="euclidean")
        opt_k = max(km_sil, key=km_sil.get)
        km_final = KMeans(
            n_clusters=opt_k, random_state=self.random_state, n_init=self.n_init
        )
        labels = km_final.fit_predict(embedding)
        header = [
            f"K (KMeans Silhouette):        {opt_k}  (score: {km_sil[opt_k]:.4f})",
            "",
            "Silhouette scores per K:",
        ] + [
            f"  K={k}: {km_sil[k]:.4f}{' <-- optimal' if k == opt_k else ''}"
            for k in self.k_range
        ]
        return ClusteringResult(
            labels=labels,
            method_display_name="KMeans",
            file_tag="kmeans",
            extra={
                "km_silhouette": km_sil,
                "opt_k": opt_k,
                "k_range": self.k_range,
                "summary_header_lines": header,
            },
        )


class HDBSCANStrategy(ClusteringStrategy):
    """
    HDBSCAN (hierarchical density-based clustering).

    Labels use ``-1`` for noise; dense clusters are ``0 .. K-1``.
    """

    def __init__(
        self,
        min_cluster_size: int | None = None,
        min_samples: int | None = None,
        cluster_selection_epsilon: float = 0.0,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def run(
        self,
        embedding: np.ndarray,
        ward: WardLinkageContext | None = None,
    ) -> ClusteringResult:
        n = embedding.shape[0]
        mcs = self.min_cluster_size
        if mcs is None:
            mcs = max(15, min(150, n // 60))
        ms = self.min_samples
        if ms is None:
            ms = max(5, mcs // 5)
        try:
            import hdbscan as hdbscan_lib
        except ImportError as exc:
            raise ImportError(
                "hdbscan is required for HDBSCANStrategy. Install: uv pip install hdbscan"
            ) from exc
        clusterer = hdbscan_lib.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_method="eom",
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        labels = clusterer.fit_predict(embedding).astype(np.int64)
        unique = set(labels.tolist())
        n_clusters = len(unique) - (1 if -1 in unique else 0)
        n_noise = int((labels == -1).sum())
        header = [
            f"min_cluster_size:             {mcs}",
            f"min_samples:                  {ms}",
            f"Clusters (excl. noise):       {n_clusters}",
            f"Noise points:                 {n_noise}",
            "",
        ]
        return ClusteringResult(
            labels=labels,
            method_display_name="HDBSCAN",
            file_tag="hdbscan",
            extra={
                "min_cluster_size": mcs,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "summary_header_lines": header,
            },
        )


def default_clustering_strategies() -> list[ClusteringStrategy]:
    return [
        WardSilhouetteStrategy(),
        WardGapStrategy(),
        KMeansSilhouetteStrategy(),
        HDBSCANStrategy(),
    ]


CLUSTERING_STRATEGY_IDS = ("ward_silhouette", "ward_gap", "kmeans", "hdbscan")


def clustering_strategies_from_names(names: Sequence[str]) -> list[ClusteringStrategy]:
    """
    Instantiate strategies from stable ids (CLI / config).

    ``names`` entries must be among :data:`CLUSTERING_STRATEGY_IDS`.
    """
    factory = {
        "ward_silhouette": WardSilhouetteStrategy,
        "ward_gap": WardGapStrategy,
        "kmeans": KMeansSilhouetteStrategy,
        "hdbscan": HDBSCANStrategy,
    }
    out: list[ClusteringStrategy] = []
    for raw in names:
        key = raw.strip().lower()
        if key not in factory:
            raise ValueError(
                f"Unknown clustering strategy {raw!r}. "
                f"Choose from: {', '.join(CLUSTERING_STRATEGY_IDS)}"
            )
        out.append(factory[key]())
    return out


def run_clustering_strategies(
    embedding: np.ndarray,
    strategies: Sequence[ClusteringStrategy] | None = None,
) -> list[ClusteringResult]:
    """
    Run each strategy on the same point cloud.

    Ward-based strategies share one :class:`WardLinkageContext` so linkage is
    computed once per embedding.
    """
    strategies = list(strategies or default_clustering_strategies())
    needs_ward = any(isinstance(s, (WardSilhouetteStrategy, WardGapStrategy)) for s in strategies)
    ward_ctx: WardLinkageContext | None = None
    if needs_ward:
        ward_ctx = WardLinkageContext.from_embedding(embedding)
    results: list[ClusteringResult] = []
    for strategy in strategies:
        w = ward_ctx if isinstance(strategy, (WardSilhouetteStrategy, WardGapStrategy)) else None
        results.append(strategy.run(embedding, ward=w))
    return results
