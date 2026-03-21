"""Dimensionality reduction strategies for 2D neuron embedding visualization (PCA or UMAP)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DimReductionSpec:
    """Human-readable labels for plots and file naming."""

    name: str
    slug: str
    axis_x: str
    axis_y: str


class DimReductionStrategy(ABC):
    """Reduce neuron feature rows to 2D for visualization (clustering uses full ``X_neurons`` in the pipeline)."""

    @property
    @abstractmethod
    def spec(self) -> DimReductionSpec:
        ...

    @abstractmethod
    def fit_transform(self, X_neurons: np.ndarray) -> np.ndarray:
        """
        Args:
            X_neurons: (n_neurons, n_samples) — one row per neuron.

        Returns:
            embedding: (n_neurons, 2), standardized per axis for readable plots.
        """


class PCAReduction(DimReductionStrategy):
    """PCA to 2 components, then per-axis standardization for plotting."""

    def __init__(self, random_state: int = 42):
        self._random_state = random_state

    @property
    def spec(self) -> DimReductionSpec:
        return DimReductionSpec(
            name="PCA",
            slug="pca",
            axis_x="PC 1",
            axis_y="PC 2",
        )

    def fit_transform(self, X_neurons: np.ndarray) -> np.ndarray:
        raw = PCA(n_components=2, random_state=self._random_state).fit_transform(X_neurons)
        return StandardScaler().fit_transform(raw)


class UMAPReduction(DimReductionStrategy):
    """UMAP with correlation metric on neuron rows; output standardized for plotting."""

    def __init__(
        self,
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "correlation",
    ):
        self._random_state = random_state
        self._n_neighbors = n_neighbors
        self._min_dist = min_dist
        self._metric = metric

    @property
    def spec(self) -> DimReductionSpec:
        return DimReductionSpec(
            name="UMAP",
            slug="umap",
            axis_x="UMAP 1",
            axis_y="UMAP 2",
        )

    def fit_transform(self, X_neurons: np.ndarray) -> np.ndarray:
        from umap import UMAP

        n = X_neurons.shape[0]
        n_neighbors = min(self._n_neighbors, max(2, n - 1))
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=self._min_dist,
            metric=self._metric,
            random_state=self._random_state,
            verbose=False,
        )
        raw = reducer.fit_transform(X_neurons)
        return StandardScaler().fit_transform(raw)


def get_dim_reduction(kind: str, **kwargs) -> DimReductionStrategy:
    kind_l = kind.lower().strip()
    if kind_l == "pca":
        return PCAReduction(**kwargs)
    if kind_l == "umap":
        return UMAPReduction(**kwargs)
    raise ValueError(f"Unknown dim reduction: {kind!r} (use 'pca' or 'umap')")
