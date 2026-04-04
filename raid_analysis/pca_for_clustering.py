"""PCA subspace for clustering algorithms and optional 2D projection for figures."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def format_pca_cluster_variance_report(pca: PCA, requested_components: int) -> str:
    """Human-readable per-component and cumulative explained variance."""
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    lines = [
        "PCA subspace used for clustering (not used for --cluster-viz umap; "
        "UMAP fits on raw activations)",
        f"Requested n_components: {requested_components}",
        f"Fitted n_components_: {pca.n_components_}",
        "",
        "  k    explained_var_ratio    cumulative",
    ]
    for k, (r, c) in enumerate(zip(evr, cum), start=1):
        lines.append(f"  {k:3d}  {r:20.8f}  {c:.8f}")
    lines += [
        "",
        f"Sum of explained variance ratio: {float(np.sum(evr)):.8f}",
    ]
    return "\n".join(lines)


def fit_pca_cluster_subspace(
    X_all: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """
    Reduce neuron activation rows to a PCA subspace for clustering / HDBSCAN.

    X_all: (n_neurons, n_samples) — same layout as build_full_neuron_matrix.
    """
    n_max = min(n_components, X_all.shape[0], X_all.shape[1])
    if n_max < 2:
        raise ValueError(
            f"Need at least 2 PCA components for clustering; got n_max={n_max} "
            f"(shape={X_all.shape})."
        )
    pca = PCA(n_components=n_max, random_state=random_state)
    X_red = pca.fit_transform(X_all)
    return X_red, pca


def embedding_2d_for_visualization(
    X_cluster: np.ndarray,
    X_raw: np.ndarray,
    viz: str,
    random_state: int = 42,
) -> np.ndarray:
    """
    2D coordinates for clustering figures only (algorithms use X_cluster).

    - pca: first two columns of the clustering PCA.
    - umap: UMAP on the raw per-neuron activation matrix, metric ``correlation``.
    """
    if X_raw.shape[0] != X_cluster.shape[0]:
        raise ValueError(
            f"X_raw and X_cluster must have the same number of rows (neurons); "
            f"got {X_raw.shape[0]} vs {X_cluster.shape[0]}"
        )
    if viz == "pca":
        if X_cluster.shape[1] < 2:
            raise ValueError("Cluster PCA must retain at least 2 components for PC viz.")
        return np.asarray(X_cluster[:, :2], dtype=np.float64, order="C")
    if viz == "umap":
        try:
            import umap as umap_lib
        except ImportError as exc:
            raise ImportError(
                "umap-learn is required for --cluster-viz umap. Install: uv add umap-learn"
            ) from exc
        reducer = umap_lib.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="correlation",
            random_state=random_state,
        )
        return reducer.fit_transform(X_raw)
    raise ValueError(f"Unknown cluster viz mode: {viz!r}; use 'pca' or 'umap'")


def viz_axis_labels(viz: str) -> tuple[str, str]:
    if viz == "umap":
        return "UMAP 1", "UMAP 2"
    if viz == "pca":
        return "PC 1", "PC 2"
    raise ValueError(f"Unknown viz {viz!r}; use 'pca' or 'umap'")
