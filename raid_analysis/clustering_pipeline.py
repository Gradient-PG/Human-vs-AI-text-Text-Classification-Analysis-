"""Full-space clustering on a PCA subspace of activations; writes to ``.../clustering/``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .clustering import (
    ClusteringResult,
    ClusteringStrategy,
    HDBSCANStrategy,
    KMeansSilhouetteStrategy,
    WardGapStrategy,
    WardSilhouetteStrategy,
    run_clustering_strategies,
)
from .figures_embedding import fig_full_space_clustering
from .figures_hierarchy import fig_dendrogram, fig_merge_distance_gaps, fig_silhouette
from .io import save_figure, write_text
from .pca_for_clustering import (
    embedding_2d_for_visualization,
    fit_pca_cluster_subspace,
    format_pca_cluster_variance_report,
    viz_axis_labels,
)
from .summaries import clustering_summary_text


def build_strategies_for_cli(methods: tuple[str, ...]) -> list[ClusteringStrategy]:
    """CLI-compatible strategy instances (matches former ``analyze_raid_models`` defaults)."""
    out: list[ClusteringStrategy] = []
    for raw in methods:
        key = raw.strip().lower()
        if key == "ward_silhouette":
            out.append(WardSilhouetteStrategy())
        elif key == "ward_gap":
            out.append(WardGapStrategy())
        elif key == "kmeans":
            out.append(KMeansSilhouetteStrategy())
        elif key == "hdbscan":
            try:
                import hdbscan  # noqa: F401
            except ImportError:
                print("    SKIP HDBSCAN - install with: uv pip install hdbscan")
                continue
            out.append(HDBSCANStrategy(min_cluster_size=300, min_samples=30))
        else:
            raise ValueError(f"Unknown clustering method: {raw!r}")
    return out


def _save_clustering_result(
    *,
    embedding_viz: np.ndarray,
    neurons_df: pd.DataFrame,
    model: str,
    result: ClusteringResult,
    figures_path: Path,
    text_path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    prefix = result.file_tag
    save_figure(
        fig_full_space_clustering(
            embedding_viz,
            neurons_df,
            result.labels,
            model,
            result.method_display_name,
            xlabel,
            ylabel,
        ),
        figures_path / f"{prefix}_clusters.png",
    )

    extra = result.extra
    if result.file_tag == "ward_silhouette":
        k_range = extra["k_range"]
        sil = extra["silhouette"]
        opt_k = extra["opt_k"]
        Z = extra["Z"]
        save_figure(fig_silhouette(k_range, sil, opt_k, model),
                    figures_path / f"{prefix}_scores.png")
        save_figure(fig_dendrogram(Z, opt_k, model),
                    figures_path / f"{prefix}_dendrogram.png")
    elif result.file_tag == "ward_gap":
        Z = extra["Z"]
        opt_k = extra["opt_k"]
        save_figure(fig_merge_distance_gaps(Z, model),
                    figures_path / f"{prefix}_distances.png")
        save_figure(fig_dendrogram(Z, opt_k, model),
                    figures_path / f"{prefix}_dendrogram.png")
    elif result.file_tag == "kmeans":
        k_range = extra["k_range"]
        sil = extra["km_silhouette"]
        opt_k = extra["opt_k"]
        save_figure(fig_silhouette(k_range, sil, opt_k, model),
                    figures_path / f"{prefix}_silhouette_scores.png")

    header = extra.get("summary_header_lines") or []
    write_text(
        text_path,
        clustering_summary_text(
            neurons_df,
            result.labels,
            model,
            method=result.method_display_name,
            extra_header_lines=list(header),
        ),
    )


def run_clustering_analysis(
    model: str,
    neurons_df: pd.DataFrame,
    results_path: Path,
    clustering_out: Path,
    X_all: np.ndarray,
    cluster_pca_components: int,
    cluster_viz: str,
    methods: tuple[str, ...],
) -> bool:
    """
    PCA subspace clustering + 2D projection for figures only.

    Returns False if 2D embedding fails (e.g. missing umap-learn); True otherwise.
    """
    strategies = build_strategies_for_cli(methods)
    if not strategies:
        print("    No clustering strategies to run (empty after resolving deps).")
        return True

    clustering_out.mkdir(parents=True, exist_ok=True)
    figures_path = clustering_out / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    print(f"    PCA (n_components={cluster_pca_components}) for clustering ...")
    X_cluster, pca_cluster = fit_pca_cluster_subspace(
        X_all, cluster_pca_components, random_state=42
    )
    var_report = format_pca_cluster_variance_report(pca_cluster, cluster_pca_components)
    write_text(clustering_out / "pca_cluster_variance.txt", var_report)
    evr = pca_cluster.explained_variance_ratio_
    head = min(5, len(evr))
    print(
        "    Explained variance (first components): "
        + ", ".join(f"PC{i + 1}={evr[i]:.4f}" for i in range(head))
    )
    print(
        f"    Total captured variance ({pca_cluster.n_components_} comps): "
        f"{float(np.sum(evr)):.4f}"
    )

    xlabel, ylabel = viz_axis_labels(cluster_viz)
    print(f"    2D embedding for clustering figures (--cluster-viz {cluster_viz}) ...")
    try:
        embedding_viz = embedding_2d_for_visualization(
            X_cluster, X_all, cluster_viz, random_state=42
        )
    except ImportError as exc:
        print(f"  ERROR: {exc}")
        return False

    results = run_clustering_strategies(X_cluster, strategies)
    for strategy, result in zip(strategies, results):
        print(f"    {strategy.__class__.__name__} → {result.method_display_name}")
        text_path = clustering_out / f"clustering_{result.file_tag}.txt"
        _save_clustering_result(
            embedding_viz=embedding_viz,
            neurons_df=neurons_df,
            model=model,
            result=result,
            figures_path=figures_path,
            text_path=text_path,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    return True
