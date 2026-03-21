"""Per-model RAID analysis: neuron figures + embedding + clustering strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from .clustering import (
    ClusteringResult,
    ClusteringStrategy,
    KMeansSilhouetteStrategy,
    WardGapStrategy,
    WardSilhouetteStrategy,
    default_clustering_strategies,
    run_clustering_strategies,
)
from .data import build_full_neuron_matrix, load_stats
from .dim_reduction import DimReductionStrategy, PCAReduction, get_dim_reduction
from .traits import build_trait_matrix, sanitize_trait_matrix, traits_matrix_to_csv
from .figures_embedding import fig_embedding_all_neurons, fig_full_space_clustering
from .figures_hierarchy import fig_dendrogram, fig_merge_distance_gaps, fig_silhouette
from .figures_neurons import (
    fig_activation_boxplots,
    fig_activation_scatter,
    fig_layer_distribution,
)
from .io import save_figure, write_text
from .summaries import clustering_summary_text, exemplar_text, neuron_summary_text


def _save_clustering_outputs(
    *,
    embedding: np.ndarray,
    neurons_df: DataFrame,
    model: str,
    axis_x: str,
    axis_y: str,
    dim_slug: str,
    result: ClusteringResult,
    figures_path: Path,
    output_path: Path,
) -> None:
    prefix = f"{dim_slug}_{result.file_tag}"
    save_figure(
        fig_full_space_clustering(
            embedding, neurons_df, result.labels, model,
            result.method_display_name, axis_x, axis_y,
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

    header = extra.get("summary_header_lines")
    write_text(
        output_path / f"clustering_{prefix}.txt",
        clustering_summary_text(
            neurons_df, result.labels, model,
            method=result.method_display_name,
            extra_header_lines=header,
        ),
    )


def _save_traits_clustering_outputs(
    *,
    neurons_df: DataFrame,
    model: str,
    result: ClusteringResult,
    figures_path: Path,
    output_path: Path,
    act_pca_2d: np.ndarray,
    act_umap_2d: np.ndarray,
) -> None:
    """Trait-space hierarchy artifacts + cluster overlays on activation PCA & UMAP."""
    prefix_ts = f"traits_space_{result.file_tag}"
    extra = result.extra

    if result.file_tag == "ward_silhouette":
        k_range = extra["k_range"]
        sil = extra["silhouette"]
        opt_k = extra["opt_k"]
        Z = extra["Z"]
        save_figure(fig_silhouette(k_range, sil, opt_k, model),
                    figures_path / f"{prefix_ts}_scores.png")
        save_figure(fig_dendrogram(Z, opt_k, model),
                    figures_path / f"{prefix_ts}_dendrogram.png")
    elif result.file_tag == "ward_gap":
        Z = extra["Z"]
        opt_k = extra["opt_k"]
        save_figure(fig_merge_distance_gaps(Z, model),
                    figures_path / f"{prefix_ts}_distances.png")
        save_figure(fig_dendrogram(Z, opt_k, model),
                    figures_path / f"{prefix_ts}_dendrogram.png")
    elif result.file_tag == "kmeans":
        k_range = extra["k_range"]
        sil = extra["km_silhouette"]
        opt_k = extra["opt_k"]
        save_figure(fig_silhouette(k_range, sil, opt_k, model),
                    figures_path / f"{prefix_ts}_silhouette_scores.png")

    method_pca = f"{result.method_display_name} (trait clusters → act PCA)"
    method_umap = f"{result.method_display_name} (trait clusters → act UMAP)"
    save_figure(
        fig_full_space_clustering(
            act_pca_2d, neurons_df, result.labels, model,
            method_pca, "PC 1", "PC 2",
        ),
        figures_path / f"traits_on_act_pca_{result.file_tag}_clusters.png",
    )
    save_figure(
        fig_full_space_clustering(
            act_umap_2d, neurons_df, result.labels, model,
            method_umap, "UMAP 1", "UMAP 2",
        ),
        figures_path / f"traits_on_act_umap_{result.file_tag}_clusters.png",
    )

    header = extra.get("summary_header_lines") or []
    trait_header = [
        "Clustering input: standardized trait matrix (see traits_matrix.csv).",
        "Plots: trait-derived labels projected onto 2D activation PCA / UMAP.",
        "",
    ] + list(header)
    write_text(
        output_path / f"clustering_traits_{result.file_tag}.txt",
        clustering_summary_text(
            neurons_df, result.labels, model,
            method=f"{result.method_display_name} [trait space]",
            extra_header_lines=trait_header,
        ),
    )


def analyze_model(
    model: str,
    activations_root: Path,
    tokenized_root: Path,
    output_root: Path,
    run_exemplars: bool,
    split: str,
    dim_reduction: DimReductionStrategy | None = None,
    clustering_strategies: Sequence[ClusteringStrategy] | None = None,
    *,
    output_subdir: str | None = None,
    skip_neuron_analysis: bool = False,
    traits_clustering: bool = False,
) -> bool:
    from utils.raid_loader import slug

    model_slug = slug(model)
    results_path = activations_root / f"activations_raid_{model_slug}"
    tokenized_path = tokenized_root / f"raid_{model_slug}"
    base = output_root / model_slug
    output_path = base / output_subdir if output_subdir else base
    figures_path = output_path / "figures"

    dim_reduction = dim_reduction or get_dim_reduction("pca")
    strategies = list(clustering_strategies) if clustering_strategies else default_clustering_strategies()

    spec = dim_reduction.spec

    print(f"\n{'='*60}")
    print(f"  ANALYSIS: {model} vs human")
    print(f"  Activations: {results_path}")
    print(f"  Output:      {output_path}")
    if output_subdir:
        print(f"  (nested under {base} by dim reduction)")
    print(f"  Embedding:   {spec.name}")
    print(f"{'='*60}")

    if not results_path.exists():
        print(f"  SKIP — activations not found: {results_path}")
        return False

    from .constants import ALL_LAYERS

    missing_csvs = [
        results_path / f"layer_{l}_neuron_stats.csv"
        for l in ALL_LAYERS
        if not (results_path / f"layer_{l}_neuron_stats.csv").exists()
    ]
    if missing_csvs:
        print(f"  SKIP — {len(missing_csvs)} neuron stats CSVs missing. "
              f"Run analyze_activations.py first.")
        return False

    output_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    neurons_df, disc_df, labels = load_stats(results_path)

    if not skip_neuron_analysis:
        print("\n  [1/3] Neuron analysis")
        print(f"    Neurons total: {len(neurons_df):,}  |  discriminative: {len(disc_df):,}")
        save_figure(fig_layer_distribution(neurons_df, disc_df, model),
                    figures_path / "figure1_layer_distribution.png")
        save_figure(fig_activation_boxplots(disc_df, labels, results_path, model),
                    figures_path / "figure2_activation_boxplots.png")
        save_figure(fig_activation_scatter(disc_df, model),
                    figures_path / "figure3_activation_scatter.png")
        write_text(output_path / "neurons_summary.txt",
                   neuron_summary_text(neurons_df, disc_df, results_path, model))
    else:
        print("\n  [1/3] Neuron analysis — skipped (already written for this model)")

    print("\n  [2/3] Clustering (full neuron space)")
    X_all = build_full_neuron_matrix(neurons_df, results_path)
    print(f"    Computing {spec.name} embedding ...")
    embedding = dim_reduction.fit_transform(X_all)

    save_figure(
        fig_embedding_all_neurons(
            embedding, neurons_df, model,
            axis_x=spec.axis_x, axis_y=spec.axis_y, embed_name=spec.name,
        ),
        figures_path / f"neuron_embedding_{spec.slug}.png",
    )

    if any(isinstance(s, (WardSilhouetteStrategy, WardGapStrategy)) for s in strategies):
        print("    Ward linkage (shared for hierarchical strategies) ...")

    cluster_for_exemplars: np.ndarray | None = None
    last_result: ClusteringResult | None = None

    emb_results = run_clustering_strategies(embedding, strategies)
    for strategy, result in zip(strategies, emb_results):
        print(f"    {strategy.__class__.__name__} ...")
        last_result = result
        if isinstance(strategy, WardSilhouetteStrategy):
            cluster_for_exemplars = result.labels
        n_clust = len({c for c in result.labels if c >= 0})
        print(f"    → {result.method_display_name}: K={n_clust}")

        _save_clustering_outputs(
            embedding=embedding,
            neurons_df=neurons_df,
            model=model,
            axis_x=spec.axis_x,
            axis_y=spec.axis_y,
            dim_slug=spec.slug,
            result=result,
            figures_path=figures_path,
            output_path=output_path,
        )

    if traits_clustering:
        print("\n  [2b] Trait-based clustering (overlays on activation PCA & UMAP)")
        T_raw, trait_names = build_trait_matrix(neurons_df, X_all, labels)
        T_raw = sanitize_trait_matrix(T_raw)
        traits_matrix_to_csv(output_path / "traits_matrix.csv", T_raw, trait_names)
        print(f"    Trait matrix: {T_raw.shape[1]} columns → traits_matrix.csv")
        T_scaled = StandardScaler().fit_transform(T_raw)

        print("    2D activation embeddings for trait-cluster visualization ...")
        if spec.slug == "pca":
            embed_act_pca = embedding
            embed_act_umap = get_dim_reduction("umap").fit_transform(X_all)
        else:
            embed_act_pca = PCAReduction().fit_transform(X_all)
            embed_act_umap = embedding

        if any(isinstance(s, (WardSilhouetteStrategy, WardGapStrategy)) for s in strategies):
            print("    Ward linkage on standardized traits ...")
        trait_results = run_clustering_strategies(T_scaled, strategies)
        for strategy, result in zip(strategies, trait_results):
            print(f"    [traits] {strategy.__class__.__name__} ...")
            n_clust = len({c for c in result.labels if c >= 0})
            print(f"    → {result.method_display_name}: K={n_clust}")

            _save_traits_clustering_outputs(
                neurons_df=neurons_df,
                model=model,
                result=result,
                figures_path=figures_path,
                output_path=output_path,
                act_pca_2d=embed_act_pca,
                act_umap_2d=embed_act_umap,
            )

    neurons_df = neurons_df.copy()
    if cluster_for_exemplars is not None:
        neurons_df["cluster"] = cluster_for_exemplars
    elif last_result is not None:
        neurons_df["cluster"] = last_result.labels

    print("\n  [3/3] Exemplars")
    if run_exemplars:
        if not tokenized_path.exists():
            print(f"    SKIP — tokenized dataset not found: {tokenized_path}")
        else:
            try:
                text = exemplar_text(
                    X_all, neurons_df, labels, results_path,
                    tokenized_path, split, model,
                )
                write_text(output_path / "exemplars.txt", text)
            except Exception as exc:
                print(f"    ERROR: {exc}")
    else:
        print("    Skipped (pass --exemplars to enable)")

    return True
