"""Per-model RAID analysis: neuron figures, 2D embedding plots, full-space clustering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

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


@dataclass(frozen=True)
class ClusteringScatterSpec:
    """One 2D neuron embedding plot colored by cluster labels."""

    xy: np.ndarray
    axis_x: str
    axis_y: str
    file_stem: str
    method_title: str


def _save_clustering_outputs(
    *,
    neurons_df: DataFrame,
    model: str,
    result: ClusteringResult,
    figures_path: Path,
    text_path: Path,
    hierarchy_figure_prefix: str,
    scatter_specs: Sequence[ClusteringScatterSpec],
    summary_intro: Sequence[str],
    summary_method_name: str | None = None,
) -> None:
    """Hierarchy diagnostics (per strategy) plus one or more cluster overlay scatters."""
    prefix = hierarchy_figure_prefix
    for spec in scatter_specs:
        save_figure(
            fig_full_space_clustering(
                spec.xy, neurons_df, result.labels, model,
                spec.method_title, spec.axis_x, spec.axis_y,
            ),
            figures_path / f"{spec.file_stem}_clusters.png",
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
    summary_header = list(summary_intro) + list(header)
    method_name = summary_method_name if summary_method_name is not None else result.method_display_name
    write_text(
        text_path,
        clustering_summary_text(
            neurons_df, result.labels, model,
            method=method_name,
            extra_header_lines=summary_header,
        ),
    )


@dataclass
class ClusteringSpaceConfig:
    """
    One feature matrix to cluster (activations or traits) plus output naming and plots.

    Strategies are the same :class:`~raid_analysis.clustering.ClusteringStrategy` list;
    only the standardized input matrix and visualization targets differ.
    """

    X: np.ndarray
    ward_log: str | None
    log_prefix: str
    summary_intro: list[str]
    hierarchy_prefix: Callable[[ClusteringResult], str]
    scatter_specs: Callable[[ClusteringResult], list[ClusteringScatterSpec]]
    text_path: Callable[[ClusteringResult], Path]
    method_name_for_summary: Callable[[ClusteringResult], str] | None = None


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

    print("\n  [2/3] Clustering (full activation space) + 2D embedding (visualization)")
    X_all = build_full_neuron_matrix(neurons_df, results_path)
    X_cluster = StandardScaler().fit_transform(X_all)
    print(
        f"    Clustering input: {X_cluster.shape[0]:,} neurons × {X_cluster.shape[1]:,} dims "
        "(column-standardized activations)"
    )
    print(f"    {spec.name} 2D embedding for plots only ...")
    embedding = dim_reduction.fit_transform(X_all)

    save_figure(
        fig_embedding_all_neurons(
            embedding, neurons_df, model,
            axis_x=spec.axis_x, axis_y=spec.axis_y, embed_name=spec.name,
        ),
        figures_path / f"neuron_embedding_{spec.slug}.png",
    )

    activation_intro = [
        "Clustering input: column-standardized full activation matrix (one feature per sample).",
        "Plots: cluster labels on the chosen 2D embedding are for visualization only.",
        "",
    ]

    trait_bundle: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if traits_clustering:
        print("\n  Trait space: neuron×trait matrix → traits_matrix.csv")
        T_raw, trait_names = build_trait_matrix(neurons_df, X_all, labels)
        T_raw = sanitize_trait_matrix(T_raw)
        traits_matrix_to_csv(output_path / "traits_matrix.csv", T_raw, trait_names)
        print(f"    {T_raw.shape[1]} columns saved")
        T_scaled = StandardScaler().fit_transform(T_raw)
        print("    2D activation embeddings for trait-cluster overlays ...")
        if spec.slug == "pca":
            embed_act_pca = embedding
            embed_act_umap = get_dim_reduction("umap").fit_transform(X_all)
        else:
            embed_act_pca = PCAReduction().fit_transform(X_all)
            embed_act_umap = embedding
        trait_bundle = (T_scaled, embed_act_pca, embed_act_umap)

    clustering_spaces: list[ClusteringSpaceConfig] = [
        ClusteringSpaceConfig(
            X=X_cluster,
            ward_log="    Ward linkage on full space (shared for hierarchical strategies) ...",
            log_prefix="    ",
            summary_intro=activation_intro,
            hierarchy_prefix=lambda r: f"{spec.slug}_{r.file_tag}",
            scatter_specs=lambda r: [
                ClusteringScatterSpec(
                    xy=embedding,
                    axis_x=spec.axis_x,
                    axis_y=spec.axis_y,
                    file_stem=f"{spec.slug}_{r.file_tag}",
                    method_title=r.method_display_name,
                )
            ],
            text_path=lambda r: output_path / f"clustering_{spec.slug}_{r.file_tag}.txt",
            method_name_for_summary=None,
        ),
    ]
    if trait_bundle is not None:
        T_scaled, embed_act_pca, embed_act_umap = trait_bundle
        trait_intro = [
            "Clustering input: standardized trait matrix (see traits_matrix.csv).",
            "Plots: trait-derived labels projected onto 2D activation PCA / UMAP.",
            "",
        ]
        clustering_spaces.append(
            ClusteringSpaceConfig(
                X=T_scaled,
                ward_log="    Ward linkage on standardized traits ...",
                log_prefix="    [traits] ",
                summary_intro=trait_intro,
                hierarchy_prefix=lambda r: f"traits_space_{r.file_tag}",
                scatter_specs=lambda r: [
                    ClusteringScatterSpec(
                        embed_act_pca,
                        "PC 1",
                        "PC 2",
                        f"traits_on_act_pca_{r.file_tag}",
                        f"{r.method_display_name} (trait clusters → act PCA)",
                    ),
                    ClusteringScatterSpec(
                        embed_act_umap,
                        "UMAP 1",
                        "UMAP 2",
                        f"traits_on_act_umap_{r.file_tag}",
                        f"{r.method_display_name} (trait clusters → act UMAP)",
                    ),
                ],
                text_path=lambda r: output_path / f"clustering_traits_{r.file_tag}.txt",
                method_name_for_summary=lambda r: f"{r.method_display_name} [trait space]",
            ),
        )

    cluster_for_exemplars: np.ndarray | None = None
    last_result: ClusteringResult | None = None

    needs_ward = any(isinstance(s, (WardSilhouetteStrategy, WardGapStrategy)) for s in strategies)

    for i, space_cfg in enumerate(clustering_spaces):
        if needs_ward and space_cfg.ward_log:
            print(space_cfg.ward_log)
        cluster_results = run_clustering_strategies(space_cfg.X, strategies)
        for strategy, result in zip(strategies, cluster_results):
            print(f"{space_cfg.log_prefix}{strategy.__class__.__name__} ...")
            if i == 0:
                last_result = result
                if isinstance(strategy, WardSilhouetteStrategy):
                    cluster_for_exemplars = result.labels
            n_clust = len({c for c in result.labels if c >= 0})
            print(f"    → {result.method_display_name}: K={n_clust}")

            sm = (
                space_cfg.method_name_for_summary(result)
                if space_cfg.method_name_for_summary
                else None
            )
            _save_clustering_outputs(
                neurons_df=neurons_df,
                model=model,
                result=result,
                figures_path=figures_path,
                text_path=space_cfg.text_path(result),
                hierarchy_figure_prefix=space_cfg.hierarchy_prefix(result),
                scatter_specs=space_cfg.scatter_specs(result),
                summary_intro=space_cfg.summary_intro,
                summary_method_name=sm,
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
