"""Orchestrate neurons analysis, optional clustering, and optional exemplars."""

from __future__ import annotations

from pathlib import Path

from utils.raid_loader import slug

from .constants import ALL_LAYERS
from .clustering.pipeline import run_clustering_analysis
from .reports.exemplars import run_exemplars_analysis
from .neurons_pipeline import run_neurons_analysis


def analyze_raid_model(
    model: str,
    activations_root: Path,
    tokenized_root: Path,
    output_root: Path,
    *,
    run_exemplars: bool,
    split: str,
    cluster_pca_components: int,
    neuron_viz: str,
    cluster_viz: str,
    clustering: tuple[str, ...] | None,
) -> bool:
    """
    Write ``output_root/{slug}/neurons/`` and optionally ``.../clustering/``.

    ``clustering`` is ``None`` or empty to skip clustering; otherwise method names
    from :data:`~raid_analysis.clustering.strategies.CLUSTERING_STRATEGY_IDS`.
    """
    model_slug = slug(model)
    results_path = activations_root / f"activations_raid_{model_slug}"
    tokenized_path = tokenized_root / f"raid_{model_slug}"
    base_out = output_root / model_slug
    neurons_path = base_out / "neurons"
    clustering_path = base_out / "clustering"

    print(f"\n{'='*60}")
    print(f"  ANALYSIS: {model} vs human")
    print(f"  Activations: {results_path}")
    print(f"  Output:      {base_out}")
    print(f"    neurons:    {neurons_path}")
    print(f"    clustering: {clustering_path}")
    print(f"{'='*60}")

    if not results_path.exists():
        print(f"  SKIP - activations not found: {results_path}")
        return False

    missing_csvs = [
        results_path / f"layer_{l}_neuron_stats.csv"
        for l in ALL_LAYERS
        if not (results_path / f"layer_{l}_neuron_stats.csv").exists()
    ]
    if missing_csvs:
        print(f"  SKIP - {len(missing_csvs)} neuron stats CSVs missing. "
              f"Run analyze_activations.py first.")
        return False

    print("\n  [1/3] Neuron analysis")
    neurons_df, _disc_df, labels, X_all = run_neurons_analysis(
        model, results_path, neurons_path, neuron_viz=neuron_viz
    )

    methods = tuple(clustering) if clustering else ()
    if methods:
        print(
            "\n  [2/3] Clustering "
            f"(methods: {', '.join(methods)}; PCA subspace; 2D viz separate)"
        )
        ok_c = run_clustering_analysis(
            model=model,
            neurons_df=neurons_df,
            results_path=results_path,
            clustering_out=clustering_path,
            X_all=X_all,
            cluster_pca_components=cluster_pca_components,
            cluster_viz=cluster_viz,
            methods=methods,
        )
        if not ok_c:
            return False
    else:
        print("\n  [2/3] Clustering - skipped (pass --clustering METHOD [...] to enable)")

    if run_exemplars:
        print("\n  [3/3] Exemplars (AUC preference groups)")
        ok_e = run_exemplars_analysis(
            model=model,
            neurons_df=neurons_df,
            X_all=X_all,
            labels=labels,
            results_path=results_path,
            tokenized_path=tokenized_path,
            split=split,
            neurons_out=neurons_path,
        )
        if not ok_e:
            return False
    else:
        print("\n  [3/3] Exemplars - skipped (pass --exemplars to enable)")

    return True
