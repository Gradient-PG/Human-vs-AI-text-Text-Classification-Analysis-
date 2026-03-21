#!/usr/bin/env python
"""
Run neuron analysis and clustering for each RAID model's activations.

Implementation lives in ``raid_analysis/`` (embedding + clustering strategies).

Prerequisites:
    uv run scripts/run_raid_pipeline.py --samples 1000

Usage:
    uv run scripts/analyze_raid_models.py
    uv run scripts/analyze_raid_models.py --models gpt4 chatgpt
    uv run scripts/analyze_raid_models.py --dim-reduction umap
    uv run scripts/analyze_raid_models.py --dim-reduction pca umap
    uv run scripts/analyze_raid_models.py --traits-clustering
    uv run scripts/analyze_raid_models.py --clustering ward_silhouette ward_gap
    uv run scripts/analyze_raid_models.py --models gpt4 --exemplars
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.append(str(Path(__file__).parent.parent))

from raid_analysis import clustering_strategies_from_names, get_dim_reduction
from raid_analysis.clustering import CLUSTERING_STRATEGY_IDS
from raid_analysis.pipeline import analyze_model
from utils.raid_loader import ALL_RAID_MODELS, slug


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Neuron + clustering analysis for each RAID model"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to analyze (default: all). Choices: {', '.join(ALL_RAID_MODELS)}",
    )
    parser.add_argument(
        "--activations-dir", default="results",
        help="Root dir containing activations_raid_* folders (default: results)",
    )
    parser.add_argument(
        "--tokenized-dir", default="data/processed",
        help="Root dir containing raid_* tokenized datasets (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir", default="results/analysis",
        help="Root dir for analysis outputs (default: results/analysis)",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split used during extraction (default: train)",
    )
    parser.add_argument(
        "--dim-reduction",
        nargs="+",
        choices=("pca", "umap"),
        default=["pca"],
        metavar="METHOD",
        help="One or more 2D embeddings for visualization only (PCA / UMAP). "
        "Neuron clustering uses the full activation matrix; multiple methods each get a "
        "subfolder under each model when more than one is given.",
    )
    parser.add_argument(
        "--exemplars", action="store_true",
        help="Also extract cluster exemplar texts (requires tokenized dataset; "
        "uses Ward+silhouette cluster ids from the first --dim-reduction output only "
        "to avoid duplicate exemplar files)",
    )
    parser.add_argument(
        "--traits-clustering",
        action="store_true",
        help="Also cluster neurons in standardized trait space (stats + simple activation summaries); "
        "save traits_matrix.csv and overlay trait clusters on activation PCA & UMAP figures",
    )
    parser.add_argument(
        "--clustering",
        nargs="+",
        default=None,
        metavar="METHOD",
        choices=CLUSTERING_STRATEGY_IDS,
        help="Which clustering strategies to run on full activations (and on traits if --traits-clustering). "
        "ward_silhouette: Ward + optimal K by silhouette; ward_gap: Ward + merge-gap K; "
        "kmeans: KMeans + silhouette K. Default: all three.",
    )
    args = parser.parse_args()

    models = args.models or ALL_RAID_MODELS
    invalid = [m for m in models if m not in ALL_RAID_MODELS]
    if invalid:
        print(f"Unknown models: {invalid}. Valid: {ALL_RAID_MODELS}")
        sys.exit(1)

    dim_kinds = _dedupe_preserve_order(list(args.dim_reduction))
    multi_dim = len(dim_kinds) > 1

    activations_root = Path(args.activations_dir)
    tokenized_root = Path(args.tokenized_dir)
    output_root = Path(args.output_dir)

    names = [get_dim_reduction(k).spec.name for k in dim_kinds]
    print("=" * 60)
    print("  RAID ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"  Models:           {models}")
    print(f"  Activations dir:  {activations_root}")
    print(f"  Output dir:       {output_root}")
    print(f"  Dim reduction(s): {', '.join(names)}")
    if multi_dim:
        print(f"  (outputs under each model: {' / '.join(dim_kinds)}/ )")
    print(f"  Exemplars:         {args.exemplars}")
    print(f"  Traits clustering: {args.traits_clustering}")
    clustering_kinds = _dedupe_preserve_order(list(args.clustering)) if args.clustering else None
    clustering_strategies = (
        clustering_strategies_from_names(clustering_kinds) if clustering_kinds else None
    )
    if clustering_kinds:
        print(f"  Clustering:        {', '.join(clustering_kinds)}")
    else:
        print("  Clustering:        (default: ward_silhouette, ward_gap, kmeans)")

    results: dict[tuple[str, str], bool] = {}
    for model in models:
        for i, kind in enumerate(dim_kinds):
            dim_red = get_dim_reduction(kind)
            out_sub = kind if multi_dim else None
            skip_neuron = multi_dim and i > 0
            exemplars_here = args.exemplars and i == 0

            ok = analyze_model(
                model=model,
                activations_root=activations_root,
                tokenized_root=tokenized_root,
                output_root=output_root,
                run_exemplars=exemplars_here,
                split=args.split,
                dim_reduction=dim_red,
                clustering_strategies=clustering_strategies,
                output_subdir=out_sub,
                skip_neuron_analysis=skip_neuron,
                traits_clustering=args.traits_clustering and not skip_neuron,
            )
            results[(model, kind)] = ok

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for model in models:
        for kind in dim_kinds:
            ok = results[(model, kind)]
            status = "OK  " if ok else "SKIP"
            rel = output_root / slug(model)
            if multi_dim:
                rel = rel / kind
            print(f"  {model:16s}  {kind:6s}  {status}  {rel}/")

    n_ok = sum(results.values())
    print(f"\n  {n_ok}/{len(results)} model×embedding runs completed")
    print(f"  Results in: {output_root.resolve()}")


if __name__ == "__main__":
    main()
