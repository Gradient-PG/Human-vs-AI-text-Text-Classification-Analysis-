#!/usr/bin/env python
"""
CLI for per-model RAID analysis: neuron statistics, optional clustering, optional exemplars.

Outputs (per model slug under ``--output-dir``):

- ``{model}/neurons/`` — layer/box/scatter figures, ``neurons_summary.txt``,
  ``figures/neuron_{pca|umap}_all.png`` (2D embedding of all neurons).
- ``{model}/clustering/`` — only if ``--clustering`` is passed: PCA variance report,
  algorithm-specific figures and ``clustering_*.txt`` files.
- ``{model}/neurons/exemplars.txt`` — with ``--exemplars`` (AUC preference groups,
  not clustering algorithms).

Prerequisites: run the RAID pipeline first, e.g.
``uv run scripts/run_raid_pipeline.py --samples 1000``

Usage:
    uv run scripts/analyze_raid_models.py
    uv run scripts/analyze_raid_models.py --models gpt4 chatgpt
    uv run scripts/analyze_raid_models.py --clustering ward_gap hdbscan kmeans
    uv run scripts/analyze_raid_models.py --neuron-viz umap --cluster-viz umap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from raid_analysis.clustering import CLUSTERING_STRATEGY_IDS
from raid_analysis.run_analysis import analyze_raid_model
from raid_pipeline.raid_loader import ALL_RAID_MODELS, slug


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAID neuron analysis (neurons/ + optional clustering/ + exemplars)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to analyze (default: all). Choices: {', '.join(ALL_RAID_MODELS)}",
    )
    parser.add_argument(
        "--activations-dir",
        default="results",
        help="Root dir containing activations_raid_* folders (default: results)",
    )
    parser.add_argument(
        "--tokenized-dir",
        default="data/processed",
        help="Root dir containing raid_* tokenized datasets (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analysis",
        help="Root dir for analysis outputs (default: results/analysis)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split used during extraction (default: train)",
    )
    parser.add_argument(
        "--exemplars",
        action="store_true",
        help=(
            "Write neurons/exemplars.txt (AUC preference groups). "
            "Requires data/processed/raid_<slug>/ from tokenize_raid.py for each model."
        ),
    )
    parser.add_argument(
        "--cluster-pca-components",
        type=int,
        default=50,
        metavar="K",
        help="PCA dimensionality for clustering algorithms (default: 50)",
    )
    parser.add_argument(
        "--neuron-viz",
        choices=("pca", "umap"),
        default="pca",
        help="2D projection for neuron overview figures only (default: pca)",
    )
    parser.add_argument(
        "--cluster-viz",
        choices=("pca", "umap"),
        default="pca",
        help="2D projection for clustering overlay figures only (default: pca)",
    )
    parser.add_argument(
        "--clustering",
        nargs="+",
        default=None,
        choices=CLUSTERING_STRATEGY_IDS,
        metavar="METHOD",
        help=(
            "Clustering methods to run into clustering/ (default: off). "
            f"Choices: {', '.join(CLUSTERING_STRATEGY_IDS)}."
        ),
    )
    args = parser.parse_args()

    if args.cluster_pca_components < 2:
        parser.error("--cluster-pca-components must be >= 2")

    models = args.models or ALL_RAID_MODELS
    invalid = [m for m in models if m not in ALL_RAID_MODELS]
    if invalid:
        print(f"Unknown models: {invalid}. Valid: {ALL_RAID_MODELS}")
        sys.exit(1)

    activations_root = Path(args.activations_dir)
    tokenized_root = Path(args.tokenized_dir)
    output_root = Path(args.output_dir)

    clustering = tuple(args.clustering) if args.clustering else None

    print("=" * 60)
    print("  RAID ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"  Models:          {models}")
    print(f"  Activations dir: {activations_root}")
    print(f"  Output dir:      {output_root}")
    print(f"  Layout:          {{model}}/neurons/  and  {{model}}/clustering/")
    print(f"  Exemplars:       {args.exemplars}")
    print(f"  Cluster PCA K:   {args.cluster_pca_components}")
    print(f"  Neuron viz 2D:   {args.neuron_viz}")
    print(f"  Cluster viz 2D:  {args.cluster_viz}")
    print(f"  Clustering:      {', '.join(clustering) if clustering else '(off)'}")

    results: dict[str, bool] = {}
    for model in models:
        ok = analyze_raid_model(
            model=model,
            activations_root=activations_root,
            tokenized_root=tokenized_root,
            output_root=output_root,
            run_exemplars=args.exemplars,
            split=args.split,
            cluster_pca_components=args.cluster_pca_components,
            neuron_viz=args.neuron_viz,
            cluster_viz=args.cluster_viz,
            clustering=clustering,
        )
        results[model] = ok

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for model, ok in results.items():
        status = "OK  " if ok else "SKIP"
        base = output_root / slug(model)
        print(f"  {model:20s}  {status}  {base}/")
        print(f"    {base / 'neurons'}")
        if clustering:
            print(f"    {base / 'clustering'}")

    n_ok = sum(results.values())
    print(f"\n  {n_ok}/{len(results)} models analyzed")
    print(f"  Results in: {output_root.resolve()}")


if __name__ == "__main__":
    main()
