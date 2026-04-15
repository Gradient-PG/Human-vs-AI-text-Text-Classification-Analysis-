#!/usr/bin/env python
"""
Run a single experiment by name with a YAML config.

Usage:
    uv run scripts/experiments/run_experiment.py sparse_probe
    uv run scripts/experiments/run_experiment.py ablation --config config/experiments/ablation.yaml
    uv run scripts/experiments/run_experiment.py --list
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from raid_analysis.data.activations import concat_all_layers, load_activations
from raid_analysis.data.metadata import load_metadata
from raid_analysis.data.splits import (
    generate_multi_seed_splits,
    load_splits,
    save_splits,
)
from raid_analysis.experiments.config import (
    ExperimentConfig,
    SparseProbeSweepConfig,
    load_config,
)

EXPERIMENT_NAMES = [
    "sparse_probe",
    "ablation",
    "patching",
    "confound",
    "characterize",
    "auc_comparison",
    "mlp_probe",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single experiment from the experiment catalog."
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        choices=EXPERIMENT_NAMES,
        help="Experiment to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. Uses default config if not provided.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="gpt4",
        help="Generator to run on (default: gpt4).",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Source experiment directory (for dependent experiments).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory override.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name in EXPERIMENT_NAMES:
            print(f"  {name}")
        return

    if args.experiment is None:
        parser.error("experiment is required (or use --list)")

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        default_config = project_root / "config" / "experiments" / f"{args.experiment}.yaml"
        if default_config.exists():
            config = load_config(default_config)
        else:
            config = ExperimentConfig(experiment=args.experiment)

    config.experiment = args.experiment

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.output_dir) / args.experiment / args.generator

    print(f"Experiment: {args.experiment}")
    print(f"Generator: {args.generator}")
    print(f"Output: {output_dir}")
    print()

    t0 = time.time()

    if args.experiment == "characterize":
        _run_characterize(config, args, output_dir)
    else:
        _run_standard(config, args, output_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


def _load_data(config: ExperimentConfig, generator: str):
    """Load activations, labels, metadata, and optionally subsample.

    When ``config.max_samples`` is set and smaller than the number of
    extracted samples, a balanced (50/50 label) random subsample is drawn
    so experiments can iterate quickly without re-extracting activations.
    """
    from raid_pipeline.raid_loader import slug

    results_root = Path(config.activations_root)
    results_path = results_root / f"activations_raid_{slug(generator)}"

    acts_dict, labels, meta_json = load_activations(results_path)
    activations = concat_all_layers(acts_dict)

    metadata = load_metadata(results_path)

    n_total = len(labels)
    max_n = config.max_samples
    if max_n and max_n < n_total:
        import numpy as np

        rng = np.random.RandomState(42)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        per_class = max_n // 2

        chosen_pos = rng.choice(pos_idx, size=min(per_class, len(pos_idx)), replace=False)
        chosen_neg = rng.choice(neg_idx, size=min(per_class, len(neg_idx)), replace=False)
        keep = np.sort(np.concatenate([chosen_pos, chosen_neg]))

        activations = activations[keep]
        labels = labels[keep]
        metadata = metadata[keep]
        print(f"Subsampled {n_total} → {len(labels)} samples (balanced)")

    return activations, labels, metadata


def _get_splits(config, labels, metadata, output_dir):
    """Generate or load CV splits."""
    splits_path = output_dir / "splits.json"
    if splits_path.exists():
        print(f"Loading existing splits from {splits_path}")
        return load_splits(splits_path)

    splits_by_seed = generate_multi_seed_splits(
        labels,
        n_folds=config.n_folds,
        seeds=config.seeds,
        domain_ids=metadata.domain_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_splits(splits_by_seed, splits_path)
    print(f"Saved splits to {splits_path}")
    return splits_by_seed


def _run_standard(config, args, output_dir):
    """Run a standard fold-based experiment."""
    activations, labels, metadata = _load_data(config, args.generator)
    splits_by_seed = _get_splits(config, labels, metadata, output_dir)

    if args.experiment == "sparse_probe":
        from raid_analysis.experiments.exp_sparse_probe import run_sparse_probe_sweep

        if not isinstance(config, SparseProbeSweepConfig):
            valid_fields = {f.name for f in dataclasses.fields(SparseProbeSweepConfig)}
            config = SparseProbeSweepConfig(
                **{k: v for k, v in config.__dict__.items() if k in valid_fields}
            )
        run_sparse_probe_sweep(
            activations, labels, metadata, splits_by_seed,
            config, output_dir=output_dir,
        )

    elif args.experiment == "ablation":
        from raid_analysis.experiments.exp_ablation import run_ablation_experiment

        source_dir = _resolve_source(args, config)
        run_ablation_experiment(
            activations, labels, metadata, splits_by_seed,
            config, source_dir, output_dir=output_dir,
        )

    elif args.experiment == "patching":
        from raid_analysis.experiments.exp_patching import run_patching_experiment

        source_dir = _resolve_source(args, config)
        run_patching_experiment(
            activations, labels, metadata, splits_by_seed,
            config, source_dir, output_dir=output_dir,
        )

    elif args.experiment == "confound":
        from raid_analysis.experiments.exp_confound import run_confound_experiment

        source_dir = _resolve_source(args, config)
        run_confound_experiment(
            activations, labels, metadata, splits_by_seed,
            config, source_dir, output_dir=output_dir,
        )

    elif args.experiment == "auc_comparison":
        from raid_analysis.experiments.exp_auc_comparison import run_auc_comparison

        source_dir = _resolve_source(args, config)
        run_auc_comparison(
            activations, labels, metadata, splits_by_seed,
            config, source_dir, output_dir=output_dir,
        )

    elif args.experiment == "mlp_probe":
        from raid_analysis.experiments.exp_mlp_probe import run_mlp_probe_experiment

        source_dir = _resolve_source(args, config)
        run_mlp_probe_experiment(
            activations, labels, metadata, splits_by_seed,
            config, source_dir, output_dir=output_dir,
        )


def _run_characterize(config, args, output_dir):
    """Run the characterize experiment across multiple generators."""
    from raid_analysis.experiments.exp_characterize import run_characterize_experiment

    generators = config.generators
    source_dir = _resolve_source(args, config)

    stable_sets: dict[str, set] = {}
    activations_by_gen: dict[str, any] = {}
    labels_by_gen: dict[str, any] = {}

    for gen in generators:
        acts, labels, metadata = _load_data(config, gen)
        activations_by_gen[gen] = acts
        labels_by_gen[gen] = labels

        # Load stable set from source experiment
        gen_source = source_dir.parent / gen
        if gen_source.exists():
            import json
            sweep_path = gen_source / "sweep_result.json"
            if sweep_path.exists():
                with open(sweep_path) as f:
                    sweep = json.load(f)
                stable_sets[gen] = {
                    tuple(n) for n in sweep.get("stable_neurons", [])
                }
            else:
                print(f"  Warning: no sweep_result.json for {gen}, skipping")
        else:
            print(f"  Warning: no source directory for {gen} at {gen_source}")

    if not stable_sets:
        print("No stable sets found. Run sparse_probe experiment first.")
        return

    run_characterize_experiment(
        stable_sets, activations_by_gen, labels_by_gen,
        config, output_dir=output_dir,
    )


def _resolve_source(args, config) -> Path:
    """Resolve the source experiment directory.

    For non-characterize experiments, appends the generator name so the path
    points to the per-generator sweep root (e.g. ``results/experiments/sparse_probe/gpt4``).
    """
    if args.source_dir:
        return Path(args.source_dir)
    if config.source_experiment:
        return Path(config.output_dir) / config.source_experiment / args.generator
    return Path(config.output_dir) / "sparse_probe" / args.generator


if __name__ == "__main__":
    main()
