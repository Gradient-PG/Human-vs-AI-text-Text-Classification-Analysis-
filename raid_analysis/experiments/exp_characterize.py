"""Experiment 5 — Characterize the validated neuron set.

No fold loop — operates on the aggregate stable neuron sets from Experiment 1.
Computes layer distribution, cross-generator Jaccard overlap, core neuron set,
and linear direction alignment (CAV).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.set_ops import core_neurons, jaccard_matrix
from ..utils.linear import lr_weight_vector, mean_difference_vector
from .config import ExperimentConfig, save_config


def run_characterize_experiment(
    stable_sets: dict[str, set[tuple[int, int]]],
    activations_by_generator: dict[str, np.ndarray],
    labels_by_generator: dict[str, np.ndarray],
    config: ExperimentConfig,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Characterize validated neuron sets across generators.

    Args:
        stable_sets: ``{generator: {(layer, neuron_idx), ...}}`` stable neuron
            sets from Experiment 1, one per generator.
        activations_by_generator: ``{generator: (N, D)}`` concatenated
            activations for computing linear directions.
        labels_by_generator: ``{generator: (N,)}`` binary labels.
        config: Experiment configuration.
        output_dir: If provided, persist results here.

    Returns:
        Dict with layer distributions, Jaccard matrix, core neurons, and
        CAV alignment scores.
    """
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")

    generators = sorted(stable_sets.keys())
    print(f"Characterize experiment: {len(generators)} generators")

    # Layer distribution per generator
    layer_distributions: dict[str, dict[int, int]] = {}
    for gen in generators:
        counts: Counter[int] = Counter()
        for layer, _ in stable_sets[gen]:
            counts[layer] += 1
        layer_distributions[gen] = dict(sorted(counts.items()))

    # Cross-generator Jaccard matrix
    jaccard_df, names = jaccard_matrix(stable_sets)
    jaccard_data = {
        "matrix": jaccard_df.values.tolist(),
        "generators": names,
    }

    # Core neurons
    min_k = max(2, len(generators) // 2)
    core_set = core_neurons(stable_sets, min_generators=min_k)
    core_list = sorted([list(t) for t in core_set])

    # Linear direction alignment (CAV)
    cav_scores: dict[str, dict[str, Any]] = {}
    for gen in generators:
        if gen not in activations_by_generator:
            continue
        acts = activations_by_generator[gen]
        lbls = labels_by_generator[gen]

        mean_diff = mean_difference_vector(acts, lbls)
        lr_weights, lr_acc = lr_weight_vector(acts, lbls)

        mean_diff_norm = np.linalg.norm(mean_diff)
        lr_norm = np.linalg.norm(lr_weights)
        if mean_diff_norm > 0 and lr_norm > 0:
            cosine = float(
                np.dot(mean_diff, lr_weights) / (mean_diff_norm * lr_norm)
            )
        else:
            cosine = 0.0

        cav_scores[gen] = {
            "cosine_similarity": cosine,
            "lr_accuracy": float(lr_acc),
            "mean_diff_norm": float(mean_diff_norm),
            "lr_norm": float(lr_norm),
        }

    result = {
        "layer_distributions": layer_distributions,
        "jaccard": jaccard_data,
        "core_neurons": core_list,
        "core_min_generators": min_k,
        "n_core": len(core_set),
        "cav_scores": cav_scores,
        "n_generators": len(generators),
        "per_generator_n_stable": {
            gen: len(stable_sets[gen]) for gen in generators
        },
    }

    if output_dir is not None:
        with open(output_dir / "characterize_results.json", "w") as f:
            json.dump(result, f, indent=2)

    return result
