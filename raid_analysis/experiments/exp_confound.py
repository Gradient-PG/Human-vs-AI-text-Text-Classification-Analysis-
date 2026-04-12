"""Experiment 4 — Confound checks.

Loads Experiment 1's splits and per-fold selection results, then runs
:class:`ConfoundEvaluator` on the same folds.  Does not re-select neurons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.metadata import SampleMetadata
from ..data.splits import CVSplit
from ..evaluation.confound import ConfoundEvaluator
from .config import ExperimentConfig, save_config
from .runner import ExperimentResult, run_experiment
from .source_loader import load_source_selections, resolve_knee_dir


def run_confound_experiment(
    activations: np.ndarray,
    labels: np.ndarray,
    metadata: SampleMetadata,
    splits_by_seed: dict[int, list[CVSplit]],
    config: ExperimentConfig,
    source_dir: Path,
    *,
    output_dir: Path | None = None,
) -> ExperimentResult:
    """Run confound checks using Experiment 1's neuron selections.

    Args:
        activations: ``(N, D)`` concatenated activations.
        labels: ``(N,)`` binary labels.
        metadata: Per-sample metadata.
        splits_by_seed: ``{seed: [CVSplit, ...]}`` — must match Exp 1's splits.
        config: Experiment configuration.
        source_dir: Root of the sparse probe sweep output.
        output_dir: If provided, persist results here.

    Returns:
        :class:`ExperimentResult` with confound correlation metrics per fold.
    """
    knee_dir = resolve_knee_dir(source_dir)
    selections = load_source_selections(knee_dir, splits_by_seed)

    evaluator = ConfoundEvaluator(
        confounds=config.evaluator_params.get("confounds"),
        significance_threshold=config.evaluator_params.get(
            "significance_threshold", 0.05,
        ),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")

    print(f"Confound experiment: source={knee_dir}")

    return run_experiment(
        activations=activations,
        labels=labels,
        metadata=metadata,
        splits_by_seed=splits_by_seed,
        evaluator=evaluator,
        precomputed_selections=selections,
        output_dir=output_dir,
        stability_threshold=config.stability_threshold,
        eval_probe_C=config.eval_probe_C,
        eval_probe_max_iter=config.eval_probe_max_iter,
    )
