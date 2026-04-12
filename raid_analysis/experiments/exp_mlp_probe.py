"""Experiment 7 — MLP probe + Integrated Gradients (conditional).

Stub orchestrator.  Only run if Experiment 1 shows no clean knee in the
accuracy-vs-sparsity curve.  Will train an MLP probe, run IG attribution,
and compare the resulting neuron ranking with L1 selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..data.metadata import SampleMetadata
from ..data.splits import CVSplit
from .config import ExperimentConfig


def run_mlp_probe_experiment(
    activations: np.ndarray,
    labels: np.ndarray,
    metadata: SampleMetadata,
    splits_by_seed: dict[int, list[CVSplit]],
    config: ExperimentConfig,
    source_dir: Path,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run MLP probe + IG experiment.

    .. warning::
        Stub — raises ``NotImplementedError``.  Will be implemented when
        Experiment 1 results warrant it (no clean knee).

    Args:
        activations: ``(N, D)`` concatenated activations.
        labels: ``(N,)`` binary labels.
        metadata: Per-sample metadata.
        splits_by_seed: Must match Experiment 1's splits.
        config: Experiment configuration.
        source_dir: Root of the sparse probe sweep output.
        output_dir: If provided, persist results here.
    """
    raise NotImplementedError(
        "Experiment 7 (MLP probe + IG) is conditional. "
        "Implement when Experiment 1 shows no clean knee in the "
        "accuracy-vs-sparsity curve."
    )
