"""Load selections and splits from a completed source experiment.

Dependent experiments (ablation, patching, confound) reuse Experiment 1's
neuron selections and CV splits.  This module provides the loading logic
so each orchestrator stays thin.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..data.splits import CVSplit
from ..selection.protocol import SelectionResult, load_selection


def load_source_selections(
    source_dir: Path,
    splits_by_seed: dict[int, list[CVSplit]],
) -> dict[tuple[int, int], SelectionResult]:
    """Load per-fold selections from a source experiment directory.

    Expects the directory layout produced by :func:`run_experiment`::

        source_dir/
          seed_{s}/
            fold_{f}/
              selection.json
              train_mean.npy

    Args:
        source_dir: Root of the source experiment output (e.g. the knee-C
            directory from a sparse probe sweep).
        splits_by_seed: The splits to iterate — only loads folds that exist.

    Returns:
        ``{(seed, fold_idx): SelectionResult}`` mapping.
    """
    selections: dict[tuple[int, int], SelectionResult] = {}

    for seed, folds in sorted(splits_by_seed.items()):
        for fold in folds:
            fold_dir = source_dir / f"seed_{seed}" / f"fold_{fold.fold_idx}"
            if not (fold_dir / "selection.json").exists():
                raise FileNotFoundError(
                    f"Selection not found at {fold_dir}. "
                    f"Did Experiment 1 run with seed={seed}, fold={fold.fold_idx}?"
                )
            selections[(seed, fold.fold_idx)] = load_selection(fold_dir)

    return selections


def resolve_knee_dir(sweep_dir: Path) -> Path:
    """Read ``sweep_result.json`` and return the path to the knee-C directory.

    Args:
        sweep_dir: Root of a sparse probe sweep output containing
            ``sweep_result.json``.

    Returns:
        Path to ``sweep_dir / C_{knee_C}``.
    """
    result_path = sweep_dir / "sweep_result.json"
    with open(result_path) as f:
        sweep = json.load(f)

    knee_C = sweep["knee_C"]
    knee_dir = sweep_dir / f"C_{knee_C}"

    if not knee_dir.exists():
        raise FileNotFoundError(
            f"Knee directory {knee_dir} does not exist. "
            f"Expected from knee_C={knee_C} in {result_path}"
        )

    return knee_dir
