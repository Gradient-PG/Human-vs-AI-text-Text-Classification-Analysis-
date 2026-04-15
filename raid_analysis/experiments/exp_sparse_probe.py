"""Experiment 1 — Sparse probe sweep.

Sweeps L1 penalty (C values), calling the generic experiment runner for each C.
Builds the accuracy-vs-sparsity curve, detects the knee, and computes the stable
neuron set across seeds × folds at the knee C value.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.metadata import SampleMetadata
from ..data.splits import CVSplit
from ..evaluation.probe_accuracy import ProbeAccuracyEvaluator
from ..selection.sparse_probe import SparseProbeSelector
from .config import SparseProbeSweepConfig, save_config
from .runner import ExperimentResult, run_experiment


@dataclass
class SweepPoint:
    """Results for a single C value across all folds × seeds."""

    C: float
    mean_accuracy: float
    std_accuracy: float
    mean_n_nonzero: float
    std_n_nonzero: float
    experiment_result: ExperimentResult


@dataclass
class SparseProbeSweepResult:
    """Full output of the sparse probe sweep experiment."""

    sweep_points: list[SweepPoint]
    knee_idx: int
    knee_C: float
    stable_neurons: set[tuple[int, int]]
    neuron_stability: dict[tuple[int, int], float]
    config: SparseProbeSweepConfig

    @property
    def knee_point(self) -> SweepPoint:
        return self.sweep_points[self.knee_idx]


def run_sparse_probe_sweep(
    activations: np.ndarray,
    labels: np.ndarray,
    metadata: SampleMetadata,
    splits_by_seed: dict[int, list[CVSplit]],
    config: SparseProbeSweepConfig,
    *,
    output_dir: Path | None = None,
) -> SparseProbeSweepResult:
    """Run the full sparse probe sweep experiment.

    For each C value in ``config.C_values``, constructs a
    :class:`SparseProbeSelector` and delegates to :func:`run_experiment`
    which handles the fold loop, data separation, and per-fold persistence.

    Args:
        activations: ``(N, D)`` concatenated activations.
        labels: ``(N,)`` binary labels.
        metadata: Per-sample metadata.
        splits_by_seed: ``{seed: [CVSplit, ...]}`` CV partitions.
        config: Sweep configuration.
        output_dir: If provided, persist results per C value.

    Returns:
        :class:`SparseProbeSweepResult` with sweep curve, knee, and stable set.
    """
    if not config.C_values:
        raise ValueError("C_values must be non-empty for sparse probe sweep")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, output_dir / "config.yaml")

    evaluator = ProbeAccuracyEvaluator()
    sweep_points: list[SweepPoint] = []

    total_folds = sum(len(folds) for folds in splits_by_seed.values())
    print(f"Sparse probe sweep: {len(config.C_values)} C values")
    print(f"  Folds per seed: {config.n_folds}")
    print(f"  Seeds: {config.seeds}")
    print(f"  Total fold runs per C: {total_folds}")

    for c_idx, C in enumerate(config.C_values):
        t0 = time.time()
        print(f"\n{'='*50}")
        print(f"  C = {C}  ({c_idx + 1}/{len(config.C_values)})")
        print(f"{'='*50}")

        selector = SparseProbeSelector(
            C=C,
            solver=config.solver,
            max_iter=config.max_iter,
        )

        c_dir = output_dir / f"C_{C}" if output_dir else None

        result = run_experiment(
            activations=activations,
            labels=labels,
            metadata=metadata,
            splits_by_seed=splits_by_seed,
            selector=selector,
            evaluator=evaluator,
            output_dir=c_dir,
            stability_threshold=config.stability_threshold,
            eval_probe_C=config.eval_probe_C,
            eval_probe_max_iter=config.eval_probe_max_iter,
        )

        accuracies = [fr.metrics["accuracy"] for fr in result.fold_results]
        n_nonzeros = [len(fr.neuron_indices) for fr in result.fold_results]

        sp = SweepPoint(
            C=C,
            mean_accuracy=float(np.mean(accuracies)),
            std_accuracy=float(np.std(accuracies)),
            mean_n_nonzero=float(np.mean(n_nonzeros)),
            std_n_nonzero=float(np.std(n_nonzeros)),
            experiment_result=result,
        )
        sweep_points.append(sp)

        elapsed = time.time() - t0
        print(
            f"  → acc={sp.mean_accuracy:.4f}±{sp.std_accuracy:.4f}  "
            f"nonzero={sp.mean_n_nonzero:.0f}±{sp.std_n_nonzero:.0f}  "
            f"({elapsed:.1f}s)"
        )

    # Knee detection
    knee_idx = _find_knee(sweep_points)
    knee_C = sweep_points[knee_idx].C
    print(f"\nKnee detected at C={knee_C} (index {knee_idx})")

    # Stability at knee — already computed by runner, just read it
    knee_result = sweep_points[knee_idx].experiment_result
    stability = knee_result.neuron_stability
    stable = knee_result.stable_neurons
    print(
        f"Stable neurons (threshold={config.stability_threshold}): "
        f"{len(stable)}"
    )

    result = SparseProbeSweepResult(
        sweep_points=sweep_points,
        knee_idx=knee_idx,
        knee_C=knee_C,
        stable_neurons=stable,
        neuron_stability=stability,
        config=config,
    )

    if output_dir is not None:
        _save_sweep_result(result, output_dir)

    return result


# ---------------------------------------------------------------------------
# Knee detection
# ---------------------------------------------------------------------------

def _find_knee(sweep_points: list[SweepPoint]) -> int:
    """Find the elbow in the accuracy-vs-sparsity curve.

    Uses the maximum-distance-to-line method: draw a line from the most
    sparse point to the least sparse point, and find the point with the
    greatest perpendicular distance from that line.

    Falls back to the midpoint if all points are collinear or there are
    fewer than 3 points.
    """
    if len(sweep_points) < 3:
        return len(sweep_points) // 2

    xs = np.array([sp.mean_n_nonzero for sp in sweep_points])
    ys = np.array([sp.mean_accuracy for sp in sweep_points])

    order = np.argsort(xs)
    xs_sorted = xs[order]
    ys_sorted = ys[order]

    p1 = np.array([xs_sorted[0], ys_sorted[0]])
    p2 = np.array([xs_sorted[-1], ys_sorted[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-12:
        return len(sweep_points) // 2

    line_unit = line_vec / line_len

    distances = np.zeros(len(xs_sorted))
    for i in range(len(xs_sorted)):
        point = np.array([xs_sorted[i], ys_sorted[i]])
        proj = np.dot(point - p1, line_unit)
        closest = p1 + proj * line_unit
        distances[i] = np.linalg.norm(point - closest)

    best_sorted_idx = int(np.argmax(distances))
    return int(order[best_sorted_idx])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _save_sweep_result(result: SparseProbeSweepResult, output_dir: Path) -> None:
    """Save sweep summary alongside per-C fold data."""
    sweep_summary = []
    for sp in result.sweep_points:
        sweep_summary.append({
            "C": sp.C,
            "mean_accuracy": sp.mean_accuracy,
            "std_accuracy": sp.std_accuracy,
            "mean_n_nonzero": sp.mean_n_nonzero,
            "std_n_nonzero": sp.std_n_nonzero,
        })

    stable_serializable = sorted([list(t) for t in result.stable_neurons])
    stability_serializable = {
        f"{layer}_{neuron}": frac
        for (layer, neuron), frac in sorted(result.neuron_stability.items())
    }

    payload = {
        "sweep": sweep_summary,
        "knee_idx": result.knee_idx,
        "knee_C": result.knee_C,
        "n_stable_neurons": len(result.stable_neurons),
        "stable_neurons": stable_serializable,
        "stability_threshold": result.config.stability_threshold,
        "neuron_stability": stability_serializable,
    }

    with open(output_dir / "sweep_result.json", "w") as f:
        json.dump(payload, f, indent=2)
