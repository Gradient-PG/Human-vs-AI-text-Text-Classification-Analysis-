"""D1 — Probing classifier for causal validation experiments.

The typical D1 causal loop is:

1. Train a probe on original activations (``train_probe``).
2. Ablate or patch neurons (``ablate_neurons`` / ``patch_neurons``).
3. Score the **same** probe on the modified activations (``score_probe``).

The delta in accuracy between steps 1 and 3 measures the causal impact of the
ablated neurons on the detection signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedProbe:
    """A fitted logistic regression probe returned by :func:`train_probe`."""

    lr: LogisticRegression
    scaler: StandardScaler
    train_accuracy: float
    test_accuracy: float

    def __repr__(self) -> str:
        return (
            f"TrainedProbe(train_acc={self.train_accuracy:.4f}, "
            f"test_acc={self.test_accuracy:.4f})"
        )


def train_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    train_size: float = 0.8,
    max_iter: int = 1000,
    random_state: int = 42,
) -> TrainedProbe:
    """
    Fit a logistic regression probe on a train split and report held-out accuracy.

    Args:
        activations: ``(N_samples, N_neurons)`` activation array.
        labels: ``(N_samples,)`` binary labels (1 = AI, 0 = human).
        train_size: Fraction of samples used for training (default 0.8).
        max_iter: Solver iteration limit.
        random_state: Seed for reproducibility.

    Returns:
        :class:`TrainedProbe` with ``lr``, ``scaler``, ``train_accuracy``,
        and ``test_accuracy``.

    Example::

        probe = train_probe(activations[layer], labels)
        print(probe)  # TrainedProbe(train_acc=0.9800, test_acc=0.9750)

        ablated = ablate_neurons(activations[layer], ai_idx)
        delta = probe.test_accuracy - score_probe(probe, ablated, labels)
        print(f"Accuracy drop after ablation: {delta:.4f}")
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activations,
        labels,
        train_size=train_size,
        random_state=random_state,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    lr.fit(X_train_scaled, y_train)

    return TrainedProbe(
        lr=lr,
        scaler=scaler,
        train_accuracy=float(lr.score(X_train_scaled, y_train)),
        test_accuracy=float(lr.score(X_test_scaled, y_test)),
    )


def score_probe(
    probe: TrainedProbe,
    activations: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Score a pre-trained probe on (possibly modified) activations.

    The scaler fitted during :func:`train_probe` is re-used, so ``activations``
    must have the same feature dimensionality as the training data.

    Args:
        probe: A :class:`TrainedProbe` returned by :func:`train_probe`.
        activations: ``(N_samples, N_neurons)`` array — may be ablated or patched.
        labels: ``(N_samples,)`` binary ground-truth labels.

    Returns:
        Classification accuracy as a float in ``[0, 1]``.

    Example::

        # Baseline
        probe = train_probe(activations[layer], labels)

        # Necessity test (D1a)
        ai_idx, _ = get_layer_discriminative_indices(neurons_df, layer)
        ablated = ablate_neurons(activations[layer], ai_idx)
        drop = probe.test_accuracy - score_probe(probe, ablated, labels)
        print(f"Causal drop (AI neurons ablated): {drop:.4f}")
    """
    X_scaled = probe.scaler.transform(activations)
    return float(probe.lr.score(X_scaled, labels))
