"""D3 — Linear representation validation: mean difference vector and LR weight alignment."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def mean_difference_vector(
    activations: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Compute μ_AI − μ_human across all neurons.

    Args:
        activations: (N_samples, N_neurons) array.
        labels: (N_samples,) array of 0/1 labels (1 = AI).

    Returns:
        (N_neurons,) mean difference vector.
    """
    ai_mask = labels == 1
    human_mask = labels == 0
    return activations[ai_mask].mean(axis=0) - activations[human_mask].mean(axis=0)


def lr_weight_vector(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Train logistic regression on activations and return (weight_vector, accuracy).

    The weight vector is extracted after standardisation, so it lives in the
    original activation space — directly comparable to
    :func:`mean_difference_vector`.

    Args:
        activations: (N_samples, N_neurons) array.
        labels: (N_samples,) binary labels.
        max_iter: Solver iteration limit.
        random_state: Seed for reproducibility.

    Returns:
        (weights, accuracy) — weights shape is ``(N_neurons,)``.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(activations)

    lr = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    lr.fit(X_scaled, labels)
    accuracy = float(lr.score(X_scaled, labels))

    weights_scaled = lr.coef_[0]
    weights_original = weights_scaled / scaler.scale_

    return weights_original, accuracy
