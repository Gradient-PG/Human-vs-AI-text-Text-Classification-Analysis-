"""D3 — Linear representation validation: mean difference vector and LR weight alignment."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
    train_size: float = 0.8,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Train logistic regression on activations and return (weight_vector, test_accuracy).

    The model is trained on ``train_size`` fraction of the data and evaluated on
    the held-out test split, so the returned accuracy reflects generalisation, not
    memorisation.

    The weight vector is un-standardised (divided by ``scaler.scale_``) so it
    lives in the original activation space and is directly comparable to
    :func:`mean_difference_vector`.

    Args:
        activations: (N_samples, N_neurons) array.
        labels: (N_samples,) binary labels.
        train_size: Fraction of data used for training (default 0.8).
        max_iter: Solver iteration limit.
        random_state: Seed for reproducibility.

    Returns:
        (weights, test_accuracy) — weights shape is ``(N_neurons,)``.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels,
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
    test_accuracy = float(lr.score(X_test_scaled, y_test))

    weights_scaled = lr.coef_[0]
    weights_original = weights_scaled / scaler.scale_

    return weights_original, test_accuracy
