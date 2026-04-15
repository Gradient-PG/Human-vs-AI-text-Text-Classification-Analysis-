"""SparseProbeSelector — L1 logistic regression for neuron selection.

Trains an L1-penalised LogReg on the provided (train-fold) activations.
Neurons with nonzero coefficients form the selected set.  The ranking is
by descending ``|coef_|``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..data.activations import global_to_layer_neuron
from .protocol import NeuronSelector, SelectionResult


class SparseProbeSelector(NeuronSelector):
    """Select neurons via L1-penalised logistic regression.

    Args:
        C: Inverse regularisation strength.  Smaller C = stronger L1 penalty
            = fewer nonzero neurons.
        solver: Sklearn solver supporting L1 (``"saga"`` or ``"liblinear"``).
        max_iter: Solver iteration limit.
        random_state: Seed for solver reproducibility.  When run through
            :func:`~raid_analysis.experiments.runner.run_experiment`, this is
            **overridden** to the current CV seed before each fold call.
    """

    def __init__(
        self,
        *,
        C: float = 0.1,
        solver: str = "saga",
        max_iter: int = 5000,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state

    def select(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
    ) -> SelectionResult:
        """Train L1 probe and return nonzero neurons as the selected set.

        Args:
            activations: ``(N_train, D)`` concatenated activations.
            labels: ``(N_train,)`` binary labels.

        Returns:
            :class:`SelectionResult` with neuron set, ranking, trained probe,
            and train-fold mean vector.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(activations)

        lr = LogisticRegression(
            penalty="l1",
            C=self.C,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        lr.fit(X_scaled, labels)

        coef = lr.coef_[0]
        abs_coef = np.abs(coef)

        nonzero_mask = abs_coef > 0
        nonzero_global = np.where(nonzero_mask)[0]

        neuron_indices: set[tuple[int, int]] = {
            global_to_layer_neuron(int(g)) for g in nonzero_global
        }

        ranked_global = np.argsort(-abs_coef)
        ranking: list[tuple[int, int]] = [
            global_to_layer_neuron(int(g))
            for g in ranked_global
            if abs_coef[g] > 0
        ]

        train_mean = activations.mean(axis=0)

        train_accuracy = float(lr.score(X_scaled, labels))

        return SelectionResult(
            neuron_indices=neuron_indices,
            ranking=ranking,
            probe=lr,
            train_mean=train_mean,
            metadata={
                "C": self.C,
                "solver": self.solver,
                "n_nonzero": len(neuron_indices),
                "n_features": activations.shape[1],
                "train_accuracy": train_accuracy,
            },
        )

    def __repr__(self) -> str:
        return (
            f"SparseProbeSelector(C={self.C}, solver={self.solver!r}, "
            f"max_iter={self.max_iter})"
        )
