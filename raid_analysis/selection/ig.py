"""IGSelector — neuron selection via Integrated Gradients (conditional).

Stub implementation.  Only built out if Experiment 1 results show no clean
knee in the accuracy-vs-sparsity curve.  Requires torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import NeuronSelector, SelectionResult


class IGSelector(NeuronSelector):
    """Select neurons by integrated gradient attribution through a probe.

    .. warning::
        This is a stub.  The full implementation requires ``torch`` and will
        be built only if Experiment 1 results warrant it.

    Args:
        top_k: Number of neurons to select by |attribution|.
        probe_type: ``"linear"`` or ``"mlp"``.
        hidden_sizes: MLP hidden layer sizes (ignored for linear).
        baseline: IG baseline strategy (``"zero"`` or ``"mean"``).
        n_steps: Number of interpolation steps for IG.
    """

    def __init__(
        self,
        top_k: int = 100,
        probe_type: str = "mlp",
        hidden_sizes: list[int] | None = None,
        baseline: str = "zero",
        n_steps: int = 50,
    ) -> None:
        self.top_k = top_k
        self.probe_type = probe_type
        self.hidden_sizes = hidden_sizes or [256, 64]
        self.baseline = baseline
        self.n_steps = n_steps

    def select(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
    ) -> SelectionResult:
        raise NotImplementedError(
            "IGSelector is a stub. Implement when Experiment 1 results "
            "show no clean knee in the accuracy-vs-sparsity curve. "
            "Requires torch for autograd through a trained probe."
        )
