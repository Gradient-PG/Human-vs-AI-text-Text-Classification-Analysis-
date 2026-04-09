"""Experiment configuration: dataclasses, YAML loading, defaults, validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..constants import DEFAULT_N_FOLDS, DEFAULT_SEEDS


@dataclass
class ExperimentConfig:
    """Full configuration for an experiment run.

    Loadable from YAML via :func:`load_config` or constructed directly.
    """

    # Identity
    experiment: str = "sparse_probe"
    generators: list[str] = field(default_factory=lambda: ["gpt4"])
    pooled: bool = False

    # CV
    n_folds: int = DEFAULT_N_FOLDS
    seeds: list[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))

    # Selector
    selector: str = "sparse_probe"
    selector_params: dict[str, Any] = field(default_factory=dict)

    # Evaluator
    evaluator: str = "probe_accuracy"
    evaluator_params: dict[str, Any] = field(default_factory=dict)

    # Data
    max_samples: int = 10_000
    activations_root: str = "results"

    # Output
    output_dir: str = "results/experiments"

    # Experiment-specific
    stability_threshold: float = 0.8
    eval_probe_C: float = 1.0
    eval_probe_max_iter: int = 5000

    # Dependency on a prior experiment run
    source_experiment: str | None = None

    def validate(self) -> None:
        """Raise ``ValueError`` on invalid configuration."""
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if not self.generators:
            raise ValueError("generators must be non-empty")
        if self.stability_threshold < 0 or self.stability_threshold > 1:
            raise ValueError(
                f"stability_threshold must be in [0, 1], got {self.stability_threshold}"
            )


@dataclass
class SparseProbeSweepConfig(ExperimentConfig):
    """Extended config for the sparse probe sweep (Experiment 1)."""

    experiment: str = "sparse_probe"
    selector: str = "sparse_probe"
    evaluator: str = "probe_accuracy"

    C_values: list[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    )
    solver: str = "saga"
    max_iter: int = 5000


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Unknown keys are silently ignored so configs can contain comments or
    experiment-specific fields consumed by orchestrators.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return _dict_to_config(raw)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save config to YAML for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict
    data = asdict(config)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _dict_to_config(raw: dict) -> ExperimentConfig:
    """Build the right config subclass from a raw dict."""
    experiment = raw.get("experiment", "sparse_probe")

    if experiment == "sparse_probe":
        return _build(SparseProbeSweepConfig, raw)

    return _build(ExperimentConfig, raw)


def _build(cls: type, raw: dict) -> ExperimentConfig:
    """Construct *cls* from *raw*, ignoring unknown keys."""
    import dataclasses

    valid_fields = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    config = cls(**filtered)
    config.validate()
    return config
