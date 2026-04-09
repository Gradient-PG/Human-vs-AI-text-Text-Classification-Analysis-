"""Experiment primitives, CV runner, and experiment orchestrators."""

from .causal import ablate_neurons, patch_neurons
from .config import ExperimentConfig, SparseProbeSweepConfig, load_config, save_config
from .cross_generator import core_neurons, jaccard_matrix, jaccard_similarity
from .exp_sparse_probe import SparseProbeSweepResult, run_sparse_probe_sweep
from .linear import lr_weight_vector, mean_difference_vector
from .probing import TrainedProbe, score_probe, train_probe
from .runner import ExperimentResult, run_experiment

__all__ = [
    "ablate_neurons",
    "core_neurons",
    "ExperimentConfig",
    "ExperimentResult",
    "jaccard_matrix",
    "jaccard_similarity",
    "load_config",
    "lr_weight_vector",
    "mean_difference_vector",
    "patch_neurons",
    "run_experiment",
    "run_sparse_probe_sweep",
    "save_config",
    "score_probe",
    "SparseProbeSweepConfig",
    "SparseProbeSweepResult",
    "train_probe",
    "TrainedProbe",
]
