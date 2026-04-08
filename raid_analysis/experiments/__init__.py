"""Experiment primitives and CV runner."""

from .causal import ablate_neurons, patch_neurons
from .cross_generator import core_neurons, jaccard_matrix, jaccard_similarity
from .linear import lr_weight_vector, mean_difference_vector
from .probing import TrainedProbe, score_probe, train_probe
from .runner import ExperimentResult, run_experiment

__all__ = [
    "ablate_neurons",
    "core_neurons",
    "ExperimentResult",
    "jaccard_matrix",
    "jaccard_similarity",
    "lr_weight_vector",
    "mean_difference_vector",
    "patch_neurons",
    "run_experiment",
    "score_probe",
    "train_probe",
    "TrainedProbe",
]
