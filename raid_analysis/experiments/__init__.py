"""Experiment primitives, CV runner, and experiment orchestrators."""

from .causal import ablate_neurons, patch_neurons
from .config import ExperimentConfig, SparseProbeSweepConfig, load_config, save_config
from .cross_generator import core_neurons, jaccard_matrix, jaccard_similarity
from .exp_ablation import run_ablation_experiment
from .exp_confound import run_confound_experiment
from .exp_patching import run_patching_experiment
from .exp_sparse_probe import SparseProbeSweepResult, run_sparse_probe_sweep
from .linear import lr_weight_vector, mean_difference_vector
from .probing import TrainedProbe, score_probe, train_probe
from .runner import ExperimentResult, run_experiment
from .source_loader import load_source_selections, resolve_knee_dir

__all__ = [
    "ablate_neurons",
    "core_neurons",
    "ExperimentConfig",
    "ExperimentResult",
    "jaccard_matrix",
    "jaccard_similarity",
    "load_config",
    "load_source_selections",
    "lr_weight_vector",
    "mean_difference_vector",
    "patch_neurons",
    "resolve_knee_dir",
    "run_ablation_experiment",
    "run_confound_experiment",
    "run_experiment",
    "run_patching_experiment",
    "run_sparse_probe_sweep",
    "save_config",
    "score_probe",
    "SparseProbeSweepConfig",
    "SparseProbeSweepResult",
    "train_probe",
    "TrainedProbe",
]
