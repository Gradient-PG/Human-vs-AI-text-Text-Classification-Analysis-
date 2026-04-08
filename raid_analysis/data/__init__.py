"""Data loading, neuron statistics computation, and derived features."""

from .activations import (
    build_full_neuron_matrix,
    concat_all_layers,
    global_indices_to_neuron_set,
    global_to_layer_neuron,
    layer_neuron_to_global,
    load_activation_column,
    load_activations,
    load_activations_for_model,
    neuron_set_to_global_indices,
)
from .discriminative import (
    get_discriminative_neuron_indices,
    get_discriminative_set,
    get_discriminative_sets_per_generator,
    get_layer_discriminative_indices,
)
from .loader import load_stats
from .metadata import (
    SampleMetadata,
    compute_metadata_from_dataset,
    load_metadata,
    metadata_exists,
    save_metadata,
)
from .neuron_stats import compute_neuron_statistics, identify_discriminative_neurons
from .splits import (
    CVSplit,
    generate_cv_splits,
    generate_multi_seed_splits,
    load_splits,
    save_splits,
)
from .traits import (
    add_derived_neuron_columns,
    build_trait_matrix,
    sanitize_trait_matrix,
    traits_matrix_to_csv,
)

__all__ = [
    # Activations
    "build_full_neuron_matrix",
    "concat_all_layers",
    "global_indices_to_neuron_set",
    "global_to_layer_neuron",
    "layer_neuron_to_global",
    "load_activation_column",
    "load_activations",
    "load_activations_for_model",
    "neuron_set_to_global_indices",
    # Discriminative neuron helpers (AUC-based, old pipeline)
    "get_discriminative_neuron_indices",
    "get_discriminative_set",
    "get_discriminative_sets_per_generator",
    "get_layer_discriminative_indices",
    # Stats loader
    "load_stats",
    # Per-sample metadata
    "SampleMetadata",
    "compute_metadata_from_dataset",
    "load_metadata",
    "metadata_exists",
    "save_metadata",
    # Neuron statistics
    "compute_neuron_statistics",
    "identify_discriminative_neurons",
    # CV splits
    "CVSplit",
    "generate_cv_splits",
    "generate_multi_seed_splits",
    "load_splits",
    "save_splits",
    # Traits
    "add_derived_neuron_columns",
    "build_trait_matrix",
    "sanitize_trait_matrix",
    "traits_matrix_to_csv",
]
