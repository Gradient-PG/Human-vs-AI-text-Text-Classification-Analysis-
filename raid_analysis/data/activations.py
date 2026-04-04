"""Load raw activations from .npy files produced by the extraction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import ALL_LAYERS


def load_activations(
    results_path: str | Path,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """
    Load all per-layer activation arrays, labels, and metadata from disk.

    Returns:
        (activations_dict, labels, metadata) where activations_dict maps
        layer index -> (N_samples, 768) array.
    """
    results_path = Path(results_path)

    with open(results_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    activations = {}
    for layer in metadata["layers"]:
        layer_file = results_path / f"layer_{layer}_activations.npy"
        activations[layer] = np.load(layer_file)

    labels = np.load(results_path / "labels.npy")
    return activations, labels, metadata


def load_activations_for_model(
    model: str,
    results_root: str | Path,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    """
    Convenience wrapper that resolves the ``activations_raid_{slug}`` path convention.

    Args:
        model: RAID model name (e.g. ``"gpt4"``, ``"mistral-chat"``).
        results_root: Parent directory containing ``activations_raid_*`` folders.
    """
    from utils.raid_loader import slug

    results_root = Path(results_root)
    results_path = results_root / f"activations_raid_{slug(model)}"
    return load_activations(results_path)


def load_activation_column(
    results_path: Path, layer: int, neuron_idx: int
) -> np.ndarray:
    """Load a single neuron's activation vector across all samples."""
    acts = np.load(results_path / f"layer_{layer}_activations.npy")
    return acts[:, neuron_idx]


def build_full_neuron_matrix(
    neurons_df: pd.DataFrame, results_path: Path
) -> np.ndarray:
    """
    Build (n_all_neurons, n_samples) activation matrix for all 9,216 neurons.

    Row order matches neurons_df row order: layer 1 neurons 0-767, then layer 2, etc.
    """
    frames = []
    for layer in ALL_LAYERS:
        layer_acts = np.load(results_path / f"layer_{layer}_activations.npy")
        frames.append(layer_acts.T)
    return np.vstack(frames)
