"""Load neuron stats and activation matrices from analyze_activations outputs."""

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import ALL_LAYERS


def load_stats(results_path: Path):
    """Load per-layer neuron stats, compute Cohen's d, return (neurons_df, disc_df, labels)."""
    frames = []
    for layer in ALL_LAYERS:
        df = pd.read_csv(results_path / f"layer_{layer}_neuron_stats.csv")
        df["layer"] = layer
        frames.append(df)

    neurons_df = pd.concat(frames, ignore_index=True)
    disc_df = neurons_df[neurons_df["discriminative"]].copy()

    disc_df["mean_diff"] = disc_df["ai_median"] - disc_df["human_median"]
    disc_df["ai_iqr"] = disc_df["ai_q75"] - disc_df["ai_q25"]
    disc_df["human_iqr"] = disc_df["human_q75"] - disc_df["human_q25"]
    pooled_iqr = (disc_df["ai_iqr"] + disc_df["human_iqr"]) / 2 / 1.35
    disc_df["cohens_d"] = disc_df["mean_diff"] / pooled_iqr.replace(0, 1e-10)
    disc_df["variance_ratio"] = disc_df["ai_iqr"] / disc_df["human_iqr"].replace(0, 1e-10)

    labels = np.load(results_path / "labels.npy")
    return neurons_df, disc_df, labels


def load_activation_column(results_path: Path, layer: int, neuron_idx: int) -> np.ndarray:
    acts = np.load(results_path / f"layer_{layer}_activations.npy")
    return acts[:, neuron_idx]


def build_full_neuron_matrix(neurons_df: pd.DataFrame, results_path: Path) -> np.ndarray:
    """
    Build (n_all_neurons, n_samples) activation matrix for all 9,216 neurons.

    Row order matches neurons_df row order: layer 1 neurons 0-767, then layer 2, etc.
    """
    frames = []
    for layer in ALL_LAYERS:
        layer_acts = np.load(results_path / f"layer_{layer}_activations.npy")
        frames.append(layer_acts.T)
    return np.vstack(frames)
