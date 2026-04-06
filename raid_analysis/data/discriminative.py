"""Convenience helpers for retrieving discriminative neuron sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .loader import load_stats


def _require_layer_column(neurons_df: pd.DataFrame, fn_name: str) -> None:
    if "layer" not in neurons_df.columns:
        raise ValueError(
            f"{fn_name}() requires a 'layer' column in neurons_df. "
            "Pass the output of load_stats(), which concatenates per-layer CSVs and adds "
            "a 'layer' column. The raw output of compute_neuron_statistics() covers a single "
            "layer and does not include this column."
        )


def get_discriminative_neuron_indices(
    neurons_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ``(layer, neuron_idx)`` arrays for AI-preferring and human-preferring neurons.

    .. note::
        Requires ``neurons_df`` to have a ``layer`` column, i.e. the output of
        :func:`~raid_analysis.data.loader.load_stats`. The raw output of
        :func:`~raid_analysis.data.neuron_stats.compute_neuron_statistics` (single layer,
        no ``layer`` column) is **not** accepted.

    Returns:
        (ai_indices, human_indices) — each is an (N, 2) int array with columns
        ``[layer, neuron_idx]``.
    """
    _require_layer_column(neurons_df, "get_discriminative_neuron_indices")
    disc = neurons_df[neurons_df["discriminative"]].copy()
    ai = disc[disc["auc"] > 0.5][["layer", "neuron_idx"]].to_numpy(dtype=int)
    human = disc[disc["auc"] <= 0.5][["layer", "neuron_idx"]].to_numpy(dtype=int)
    return ai, human


def get_layer_discriminative_indices(
    neurons_df: pd.DataFrame,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return flat column indices (0–767) for AI-preferring and human-preferring
    discriminative neurons in a **single layer**.

    Unlike :func:`get_discriminative_neuron_indices`, the returned arrays are
    plain integer column indices suitable for direct use with
    :func:`~raid_analysis.experiments.causal.ablate_neurons` and
    :func:`~raid_analysis.experiments.causal.patch_neurons`, which operate on
    per-layer ``(N_samples, 768)`` activation arrays.

    Args:
        neurons_df: Output of :func:`~raid_analysis.data.loader.load_stats`
            (must have a ``layer`` column).
        layer: Layer index (1–12).

    Returns:
        (ai_indices, human_indices) — each is a 1-D int array of column indices
        into the per-layer activation matrix.

    Example::

        neurons_df, _, labels = load_stats(results_path)
        activations, _, _ = load_activations(results_path)

        for layer in ALL_LAYERS:
            ai_idx, human_idx = get_layer_discriminative_indices(neurons_df, layer)
            ablated = ablate_neurons(activations[layer], ai_idx)
    """
    _require_layer_column(neurons_df, "get_layer_discriminative_indices")
    layer_df = neurons_df[(neurons_df["discriminative"]) & (neurons_df["layer"] == layer)]
    ai_idx = layer_df[layer_df["auc"] > 0.5]["neuron_idx"].to_numpy(dtype=int)
    human_idx = layer_df[layer_df["auc"] <= 0.5]["neuron_idx"].to_numpy(dtype=int)
    return ai_idx, human_idx


def get_discriminative_set(neurons_df: pd.DataFrame) -> set[tuple[int, int]]:
    """Return the set of ``(layer, neuron_idx)`` tuples for all discriminative neurons."""
    _require_layer_column(neurons_df, "get_discriminative_set")
    disc = neurons_df[neurons_df["discriminative"]]
    return set(zip(disc["layer"].astype(int), disc["neuron_idx"].astype(int)))


def get_discriminative_sets_per_generator(
    models: list[str],
    results_root: str | Path,
) -> dict[str, set[tuple[int, int]]]:
    """
    Load neuron stats for each generator and return per-model discriminative sets.

    Args:
        models: RAID model names (e.g. ``["gpt4", "chatgpt"]``).
        results_root: Parent directory containing ``activations_raid_*`` folders.

    Returns:
        ``{model: {(layer, neuron_idx), ...}}`` for each generator.
    """
    from raid_pipeline.raid_loader import slug

    results_root = Path(results_root)
    out: dict[str, set[tuple[int, int]]] = {}
    for model in models:
        results_path = results_root / f"activations_raid_{slug(model)}"
        neurons_df, _disc_df, _labels = load_stats(results_path)
        out[model] = get_discriminative_set(neurons_df)
    return out
