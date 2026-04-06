"""Convenience helpers for retrieving discriminative neuron sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .loader import load_stats


def get_discriminative_neuron_indices(
    neurons_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (layer, neuron_idx) arrays for AI-preferring and human-preferring neurons.

    Returns:
        (ai_indices, human_indices) — each is an (N, 2) int array with columns [layer, neuron_idx].
    """
    disc = neurons_df[neurons_df["discriminative"]].copy()
    ai = disc[disc["auc"] > 0.5][["layer", "neuron_idx"]].to_numpy(dtype=int)
    human = disc[disc["auc"] <= 0.5][["layer", "neuron_idx"]].to_numpy(dtype=int)
    return ai, human


def get_discriminative_set(neurons_df: pd.DataFrame) -> set[tuple[int, int]]:
    """Return the set of ``(layer, neuron_idx)`` tuples for all discriminative neurons."""
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
