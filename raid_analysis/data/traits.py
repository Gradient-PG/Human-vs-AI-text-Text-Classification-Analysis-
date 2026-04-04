"""Build neuron × trait matrices from stats CSVs and raw activations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_neuron_columns(neurons_df: pd.DataFrame) -> pd.DataFrame:
    """Add mean_diff and Cohen's d (robust, IQR-based) for each neuron."""
    df = neurons_df.copy()
    df["mean_diff"] = df["ai_median"] - df["human_median"]
    ai_iqr = df["ai_q75"] - df["ai_q25"]
    human_iqr = df["human_q75"] - df["human_q25"]
    pooled = (ai_iqr + human_iqr) / 2 / 1.35
    df["cohens_d"] = df["mean_diff"] / pooled.replace(0, 1e-10)
    return df


def build_trait_matrix(
    neurons_df: pd.DataFrame,
    X_all: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Assemble a dense trait matrix (n_neurons, n_traits) and column names.

    Traits are per-neuron statistics from analyze_activations plus simple
    activation summaries (pooled mean/std and per-class means).
    """
    df = add_derived_neuron_columns(neurons_df)
    n = X_all.shape[0]

    ai_mask = labels == 1
    hu_mask = labels == 0

    mean_p = X_all.mean(axis=1)
    std_p = X_all.std(axis=1)
    mean_ai = X_all[:, ai_mask].mean(axis=1)
    mean_hu = X_all[:, hu_mask].mean(axis=1)
    mean_diff_act = mean_ai - mean_hu

    stat_cols = [
        "neuron_idx",
        "layer",
        "ai_median",
        "human_median",
        "ai_q25",
        "ai_q75",
        "human_q25",
        "human_q75",
        "mean_diff",
        "cohens_d",
        "u_statistic",
        "p_value",
        "auc",
    ]

    sub = df[stat_cols].copy()
    T_stat = sub.to_numpy(dtype=np.float64)

    act_names = [
        "act_mean_pooled",
        "act_std_pooled",
        "act_mean_ai",
        "act_mean_human",
        "act_mean_diff_class",
    ]

    T_act = np.column_stack(
        [mean_p, std_p, mean_ai, mean_hu, mean_diff_act]
    )

    T = np.hstack([T_stat, T_act])
    names = stat_cols + act_names

    assert T.shape == (n, len(names))
    return T, names


def sanitize_trait_matrix(T: np.ndarray) -> np.ndarray:
    """Replace non-finite values for downstream scaling."""
    T = np.asarray(T, dtype=np.float64)
    return np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)


def traits_matrix_to_csv(path, T: np.ndarray, names: list[str]) -> None:
    """Write trait matrix; ``names`` already includes ``neuron_idx`` and ``layer`` from stats."""
    out = pd.DataFrame(T, columns=names)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
