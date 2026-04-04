"""Exemplar texts using AUC-based preference groups (not algorithmic clustering)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import write_text
from .summaries import exemplar_text


def assign_auc_preference_clusters(neurons_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``cluster`` column: 0 = non-discriminative, 1 = AI-preferring disc., 2 = human-preferring disc.

    Preferring split uses ``auc > 0.5`` vs ``<= 0.5`` on discriminative neurons (matches embedding plots).
    """
    df = neurons_df.copy()
    is_disc = df["discriminative"].values
    auc = df["auc"].values
    cluster = np.zeros(len(df), dtype=np.int64)
    cluster[is_disc & (auc > 0.5)] = 1
    cluster[is_disc & (auc <= 0.5)] = 2
    df["cluster"] = cluster
    return df


def run_exemplars_analysis(
    model: str,
    neurons_df: pd.DataFrame,
    X_all: np.ndarray,
    labels: np.ndarray,
    results_path: Path,
    tokenized_path: Path,
    split: str,
    neurons_out: Path,
) -> bool:
    """
    Write ``exemplars.txt`` under the neurons output directory.

    Returns True if the file was written. False if the tokenized dataset is
    missing or generation failed (caller should surface this when --exemplars).
    """
    df = assign_auc_preference_clusters(neurons_df)
    out_file = neurons_out / "exemplars.txt"
    if not tokenized_path.exists():
        print(
            "\n    *** EXEMPLARS NOT WRITTEN ***\n"
            f"    Tokenized RAID not found: {tokenized_path.resolve()}\n"
            "    Run tokenization first, e.g.:\n"
            f"      uv run scripts/tokenize_raid.py --model {model}\n"
            f"    (expects a HuggingFace dataset folder at the path above.)\n"
        )
        return False
    try:
        text = exemplar_text(
            X_all,
            df,
            labels,
            results_path,
            tokenized_path,
            split,
            model,
        )
        write_text(out_file, text)
        return True
    except Exception as exc:
        print(f"    ERROR (exemplars not saved): {exc}")
        return False
