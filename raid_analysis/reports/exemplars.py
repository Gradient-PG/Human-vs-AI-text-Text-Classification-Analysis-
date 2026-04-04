"""Exemplar texts using AUC-based preference groups (AI-preferring vs human-preferring)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import write_text


def assign_auc_preference_groups(neurons_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``preference_group`` column:
    0 = non-discriminative, 1 = AI-preferring disc., 2 = human-preferring disc.
    """
    df = neurons_df.copy()
    is_disc = df["discriminative"].values
    auc = df["auc"].values
    group = np.zeros(len(df), dtype=np.int64)
    group[is_disc & (auc > 0.5)] = 1
    group[is_disc & (auc <= 0.5)] = 2
    df["preference_group"] = group
    return df


def exemplar_text(
    X_neurons: np.ndarray,
    neuron_df: pd.DataFrame,
    labels: np.ndarray,
    results_path: Path,
    tokenized_path: Path,
    split: str,
    model: str,
    top_n: int = 20,
    text_preview: int = 500,
) -> str:
    """Generate exemplar text report for each preference group."""
    from datasets import concatenate_datasets, load_from_disk
    from transformers import AutoTokenizer

    with open(results_path / "metadata.json") as f:
        meta = json.load(f)
    samples_per_class = meta["n_samples"] // 2

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tok_dict = load_from_disk(str(tokenized_path))
    full_split = tok_dict[split]

    ai_samples = full_split.filter(lambda x: x["label"] == 1)
    human_samples = full_split.filter(lambda x: x["label"] == 0)
    ai_sub = ai_samples.shuffle(seed=42).select(range(min(samples_per_class, len(ai_samples))))
    hu_sub = human_samples.shuffle(seed=42).select(range(min(samples_per_class, len(human_samples))))
    ordered = concatenate_datasets([ai_sub, hu_sub]).shuffle(seed=42)

    reconstructed = np.array(ordered["label"])
    mismatches = int(np.sum(reconstructed != labels))

    has_domain = "domain" in ordered.column_names

    def _scalar_str(val) -> str:
        if val is None:
            return "?"
        if hasattr(val, "item") and callable(val.item):
            try:
                val = val.item()
            except (ValueError, RuntimeError):
                pass
        return str(val)

    def domain_suffix(idx: int) -> str:
        if not has_domain:
            return ""
        d = _scalar_str(ordered[int(idx)]["domain"])
        return f"  domain={d}"

    def get_text(idx: int) -> str:
        return tokenizer.decode(ordered[int(idx)]["input_ids"], skip_special_tokens=True)

    X_z = (X_neurons - X_neurons.mean(axis=1, keepdims=True)) / (
        X_neurons.std(axis=1, keepdims=True) + 1e-8
    )
    group_ids = sorted(neuron_df["preference_group"].unique())
    R = {k: X_z[neuron_df["preference_group"].values == k].mean(axis=0) for k in group_ids}
    specificity = {
        k: R[k] - np.mean([R[j] for j in group_ids if j != k], axis=0)
        for k in group_ids
    }

    ai_mask = labels == 1
    human_mask = labels == 0

    group_names = {0: "non-discriminative", 1: "AI-preferring", 2: "human-preferring"}

    lines = [
        f"AUC PREFERENCE EXEMPLARS  [{model}]",
        "Groups: non-discriminative (0), AI-preferring discriminative (1), "
        "human-preferring discriminative (2).",
        f"Top {top_n} exemplars per class per group",
    ]
    if has_domain:
        lines.append("Each line includes RAID domain when present in the tokenized dataset.")
    else:
        lines.append(
            "Note: no 'domain' column in tokenized data — re-run tokenize_raid.py "
            "(keeps domain) to show domains here."
        )
    if mismatches:
        lines.append(f"WARNING: {mismatches} label mismatches detected")

    for k in group_ids:
        scores = specificity[k]
        cn = neuron_df[neuron_df["preference_group"] == k]
        dominant_dir = cn["direction"].value_counts().index[0]

        ai_scores = np.where(ai_mask, scores, -np.inf)
        hu_scores = np.where(human_mask, scores, np.inf)
        top_ai = np.argsort(ai_scores)[-top_n:][::-1]
        top_hu = np.argsort(hu_scores)[:top_n]

        gname = group_names.get(k, f"group-{k}")
        lines += [
            "",
            "=" * 80,
            f"  GROUP {k} ({gname})  |  {len(cn)} neurons  |  dominant: {dominant_dir}",
            "=" * 80,
            "",
            "  >>> AI EXEMPLARS (group maximally active)",
        ]
        for rank, idx in enumerate(top_ai, 1):
            text = get_text(idx)
            preview = text[:text_preview] + " [...]" if len(text) > text_preview else text
            lines += [
                f"\n  [AI #{rank}]  sample_idx={idx}{domain_suffix(idx)}  score={scores[idx]:+.4f}",
                f"  {preview}",
            ]

        lines += ["", "  >>> HUMAN EXEMPLARS (group maximally suppressed)"]
        for rank, idx in enumerate(top_hu, 1):
            text = get_text(idx)
            preview = text[:text_preview] + " [...]" if len(text) > text_preview else text
            lines += [
                f"\n  [Human #{rank}]  sample_idx={idx}{domain_suffix(idx)}  score={scores[idx]:+.4f}",
                f"  {preview}",
            ]

    return "\n".join(lines)


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

    Returns True if the file was written, False if tokenized dataset is missing.
    """
    df = assign_auc_preference_groups(neurons_df)
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
