#!/usr/bin/env python
"""
Analyze BERT layer activations to identify discriminative neurons.
Uses Mann-Whitney U test and AUC (with Bonferroni correction) to score each neuron.
Saves per-layer statistics CSVs that are consumed by the analysis notebooks.

Usage:
    uv run scripts/analyze_activations.py
    uv run scripts/analyze_activations.py --input results/activations
    uv run scripts/analyze_activations.py --wandb-project human-vs-ai  (optional)
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json


def load_activations(input_dir: str):
    """Load activation data from directory."""
    input_path = Path(input_dir)

    with open(input_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    print(f"Loading activations from {input_path}")
    print(f"  Layers: {metadata['layers']}")
    print(f"  Samples: {metadata['n_samples']} ({metadata['n_ai']} AI, {metadata['n_human']} Human)")

    activations = {}
    for layer in metadata["layers"]:
        layer_file = input_path / f"layer_{layer}_activations.npy"
        activations[layer] = np.load(layer_file)
        print(f"  Loaded layer {layer}: {activations[layer].shape}")

    labels = np.load(input_path / "labels.npy")

    return activations, labels, metadata


def compute_neuron_statistics(activations: np.ndarray, labels: np.ndarray):
    """
    Compute per-neuron Mann-Whitney U test and AUC effect size.

    Args:
        activations: (N_samples, N_neurons) array
        labels: (N_samples,) array of 0/1 labels

    Returns:
        DataFrame with per-neuron statistics
    """
    ai_mask = labels == 1
    human_mask = labels == 0
    n_neurons = activations.shape[1]

    results = []

    for neuron_idx in tqdm(range(n_neurons), desc="Testing neurons"):
        ai_acts = activations[ai_mask, neuron_idx]
        human_acts = activations[human_mask, neuron_idx]

        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(ai_acts, human_acts, alternative="two-sided")

        # AUC as effect size
        y_true = np.concatenate([np.ones(len(ai_acts)), np.zeros(len(human_acts))])
        y_scores = np.concatenate([ai_acts, human_acts])

        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = 0.5  # fallback if all values are identical

        ai_median = np.median(ai_acts)
        human_median = np.median(human_acts)

        results.append(
            {
                "neuron_idx": neuron_idx,
                "ai_median": ai_median,
                "ai_q25": np.percentile(ai_acts, 25),
                "ai_q75": np.percentile(ai_acts, 75),
                "human_median": human_median,
                "human_q25": np.percentile(human_acts, 25),
                "human_q75": np.percentile(human_acts, 75),
                "u_statistic": u_stat,
                "p_value": p_value,
                "auc": auc,
                "auc_deviation": abs(auc - 0.5),
                "direction": "AI-preferring" if auc > 0.5 else "Human-preferring",
            }
        )

    return pd.DataFrame(results)


def identify_discriminative_neurons(
    stats_df: pd.DataFrame, alpha: float = 0.001, auc_threshold: float = 0.7
):
    """
    Identify discriminative neurons based on p-value and AUC thresholds.
    Applies Bonferroni correction for multiple testing.

    Args:
        stats_df: DataFrame with neuron statistics
        alpha: Significance level (before Bonferroni correction)
        auc_threshold: AUC threshold (neurons with AUC > threshold or < 1-threshold
                       are considered discriminative)
    """
    n_tests = len(stats_df)
    corrected_alpha = alpha / n_tests  # Bonferroni correction

    stats_df["significant"] = stats_df["p_value"] < corrected_alpha
    stats_df["strong_effect"] = (stats_df["auc"] > auc_threshold) | (
        stats_df["auc"] < (1 - auc_threshold)
    )
    stats_df["discriminative"] = stats_df["significant"] & stats_df["strong_effect"]

    print(f"\nBonferroni corrected alpha = {corrected_alpha:.2e}")
    print(f"AUC threshold: >{auc_threshold} or <{1 - auc_threshold}")

    return stats_df, corrected_alpha


def analyze_layer(
    layer_idx: int,
    activations: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.001,
    auc_threshold: float = 0.7,
):
    """Compute statistics for a single layer and print a summary."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing Layer {layer_idx}")
    print(f"{'=' * 60}")

    stats_df = compute_neuron_statistics(activations, labels)
    stats_df, corrected_alpha = identify_discriminative_neurons(stats_df, alpha, auc_threshold)

    n_disc = stats_df["discriminative"].sum()
    n_ai_pref = (
        (stats_df["discriminative"]) & (stats_df["direction"] == "AI-preferring")
    ).sum()
    n_human_pref = (
        (stats_df["discriminative"]) & (stats_df["direction"] == "Human-preferring")
    ).sum()

    print(f"\nResults:")
    print(f"  Total neurons: {len(stats_df)}")
    print(f"  Discriminative: {n_disc} ({n_disc / len(stats_df) * 100:.1f}%)")
    print(f"    AI-preferring:    {n_ai_pref}")
    print(f"    Human-preferring: {n_human_pref}")
    print(
        f"  Median AUC (discriminative): {stats_df[stats_df['discriminative']]['auc'].median():.3f}"
    )
    print(f"  Max AUC: {stats_df['auc'].max():.3f}")
    print(f"  Min AUC: {stats_df['auc'].min():.3f}")

    top_neurons = stats_df.nlargest(5, "auc_deviation")
    print(f"\nTop 5 discriminative neurons:")
    for _, row in top_neurons.iterrows():
        print(
            f"  Neuron {int(row['neuron_idx'])}: AUC={row['auc']:.3f}, "
            f"p={row['p_value']:.2e}, {row['direction']}"
        )

    return stats_df


def log_overall_summary(all_stats: dict):
    """Print overall summary across all layers."""
    layers = sorted(all_stats.keys())

    total_neurons = sum(len(all_stats[l]) for l in layers)
    total_disc = sum(all_stats[l]["discriminative"].sum() for l in layers)

    layer_disc_counts = {l: all_stats[l]["discriminative"].sum() for l in layers}
    most_disc_layer = max(layer_disc_counts, key=layer_disc_counts.get)

    early = [l for l in layers if l <= 4]
    middle = [l for l in layers if 5 <= l <= 8]
    late = [l for l in layers if l >= 9]

    early_disc = sum(all_stats[l]["discriminative"].sum() for l in early)
    middle_disc = sum(all_stats[l]["discriminative"].sum() for l in middle)
    late_disc = sum(all_stats[l]["discriminative"].sum() for l in late)

    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total neurons analyzed:       {total_neurons:,}")
    print(f"Discriminative neurons:       {total_disc:,} ({total_disc / total_neurons * 100:.1f}%)")
    print(f"  Early layers  (1–4):        {early_disc}")
    print(f"  Middle layers (5–8):        {middle_disc}")
    print(f"  Late layers   (9–12):       {late_disc}")
    print(f"Most discriminative layer:    Layer {most_disc_layer} ({layer_disc_counts[most_disc_layer]} neurons)")


def main():
    parser = argparse.ArgumentParser(
        description="Identify discriminative BERT neurons via Mann-Whitney U + AUC"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/activations_raid",
        help="Directory containing extracted activations (default: results/activations_raid)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Significance level before Bonferroni correction (default: 0.001)",
    )
    parser.add_argument(
        "--auc-threshold",
        type=float,
        default=0.7,
        help="AUC discrimination threshold (default: 0.7, i.e. >0.7 or <0.3)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Optional: Weights & Biases project name for experiment tracking",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional: Weights & Biases entity (username/team)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="activation_analysis",
        help="Optional: Weights & Biases run name",
    )

    args = parser.parse_args()

    # Optional wandb initialization
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config={
                    "alpha": args.alpha,
                    "auc_threshold": args.auc_threshold,
                    "test": "mann_whitney_u",
                    "effect_size": "auc",
                },
            )
            print(f"Weights & Biases logging enabled: {args.wandb_project}/{args.wandb_run_name}")
        except ImportError:
            print("⚠ wandb not installed — continuing without experiment tracking.")

    print("=" * 80)
    print("BERT ACTIVATION ANALYSIS  (Mann-Whitney U + AUC)")
    print("=" * 80)
    print(f"Input:         {args.input}")
    print(f"Alpha:         {args.alpha}  (Bonferroni-corrected per layer)")
    print(f"AUC threshold: >{args.auc_threshold} or <{1 - args.auc_threshold}")

    # Load activations
    activations, labels, metadata = load_activations(args.input)

    # Analyze each layer and save per-layer CSV
    all_stats = {}
    output_path = Path(args.input)

    for layer_idx in metadata["layers"]:
        stats_df = analyze_layer(
            layer_idx=layer_idx,
            activations=activations[layer_idx],
            labels=labels,
            alpha=args.alpha,
            auc_threshold=args.auc_threshold,
        )
        all_stats[layer_idx] = stats_df

        csv_path = output_path / f"layer_{layer_idx}_neuron_stats.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Overall summary
    log_overall_summary(all_stats)

    # Log summary metrics to wandb if enabled
    if wandb_run:
        layers = sorted(all_stats.keys())
        total_neurons = sum(len(all_stats[l]) for l in layers)
        total_disc = int(sum(all_stats[l]["discriminative"].sum() for l in layers))
        wandb_run.log(
            {
                "total_neurons": total_neurons,
                "total_discriminative": total_disc,
                "pct_discriminative": total_disc / total_neurons * 100,
            }
        )
        wandb_run.finish()

    print("\n" + "=" * 80)
    print("Analysis complete.")
    print(f"  CSV files saved to: {args.input}/")
    print("  Next steps: open notebooks/neurons_analysis.ipynb and neuron_clustering.ipynb")
    print("=" * 80)


if __name__ == "__main__":
    main()
