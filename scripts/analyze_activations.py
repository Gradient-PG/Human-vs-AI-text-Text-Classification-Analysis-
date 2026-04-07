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
from pathlib import Path

from raid_analysis.constants import N_BERT_NEURONS
from raid_analysis.data.activations import load_activations
from raid_analysis.data.neuron_stats import (
    compute_neuron_statistics,
    identify_discriminative_neurons,
)


def analyze_layer(
    layer_idx: int,
    activations,
    labels,
    alpha: float = 0.001,
    auc_threshold: float = 0.7,
):
    """Compute statistics for a single layer and print a summary."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing Layer {layer_idx}")
    print(f"{'=' * 60}")

    stats_df = compute_neuron_statistics(activations, labels)
    stats_df, corrected_alpha = identify_discriminative_neurons(
        stats_df, alpha, auc_threshold, n_total_tests=N_BERT_NEURONS
    )

    n_disc = stats_df["discriminative"].sum()
    n_ai_pref = (
        (stats_df["discriminative"]) & (stats_df["direction"] == "AI-preferring")
    ).sum()
    n_human_pref = (
        (stats_df["discriminative"]) & (stats_df["direction"] == "Human-preferring")
    ).sum()

    print(f"\nBonferroni corrected alpha = {corrected_alpha:.2e}")
    print(f"AUC threshold: >{auc_threshold} or <{1 - auc_threshold}")
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
            print("wandb not installed — continuing without experiment tracking.")

    print("=" * 80)
    print("BERT ACTIVATION ANALYSIS  (Mann-Whitney U + AUC)")
    print("=" * 80)
    print(f"Input:         {args.input}")
    print(f"Alpha:         {args.alpha}  (Bonferroni-corrected per layer)")
    print(f"AUC threshold: >{args.auc_threshold} or <{1 - args.auc_threshold}")

    activations, labels, metadata = load_activations(args.input)

    print(f"Loading activations from {args.input}")
    print(f"  Layers: {metadata['layers']}")
    print(f"  Samples: {metadata['n_samples']} ({metadata['n_ai']} AI, {metadata['n_human']} Human)")

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

    log_overall_summary(all_stats)

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
    print("=" * 80)


if __name__ == "__main__":
    main()
