#!/usr/bin/env python
"""
Analyze BERT layer activations to identify discriminative neurons.
Uses Mann-Whitney U test and AUC for effect size with mandatory wandb logging.

Usage:
    python scripts/analyze_activations.py --wandb-project human-vs-ai
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
    
    with open(input_path / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"Loading activations from {input_path}")
    print(f"  Layers: {metadata['layers']}")
    print(f"  Samples: {metadata['n_samples']} ({metadata['n_ai']} AI, {metadata['n_human']} Human)")
    
    activations = {}
    for layer in metadata['layers']:
        layer_file = input_path / f"layer_{layer}_activations.npy"
        activations[layer] = np.load(layer_file)
        print(f"  Loaded layer {layer}: {activations[layer].shape}")
    
    labels = np.load(input_path / "labels.npy")
    
    return activations, labels, metadata


def compute_neuron_statistics(activations: np.ndarray, labels: np.ndarray):
    """
    Compute per-neuron Mann-Whitney U test and AUC.
    
    Args:
        activations: (N_samples, N_neurons) array
        labels: (N_samples,) array of 0/1 labels
    
    Returns:
        DataFrame with per-neuron statistics
    """
    ai_mask = labels == 1
    human_mask = labels == 0
    
    n_neurons = activations.shape[1]
    n_ai = ai_mask.sum()
    n_human = human_mask.sum()
    
    results = []
    
    for neuron_idx in tqdm(range(n_neurons), desc="Testing neurons"):
        ai_acts = activations[ai_mask, neuron_idx]
        human_acts = activations[human_mask, neuron_idx]
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value = stats.mannwhitneyu(ai_acts, human_acts, alternative='two-sided')
        
        # AUC as effect size
        # Create binary labels and activations for AUC calculation
        y_true = np.concatenate([np.ones(len(ai_acts)), np.zeros(len(human_acts))])
        y_scores = np.concatenate([ai_acts, human_acts])
        
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.5  # If all values are the same
        
        # Determine direction
        ai_median = np.median(ai_acts)
        human_median = np.median(human_acts)
        
        results.append({
            'neuron_idx': neuron_idx,
            'ai_median': ai_median,
            'ai_q25': np.percentile(ai_acts, 25),
            'ai_q75': np.percentile(ai_acts, 75),
            'human_median': human_median,
            'human_q25': np.percentile(human_acts, 25),
            'human_q75': np.percentile(human_acts, 75),
            'u_statistic': u_stat,
            'p_value': p_value,
            'auc': auc,
            'auc_deviation': abs(auc - 0.5),  # Distance from no-effect baseline
            'direction': 'AI-preferring' if auc > 0.5 else 'Human-preferring'
        })
    
    return pd.DataFrame(results)


def identify_discriminative_neurons(stats_df: pd.DataFrame, alpha: float = 0.001, auc_threshold: float = 0.7):
    """
    Identify discriminative neurons based on p-value and AUC thresholds.
    
    Args:
        stats_df: DataFrame with neuron statistics
        alpha: Significance threshold (after Bonferroni correction)
        auc_threshold: AUC threshold (>0.7 or <0.3 for discrimination)
    """
    n_tests = len(stats_df)
    corrected_alpha = alpha / n_tests
    
    # Discriminative: significant AND strong effect
    stats_df['significant'] = stats_df['p_value'] < corrected_alpha
    stats_df['strong_effect'] = (stats_df['auc'] > auc_threshold) | (stats_df['auc'] < (1 - auc_threshold))
    stats_df['discriminative'] = stats_df['significant'] & stats_df['strong_effect']
    
    print(f"\nBonferroni corrected α = {corrected_alpha:.2e}")
    print(f"AUC threshold: >{auc_threshold} or <{1-auc_threshold}")
    
    return stats_df, corrected_alpha


def analyze_layer(layer_idx: int, activations: np.ndarray, labels: np.ndarray, 
                 alpha: float = 0.001, auc_threshold: float = 0.7):
    """Analyze single layer."""
    print(f"\n{'='*60}")
    print(f"Analyzing Layer {layer_idx}")
    print(f"{'='*60}")
    
    # Compute statistics
    stats_df = compute_neuron_statistics(activations, labels)
    
    # Identify discriminative neurons
    stats_df, corrected_alpha = identify_discriminative_neurons(stats_df, alpha, auc_threshold)
    
    # Summary
    n_disc = stats_df['discriminative'].sum()
    n_ai_pref = ((stats_df['discriminative']) & (stats_df['direction'] == 'AI-preferring')).sum()
    n_human_pref = ((stats_df['discriminative']) & (stats_df['direction'] == 'Human-preferring')).sum()
    
    print(f"\nResults:")
    print(f"  Total neurons: {len(stats_df)}")
    print(f"  Discriminative: {n_disc} ({n_disc/len(stats_df)*100:.1f}%)")
    print(f"    AI-preferring: {n_ai_pref}")
    print(f"    Human-preferring: {n_human_pref}")
    print(f"  Median AUC (discriminative): {stats_df[stats_df['discriminative']]['auc'].median():.3f}")
    print(f"  Max AUC: {stats_df['auc'].max():.3f}")
    print(f"  Min AUC: {stats_df['auc'].min():.3f}")
    
    # Top neurons
    top_neurons = stats_df.nlargest(5, 'auc_deviation')
    print(f"\nTop 5 discriminative neurons:")
    for _, row in top_neurons.iterrows():
        print(f"  Neuron {int(row['neuron_idx'])}: AUC={row['auc']:.3f}, p={row['p_value']:.2e}, {row['direction']}")
    
    return stats_df


def create_wandb_visualizations(all_stats: dict, activations: dict, labels: np.ndarray, wandb_run):
    """Create and log visualizations to wandb."""
    import wandb
    
    layers = sorted(all_stats.keys())
    
    print(f"\n{'='*60}")
    print("Creating Visualizations for wandb")
    print(f"{'='*60}")
    
    # Figure 1: Layer-wise distribution (bar chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ai_pref = []
    human_pref = []
    for layer in layers:
        stats = all_stats[layer]
        disc = stats[stats['discriminative']]
        ai_pref.append(((disc['direction'] == 'AI-preferring').sum()))
        human_pref.append(((disc['direction'] == 'Human-preferring').sum()))
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, ai_pref, width, label='AI-preferring', color='#ff6b6b', alpha=0.8)
    ax.bar(x + width/2, human_pref, width, label='Human-preferring', color='#4ecdc4', alpha=0.8)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Count of Discriminative Neurons', fontsize=12)
    ax.set_title('Discriminative Neurons by Layer and Direction', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    wandb_run.log({"layer_wise_distribution": wandb.Image(fig)})
    plt.close()
    
    # Figure 2: AUC distributions (violin plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    auc_data = [all_stats[layer]['auc'].values for layer in layers]
    parts = ax.violinplot(auc_data, positions=range(len(layers)), showmedians=True, widths=0.7)
    
    # Color violins
    for pc in parts['bodies']:
        pc.set_facecolor('#9b59b6')
        pc.set_alpha(0.7)
    
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='No effect (AUC=0.5)')
    ax.axhline(0.7, color='orange', linestyle=':', alpha=0.5, label='Discrimination threshold')
    ax.axhline(0.3, color='orange', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC Distribution Across Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    wandb_run.log({"auc_distributions": wandb.Image(fig)})
    plt.close()
    
    # Figure 3: Top-10 neurons heatmap
    # Get top 10 neurons across all layers
    all_neurons_list = []
    for layer, stats in all_stats.items():
        stats_copy = stats.copy()
        stats_copy['layer'] = layer
        all_neurons_list.append(stats_copy)
    
    all_neurons_df = pd.concat(all_neurons_list)
    top10 = all_neurons_df.nlargest(10, 'auc_deviation')
    
    # Sample 100 samples (50 AI, 50 Human)
    ai_indices = np.where(labels == 1)[0][:50]
    human_indices = np.where(labels == 0)[0][:50]
    sample_indices = np.concatenate([ai_indices, human_indices])
    
    # Build heatmap data
    heatmap_data = []
    heatmap_labels = []
    
    for _, neuron in top10.iterrows():
        layer = int(neuron['layer'])
        neuron_idx = int(neuron['neuron_idx'])
        
        # Get activations for this neuron
        neuron_acts = activations[layer][sample_indices, neuron_idx]
        heatmap_data.append(neuron_acts)
        heatmap_labels.append(f"L{layer}_N{neuron_idx}\n(AUC={neuron['auc']:.3f})")
    
    heatmap_array = np.array(heatmap_data)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap_array, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    
    ax.set_xlabel('Samples (50 AI | 50 Human)', fontsize=12)
    ax.set_ylabel('Top 10 Neurons', fontsize=12)
    ax.set_title('Activation Heatmap: Top 10 Discriminative Neurons', fontsize=14, fontweight='bold')
    ax.set_yticks(range(10))
    ax.set_yticklabels(heatmap_labels, fontsize=9)
    
    # Add vertical line separating AI/Human
    ax.axvline(50, color='white', linewidth=2, linestyle='--')
    ax.text(25, -0.5, 'AI', ha='center', fontsize=10, fontweight='bold')
    ax.text(75, -0.5, 'Human', ha='center', fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Activation Intensity')
    plt.tight_layout()
    
    wandb_run.log({"top_neurons_heatmap": wandb.Image(fig)})
    plt.close()
    
    # Figure 4: Top 5 neurons box plots
    top5 = all_neurons_df.nlargest(5, 'auc_deviation')
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    
    for idx, (_, neuron) in enumerate(top5.iterrows()):
        layer = int(neuron['layer'])
        neuron_idx = int(neuron['neuron_idx'])
        
        ai_acts = activations[layer][labels == 1, neuron_idx]
        human_acts = activations[layer][labels == 0, neuron_idx]
        
        ax = axes[idx]
        bp = ax.boxplot([ai_acts, human_acts], labels=['AI', 'Human'],
                        patch_artist=True, widths=0.6)
        
        # Color boxes
        bp['boxes'][0].set_facecolor('#ff6b6b')
        bp['boxes'][1].set_facecolor('#4ecdc4')
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black')
        
        ax.set_title(f"Layer {layer}, Neuron {neuron_idx}\nAUC={neuron['auc']:.3f}", 
                    fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('Activation Value', fontsize=11)
    
    fig.suptitle('Top 5 Discriminative Neurons: Activation Distributions', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    wandb_run.log({"top5_boxplots": wandb.Image(fig)})
    plt.close()
    
    print("✓ All visualizations logged to wandb")


def create_wandb_tables(all_stats: dict, wandb_run):
    """Create and log summary tables to wandb."""
    import wandb
    
    layers = sorted(all_stats.keys())
    
    # Table 1: Per-layer summary
    layer_summary = []
    for layer in layers:
        stats = all_stats[layer]
        disc = stats[stats['discriminative']]
        
        layer_summary.append([
            f"Layer {layer}",
            len(stats),
            len(disc),
            f"{len(disc)/len(stats)*100:.1f}%",
            ((disc['direction'] == 'AI-preferring').sum()),
            ((disc['direction'] == 'Human-preferring').sum()),
            f"{disc['auc'].median():.3f}" if len(disc) > 0 else "N/A",
            f"{stats['auc'].max():.3f}",
            f"{stats['auc'].min():.3f}"
        ])
    
    wandb_run.log({"layer_summary": wandb.Table(
        columns=["Layer", "Total Neurons", "Discriminative", "% Discriminative", 
                "AI-preferring", "Human-preferring", "Median AUC", "Max AUC", "Min AUC"],
        data=layer_summary
    )})
    
    # Table 2: Top 20 neurons across all layers
    all_neurons_list = []
    for layer, stats in all_stats.items():
        stats_copy = stats.copy()
        stats_copy['layer'] = layer
        all_neurons_list.append(stats_copy)
    
    all_neurons_df = pd.concat(all_neurons_list)
    top20 = all_neurons_df.nlargest(20, 'auc_deviation')
    
    top_neurons_data = []
    for _, row in top20.iterrows():
        top_neurons_data.append([
            f"Layer {int(row['layer'])}",
            int(row['neuron_idx']),
            f"{row['auc']:.4f}",
            f"{row['p_value']:.2e}",
            row['direction'],
            "Yes" if row['discriminative'] else "No"
        ])
    
    wandb_run.log({"top_20_neurons": wandb.Table(
        columns=["Layer", "Neuron ID", "AUC", "p-value", "Direction", "Discriminative"],
        data=top_neurons_data
    )})
    
    print("✓ All tables logged to wandb")


def log_overall_metrics(all_stats: dict, wandb_run):
    """Log overall summary metrics to wandb."""
    layers = sorted(all_stats.keys())
    
    # Overall metrics
    total_neurons = sum(len(all_stats[l]) for l in layers)
    total_disc = sum(all_stats[l]['discriminative'].sum() for l in layers)
    
    # Find most discriminative layer
    layer_disc_counts = {l: all_stats[l]['discriminative'].sum() for l in layers}
    most_disc_layer = max(layer_disc_counts, key=layer_disc_counts.get)
    
    # Early vs late layers
    early_layers = [l for l in layers if l <= 6]
    late_layers = [l for l in layers if l > 6]
    
    early_disc = sum(all_stats[l]['discriminative'].sum() for l in early_layers)
    late_disc = sum(all_stats[l]['discriminative'].sum() for l in late_layers)
    
    metrics = {
        "total_neurons_analyzed": total_neurons,
        "total_discriminative": int(total_disc),
        "pct_discriminative": total_disc / total_neurons * 100,
        "most_discriminative_layer": most_disc_layer,
        "most_discriminative_layer_count": int(layer_disc_counts[most_disc_layer]),
        "early_layers_discriminative": int(early_disc),
        "late_layers_discriminative": int(late_disc),
        "interpretation": "semantic" if late_disc > early_disc else "syntactic"
    }
    
    # Per-layer metrics
    for layer in layers:
        stats = all_stats[layer]
        disc = stats[stats['discriminative']]
        metrics[f"layer_{layer}_discriminative"] = int(len(disc))
        metrics[f"layer_{layer}_pct"] = len(disc) / len(stats) * 100
        metrics[f"layer_{layer}_max_auc"] = float(stats['auc'].max())
    
    wandb_run.log(metrics)
    
    print(f"\n{'='*60}")
    print("Overall Metrics")
    print(f"{'='*60}")
    print(f"Total discriminative neurons: {total_disc} / {total_neurons} ({metrics['pct_discriminative']:.1f}%)")
    print(f"Most discriminative layer: Layer {most_disc_layer} ({layer_disc_counts[most_disc_layer]} neurons)")
    print(f"Early layers (≤6): {early_disc} discriminative")
    print(f"Late layers (>6): {late_disc} discriminative")
    print(f"→ Interpretation: {'Semantic' if late_disc > early_disc else 'Syntactic'} features dominate")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BERT activations with Mann-Whitney U and AUC"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/activations",
        help="Input directory with extracted activations",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        required=True,
        help="Weights & Biases project name (REQUIRED)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (username/team)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="activation_analysis",
        help="Weights & Biases run name",
    )
    
    args = parser.parse_args()
    
    # Fixed analysis parameters
    ALPHA = 0.001  # Significance level (before Bonferroni)
    AUC_THRESHOLD = 0.7  # AUC > 0.7 or < 0.3 for discrimination
    
    # Initialize wandb (mandatory)
    try:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "alpha": ALPHA,
                "auc_threshold": AUC_THRESHOLD,
                "test": "mann_whitney_u",
                "effect_size": "auc"
            }
        )
        print(f"✓ Weights & Biases initialized: {args.wandb_project}/{args.wandb_run_name}")
    except ImportError:
        print("ERROR: wandb not installed. Install with: pip install wandb")
        print("wandb is required for this analysis.")
        sys.exit(1)
    
    print("=" * 80)
    print("BERT ACTIVATION ANALYSIS (Mann-Whitney U + AUC)")
    print("=" * 80)
    
    # Load activations
    activations, labels, metadata = load_activations(args.input)
    
    # Analyze each layer
    all_stats = {}
    for layer_idx in metadata['layers']:
        stats_df = analyze_layer(
            layer_idx=layer_idx,
            activations=activations[layer_idx],
            labels=labels,
            alpha=ALPHA,
            auc_threshold=AUC_THRESHOLD
        )
        all_stats[layer_idx] = stats_df
        
        # Save per-layer results
        output_path = Path(args.input)
        csv_path = output_path / f"layer_{layer_idx}_neuron_stats.csv"
        stats_df.to_csv(csv_path, index=False)
        print(f"  Saved statistics: {csv_path}")
    
    # Create visualizations and log to wandb
    create_wandb_visualizations(all_stats, activations, labels, wandb_run)
    
    # Create tables and log to wandb
    create_wandb_tables(all_stats, wandb_run)
    
    # Log overall metrics
    log_overall_metrics(all_stats, wandb_run)
    
    wandb_run.finish()
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - CSV files: {args.input}/layer_X_neuron_stats.csv")
    print(f"  - Visualizations and tables: https://wandb.ai/{args.wandb_project}")


if __name__ == "__main__":
    main()
