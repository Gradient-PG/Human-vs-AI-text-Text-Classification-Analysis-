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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import json
import wandb


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
    Identify discriminative neurons based on p-value and AUC thresholds. Performs Bonferroni correction for multiple testing.
    
    Args:
        stats_df: DataFrame with neuron statistics
        alpha: Significance threshold (after Bonferroni correction)
        auc_threshold: AUC threshold (>0.7 or <0.3 for discrimination)
    """
    n_tests = len(stats_df)
    corrected_alpha = alpha / n_tests # Bonferroni correction
    
    # Discriminative: significant AND strong effect
    stats_df['significant'] = stats_df['p_value'] < corrected_alpha
    stats_df['strong_effect'] = (stats_df['auc'] > auc_threshold) | (stats_df['auc'] < (1 - auc_threshold))
    stats_df['discriminative'] = stats_df['significant'] & stats_df['strong_effect']
    
    print(f"\nBonferroni corrected alpha = {corrected_alpha:.2e}")
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


def plot_all_neurons_auc(all_stats: dict, wandb_run):
    """
    Plot AUC distribution for ALL analyzed neurons (not just discriminative).
    Shows clear discrimination thresholds at 0.3 and 0.7.
    """
    print(f"\n{'='*60}")
    print("Creating All Neurons AUC Plot")
    print(f"{'='*60}")
    
    layers = sorted(all_stats.keys())
    
    # Collect all AUC values across all layers
    all_aucs = []
    layer_labels = []
    
    for layer in layers:
        stats = all_stats[layer]
        aucs = stats['auc'].values
        all_aucs.extend(aucs)
        layer_labels.extend([layer] * len(aucs))
    
    all_aucs = np.array(all_aucs)
    layer_labels = np.array(layer_labels)
    
    # Count neurons in different categories
    total_neurons = len(all_aucs)
    discriminative_count = sum(all_stats[l]['discriminative'].sum() for l in layers)
    ai_specialist_count = sum((all_aucs > 0.7).sum() for _ in [1])
    human_specialist_count = sum((all_aucs < 0.3).sum() for _ in [1])
    neutral_count = total_neurons - ai_specialist_count - human_specialist_count
    
    print(f"  Total neurons: {total_neurons}")
    print(f"  Discriminative neurons: {discriminative_count} ({discriminative_count/total_neurons*100:.1f}%)")
    print(f"  AI-specialists (AUC > 0.7): {ai_specialist_count} ({ai_specialist_count/total_neurons*100:.1f}%)")
    print(f"  Human-specialists (AUC < 0.3): {human_specialist_count} ({human_specialist_count/total_neurons*100:.1f}%)")
    print(f"  Neutral (0.3 ≤ AUC ≤ 0.7): {neutral_count} ({neutral_count/total_neurons*100:.1f}%)")
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # ============================================================
    # Plot 1: Scatter plot of all neurons by layer
    # ============================================================
    ax1 = fig.add_subplot(gs[0])
    
    # Create color map based on AUC value
    colors_map = np.zeros((len(all_aucs), 3))
    for i, auc in enumerate(all_aucs):
        if auc > 0.7:
            # AI-preferring: red gradient
            intensity = (auc - 0.7) / 0.3  # 0.7 to 1.0
            colors_map[i] = [1.0, 0.4 - 0.4*intensity, 0.4 - 0.4*intensity]  # Red
        elif auc < 0.3:
            # Human-preferring: cyan gradient
            intensity = (0.3 - auc) / 0.3  # 0.0 to 0.3
            colors_map[i] = [0.3 - 0.3*intensity, 0.8, 0.8]  # Cyan
        else:
            # Neutral: gray
            colors_map[i] = [0.7, 0.7, 0.7]  # Gray
    
    # Add jitter to x-axis for better visibility
    jitter = np.random.normal(0, 0.15, len(layer_labels))
    x_positions = layer_labels + jitter
    
    # Plot neurons
    scatter = ax1.scatter(x_positions, all_aucs, c=colors_map, alpha=0.6, s=20, 
                         edgecolors='none', rasterized=True)
    
    # Add threshold lines
    ax1.axhline(0.5, color='black', linestyle='-', linewidth=2.5, alpha=0.8, 
                label='No effect (AUC = 0.5)', zorder=10)
    ax1.axhline(0.7, color='#ff6b6b', linestyle='--', linewidth=2.5, alpha=0.9, 
                label='AI discrimination threshold (AUC = 0.7)', zorder=10)
    ax1.axhline(0.3, color='#4ecdc4', linestyle='--', linewidth=2.5, alpha=0.9, 
                label='Human discrimination threshold (AUC = 0.3)', zorder=10)
    
    # Shade discrimination regions
    ax1.fill_between([0, len(layers)+1], 0.7, 1.0, alpha=0.1, color='#ff6b6b', 
                     label='AI-specialist region')
    ax1.fill_between([0, len(layers)+1], 0.0, 0.3, alpha=0.1, color='#4ecdc4', 
                     label='Human-specialist region')
    
    ax1.set_xlabel('BERT Layer', fontsize=14, fontweight='bold')
    ax1.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
    ax1.set_title(f'AUC Distribution: All {total_neurons} Analyzed Neurons', 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(layers)
    ax1.set_xticklabels([f'L{l}' for l in layers])
    ax1.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
    
    # ============================================================
    # Plot 2: Overall histogram
    # ============================================================
    ax2 = fig.add_subplot(gs[1])
    
    # Create histogram with color-coded bars
    bins = np.linspace(0, 1, 41)  # 40 bins
    counts, bin_edges = np.histogram(all_aucs, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Color each bar based on its position
    bar_colors = []
    for bc in bin_centers:
        if bc > 0.7:
            bar_colors.append('#ff6b6b')  # Red for AI
        elif bc < 0.3:
            bar_colors.append('#4ecdc4')  # Cyan for Human
        else:
            bar_colors.append('#999999')  # Gray for neutral
    
    ax2.bar(bin_centers, counts, width=0.024, color=bar_colors, alpha=0.8, 
            edgecolor='black', linewidth=0.5)
    
    # Add threshold lines
    ax2.axvline(0.5, color='black', linestyle='-', linewidth=2.5, alpha=0.8, zorder=10)
    ax2.axvline(0.7, color='#ff6b6b', linestyle='--', linewidth=2.5, alpha=0.9, zorder=10)
    ax2.axvline(0.3, color='#4ecdc4', linestyle='--', linewidth=2.5, alpha=0.9, zorder=10)
    
    # Add text annotations
    max_count = counts.max()
    ax2.text(0.85, max_count * 0.95, f'AI\n{ai_specialist_count}', 
             ha='center', va='top', fontsize=11, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='#ff6b6b', alpha=0.3))
    ax2.text(0.15, max_count * 0.95, f'Human\n{human_specialist_count}', 
             ha='center', va='top', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#4ecdc4', alpha=0.3))
    ax2.text(0.5, max_count * 0.95, f'Neutral\n{neutral_count}', 
             ha='center', va='top', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#999999', alpha=0.3))
    
    ax2.set_xlabel('AUC Score', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Neuron Count', fontsize=14, fontweight='bold')
    ax2.set_title('Overall Distribution', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlim(-0.02, 1.02)
    ax2.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)
    
    plt.suptitle('Complete Neuron Analysis: AUC Scores Across All Layers', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Log to wandb
    wandb_run.log({"all_neurons_auc_distribution": wandb.Image(fig)})
    plt.close()
    
    print(f"  All neurons AUC plot logged to wandb")


def create_wandb_visualizations(all_stats: dict, activations: dict, labels: np.ndarray, wandb_run):
    """Create and log visualizations to wandb."""
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
    ax.set_title('Discriminative Neurons by Layer and Direction', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(pad=1.5)
    
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
    ax.set_title('AUC Distribution Across Layers', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout(pad=1.5)
    
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
    
    # Center colormap around 0
    vmax = np.abs(heatmap_array).max()
    vmin = -vmax
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap_array, aspect='auto', cmap='RdYlBu_r', interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    
    ax.set_xlabel('Samples (50 AI | 50 Human)', fontsize=12, labelpad=15)
    ax.set_ylabel('Top 10 Neurons', fontsize=12)
    ax.set_title('Activation Heatmap: Top 10 Discriminative Neurons', fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(range(10))
    ax.set_yticklabels(heatmap_labels, fontsize=9)
    
    # Add vertical line separating AI/Human
    ax.axvline(49.5, color='white', linewidth=2, linestyle='--')
    ax.text(25, -1.2, 'AI', ha='center', fontsize=11, fontweight='bold')
    ax.text(75, -1.2, 'Human', ha='center', fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Activation Intensity')
    plt.tight_layout()
    
    wandb_run.log({"top_neurons_heatmap": wandb.Image(fig)})
    plt.close()
    
    # Figure 4: Top 5 neurons box plots
    top5 = all_neurons_df.nlargest(5, 'auc_deviation')
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)
    
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
                    fontsize=10, pad=10)
        ax.grid(axis='y', alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('Activation Value', fontsize=11)
    
    fig.suptitle('Top 5 Discriminative Neurons: Activation Distributions', 
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    wandb_run.log({"top5_boxplots": wandb.Image(fig)})
    plt.close()
    
    print("All visualizations logged to wandb")


def create_wandb_tables(all_stats: dict, wandb_run):
    """Create and log summary tables to wandb."""
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
    
    print("All tables logged to wandb")


def _extract_neuron_features(all_stats: dict, activations: dict):
    """Extract feature vectors and metadata for discriminative neurons."""
    discriminative_neurons = []
    neuron_metadata = []
    
    for layer in sorted(activations.keys()):
        stats = all_stats[layer]
        disc_mask = stats['discriminative'].values
        disc_indices = np.where(disc_mask)[0]
        
        for neuron_idx in disc_indices:
            neuron_stats = stats.iloc[neuron_idx]
            
            # Feature vector: 6 activation statistics
            features = [
                neuron_stats['ai_median'],
                neuron_stats['human_median'],
                neuron_stats['ai_q75'] - neuron_stats['ai_q25'],  # AI IQR
                neuron_stats['human_q75'] - neuron_stats['human_q25'],  # Human IQR
                neuron_stats['auc'],
                neuron_stats['auc_deviation']
            ]
            
            discriminative_neurons.append(features)
            neuron_metadata.append({
                'layer': layer,
                'neuron_idx': neuron_idx,
                'direction': neuron_stats['direction'],
                'auc': neuron_stats['auc']
            })
    
    return np.array(discriminative_neurons), neuron_metadata


def _perform_clustering(X: np.ndarray, n_clusters: int):
    """Standardize features and perform k-means clustering."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return X_scaled, cluster_labels, scaler, kmeans


def _characterize_clusters(cluster_labels: np.ndarray, neuron_metadata: list, n_clusters: int):
    """Analyze and characterize each cluster."""
    cluster_summary = []
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_neurons = [neuron_metadata[i] for i in range(len(neuron_metadata)) if mask[i]]
        
        # Compute characteristics
        ai_pref_count = sum(1 for n in cluster_neurons if n['direction'] == 'AI-preferring')
        human_pref_count = len(cluster_neurons) - ai_pref_count
        mean_auc = np.mean([n['auc'] for n in cluster_neurons])
        
        # Layer distribution
        layer_counts = {}
        for n in cluster_neurons:
            layer_counts[n['layer']] = layer_counts.get(n['layer'], 0) + 1
        dominant_layer = max(layer_counts, key=layer_counts.get) if layer_counts else None
        
        # Classify cluster type based on mean AUC
        if mean_auc > 0.7:
            cluster_type = "AI-specialists"
        elif mean_auc < 0.3:
            cluster_type = "Human-specialists"
        else:
            cluster_type = "Balanced discriminators"
        
        cluster_summary.append({
            'cluster_id': cluster_id,
            'size': len(cluster_neurons),
            'type': cluster_type,
            'ai_preferring': ai_pref_count,
            'human_preferring': human_pref_count,
            'mean_auc': mean_auc,
            'dominant_layer': dominant_layer,
            'layer_counts': layer_counts
        })
        
        # Print summary
        print(f"\nCluster {cluster_id}: {cluster_type}")
        print(f"  Size: {len(cluster_neurons)} neurons")
        print(f"  AI-preferring: {ai_pref_count}, Human-preferring: {human_pref_count}")
        print(f"  Mean AUC: {mean_auc:.3f}")
        print(f"  Dominant layer: {dominant_layer}")
        print(f"  Layer distribution: {dict(sorted(layer_counts.items()))}")
    
    return cluster_summary


def _get_cluster_colors(n_clusters: int):
    """Generate enough colors for all clusters."""
    base_colors = [
        "#1f77b4",  # blue (high contrast, primary)
        "#d62728",  # red
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#17becf",  # cyan
        "#7f7f7f",  # gray (neutral, for less important category)
    ]
    return (base_colors * ((n_clusters // len(base_colors)) + 1))[:n_clusters]


def _plot_cluster_overview(cluster_summary: list, cluster_labels: np.ndarray, neuron_metadata: list, 
                           n_clusters: int, colors: list, wandb_run):
    """Create overview plots: cluster sizes and AUC distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Cluster sizes
    ax = axes[0]
    cluster_ids = [c['cluster_id'] for c in cluster_summary]
    sizes = [c['size'] for c in cluster_summary]
    types = [c['type'] for c in cluster_summary]
    
    bars = ax.bar(cluster_ids, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, size, cluster_type in zip(bars, sizes, types):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(size)}\n({cluster_type})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Cluster ID', fontsize=13)
    ax.set_ylabel('Number of Neurons', fontsize=13)
    ax.set_title('Discriminative Neuron Clusters', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(cluster_ids)
    ax.set_xticklabels([f'C{i}' for i in cluster_ids])
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: AUC distributions
    ax = axes[1]
    
    # Use fixed bins from 0 to 1 with consistent width
    bin_edges = np.linspace(0, 1, 21)  # 20 bins of equal width (0.05 each)
    
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_aucs = [neuron_metadata[i]['auc'] for i in range(len(neuron_metadata)) if mask[i]]
        cluster_type = cluster_summary[cluster_id]['type']
        ax.hist(cluster_aucs, bins=bin_edges, alpha=0.7, 
                label=f'C{cluster_id}: {cluster_type}', 
                color=colors[cluster_id], edgecolor='black', linewidth=0.5)
    
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='No effect')
    ax.axvline(0.7, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Threshold')
    ax.axvline(0.3, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xlabel('AUC Score', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('AUC Distribution by Cluster', fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    wandb_run.log({"cluster_analysis": wandb.Image(fig)})
    plt.close()
    
    print(f"  Cluster overview logged to wandb")


def _plot_layer_clusters_distribution(cluster_summary: list, cluster_labels: np.ndarray, neuron_metadata: list,
                             activations: dict, n_clusters: int, colors: list, wandb_run):
    """Plot cluster distribution across BERT layers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(sorted(activations.keys())))
    width = 0.8 / n_clusters  # Adaptive width
    
    for idx, cluster_id in enumerate(range(n_clusters)):
        mask = cluster_labels == cluster_id
        cluster_neurons = [neuron_metadata[i] for i in range(len(neuron_metadata)) if mask[i]]
        
        layer_counts = {layer: 0 for layer in sorted(activations.keys())}
        for n in cluster_neurons:
            layer_counts[n['layer']] += 1
        
        counts = [layer_counts[layer] for layer in sorted(activations.keys())]
        offset = width * (idx - n_clusters/2 + 0.5)
        cluster_type = cluster_summary[cluster_id]['type']
        
        ax.bar(x + offset, counts, width, label=f'C{cluster_id}: {cluster_type}',
               color=colors[cluster_id], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('BERT Layer', fontsize=13)
    ax.set_ylabel('Number of Discriminative Neurons', fontsize=13)
    ax.set_title('Cluster Distribution Across BERT Layers', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {layer}' for layer in sorted(activations.keys())])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout(pad=1.5)
    wandb_run.log({"cluster_layer_distribution": wandb.Image(fig)})
    plt.close()
    
    print(f"  Cluster layer distribution logged to wandb")


def _plot_clusters_pca_2d(X_scaled: np.ndarray, cluster_labels: np.ndarray, neuron_metadata: list,
                 activations: dict, n_clusters: int, colors: list, wandb_run):
    """Create 2D PCA visualization of clusters per layer."""
    print(f"\n  Creating 2D cluster visualizations per layer...")
    
    # PCA for dimensionality reduction (6D to 2D)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    print(f"  PCA explained variance: {explained_var[0]:.1%} + {explained_var[1]:.1%} = {explained_var.sum():.1%}")
    
    # Create per-layer 2D scatter plots
    layers_in_data = sorted(activations.keys())
    n_layers = len(layers_in_data)
    
    fig, axes = plt.subplots(1, n_layers, figsize=(5.5*n_layers, 5.5), squeeze=False)
    axes = axes[0]  # Get first row
    
    for idx, layer in enumerate(layers_in_data):
        ax = axes[idx]
        layer_mask = np.array([meta['layer'] == layer for meta in neuron_metadata])
        
        if layer_mask.sum() == 0:
            ax.text(0.5, 0.5, f'No discriminative\nneurons in Layer {layer}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Layer {layer}', fontsize=13, fontweight='bold', pad=10)
            continue
        
        # Plot each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_id) & layer_mask
            
            if cluster_mask.sum() > 0:
                ax.scatter(X_2d[cluster_mask, 0], X_2d[cluster_mask, 1],
                          c=colors[cluster_id], label=f'C{cluster_id}',
                          alpha=0.7, s=80, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=11)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=11)
        ax.set_title(f'Layer {layer}\n({layer_mask.sum()} neurons)', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=8, loc='best', ncol=2)
        ax.grid(alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.suptitle(f'2D Cluster Visualization per Layer (PCA projection - explained variance: {explained_var.sum():.1%})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    wandb_run.log({"cluster_2d_per_layer": wandb.Image(fig)})
    plt.close()
    
    # Feature importance
    feature_names = ['AI_median', 'Human_median', 'AI_IQR', 'Human_IQR', 'AUC', 'AUC_dev']
    pc1_importance = np.abs(pca.components_[0])
    pc2_importance = np.abs(pca.components_[1])
    
    print(f"  Top PC1 feature: {feature_names[pc1_importance.argmax()]} ({pc1_importance.max():.2f})")
    print(f"  Top PC2 feature: {feature_names[pc2_importance.argmax()]} ({pc2_importance.max():.2f})")
    print(f"  2D cluster visualizations logged to wandb")


def cluster_analysis(activations: dict, labels: np.ndarray, all_stats: dict, wandb_run, n_clusters: int = 3):
    """
    Cluster discriminative neurons to identify functional groups.
    
    Args:
        activations: Dict of layer activations
        labels: Sample labels (0=Human, 1=AI)
        all_stats: Dict of neuron statistics per layer
        wandb_run: Wandb run object
        n_clusters: Number of clusters
    """
    print(f"\n{'='*60}")
    print(f"Clustering Analysis (k={n_clusters})")
    print(f"{'='*60}")
    
    # Step 1: Extract features
    X, neuron_metadata = _extract_neuron_features(all_stats, activations)
    
    if len(X) < n_clusters:
        print(f"  WARNING: Only {len(X)} discriminative neurons found, skipping clustering")
        return
    
    # Step 2: Perform clustering
    X_scaled, cluster_labels, scaler, kmeans = _perform_clustering(X, n_clusters)
    
    # Step 3: Characterize clusters
    cluster_summary = _characterize_clusters(cluster_labels, neuron_metadata, n_clusters)
    
    # Step 4: Generate colors
    colors = _get_cluster_colors(n_clusters)
    
    # Step 5: Create visualizations
    _plot_cluster_overview(cluster_summary, cluster_labels, neuron_metadata, n_clusters, colors, wandb_run)
    _plot_layer_clusters_distribution(cluster_summary, cluster_labels, neuron_metadata, activations, n_clusters, colors, wandb_run)
    _plot_clusters_pca_2d(X_scaled, cluster_labels, neuron_metadata, activations, n_clusters, colors, wandb_run)
    
    # Step 6: Log cluster summary table
    cluster_table_data = []
    for c in cluster_summary:
        cluster_table_data.append([
            f"Cluster {c['cluster_id']}",
            c['type'],
            c['size'],
            c['ai_preferring'],
            c['human_preferring'],
            f"{c['mean_auc']:.3f}",
            f"Layer {c['dominant_layer']}"
        ])
    
    wandb_run.log({"cluster_summary": wandb.Table(
        columns=["Cluster", "Type", "Size", "AI-preferring", "Human-preferring", "Mean AUC", "Dominant Layer"],
        data=cluster_table_data
    )})
    
    #Step 7: Log cluster metrics
    wandb_run.log({
        "n_clusters": n_clusters,
        "largest_cluster_size": max(c['size'] for c in cluster_summary),
        "smallest_cluster_size": min(c['size'] for c in cluster_summary),
    })
    
    print(f"\nClustering analysis complete!")
    print(f"  Identified {n_clusters} distinct neuron groups")
    print(f"  Results logged to wandb")
    
    # TODO: Future extensions:
    # - Try different clustering algorithms (hierarchical, DBSCAN)
    # - Analyze within-cluster activation correlations
    # - Identify prototype neurons for each cluster
    # - Examine linguistic patterns per cluster


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
    print(f"Early layers (<= 6): {early_disc} discriminative")
    print(f"Late layers (> 6): {late_disc} discriminative")
    print(f"Interpretation: {'Semantic' if late_disc > early_disc else 'Syntactic'} features dominate")


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
    ALPHA = 0.001  # Significance level (before Bonferroni)x
    AUC_THRESHOLD = 0.7  # AUC > 0.7 or < 0.3 for discrimination
    
    # Initialize wandb (mandatory)
    try:
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
        print(f"Weights & Biases initialized: {args.wandb_project}/{args.wandb_run_name}")
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
    
    # Plot all neurons AUC distribution (comprehensive view)
    plot_all_neurons_auc(all_stats, wandb_run)
    
    # Create visualizations and log to wandb
    create_wandb_visualizations(all_stats, activations, labels, wandb_run)
    
    # Create tables and log to wandb
    create_wandb_tables(all_stats, wandb_run)
    
    # Log overall metrics
    log_overall_metrics(all_stats, wandb_run)
    
    # Cluster analysis (identify neuron groups)
    cluster_analysis(activations, labels, all_stats, wandb_run, n_clusters=5)
    
    wandb_run.finish()
    
    print("\n" + "=" * 80)
    print("Analysis complete")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - CSV files: {args.input}/layer_X_neuron_stats.csv")
    print(f"  - Visualizations and tables: https://wandb.ai/{args.wandb_project}")


if __name__ == "__main__":
    main()
