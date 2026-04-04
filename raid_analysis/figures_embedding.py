"""2D embedding plots (PCA or UMAP) for all neurons and cluster overlays."""

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from .constants import AUC_HIGH, AUC_LOW


def fig_embedding_all_neurons(
    embedding: np.ndarray,
    neurons_df: pd.DataFrame,
    model: str,
    axis_x: str,
    axis_y: str,
    embed_name: str,
) -> plt.Figure:
    auc_dev = neurons_df["auc_deviation"].values
    is_disc = neurons_df["discriminative"].values
    auc = neurons_df["auc"].values

    n_disc = is_disc.sum()
    n_ai = (is_disc & (auc > 0.5)).sum()
    n_hu = (is_disc & (auc <= 0.5)).sum()
    n_non = (~is_disc).sum()

    n_auc_hi = int((auc > 0.5).sum())
    n_auc_lo = int((auc <= 0.5).sum())

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    ax_tl, ax_tr = axes[0]
    ax_bl, ax_br = axes[1]

    # (1) Top-left: discrimination strength = AUC deviation (statistical disc. threshold)
    sc1 = ax_tl.scatter(
        embedding[:, 0], embedding[:, 1],
        c=auc_dev, cmap="plasma",
        alpha=0.5, s=6,
        vmin=0, vmax=0.5,
        rasterized=True,
    )
    cb1 = fig.colorbar(sc1, ax=ax_tl, fraction=0.046, pad=0.04)
    cb1.set_label("AUC deviation from 0.5", fontsize=10)
    cb1.ax.axhline(0.2, color="white", linewidth=1.5, linestyle="--")
    cb1.ax.text(0.55, 0.41, "disc.\nthreshold", transform=cb1.ax.transAxes,
                fontsize=7, color="white", va="center")
    ax_tl.set_title(
        f"Discrimination strength (AUC deviation, disc. threshold) - ({embed_name}) [{model}]",
        fontsize=12,
        fontweight="bold",
    )
    ax_tl.set_xlabel(axis_x, fontsize=11)
    ax_tl.set_ylabel(axis_y, fontsize=11)

    # (2) Top-right: raw AUC in [0, 1]
    sc2 = ax_tr.scatter(
        embedding[:, 0], embedding[:, 1],
        c=auc, cmap="RdBu_r",
        alpha=0.5, s=6,
        vmin=0, vmax=1,
        rasterized=True,
    )
    cb2 = fig.colorbar(sc2, ax=ax_tr, fraction=0.046, pad=0.04)
    cb2.set_label("AUC  (0 = Human, 1 = AI)", fontsize=10)
    cb2.ax.axhline(0.5, color="black", linewidth=1.2, linestyle="--")
    ax_tr.set_title(
        f"Raw AUC ({embed_name}) [{model}]",
        fontsize=12,
        fontweight="bold",
    )
    ax_tr.set_xlabel(axis_x, fontsize=11)
    ax_tr.set_ylabel(axis_y, fontsize=11)

    # (3) Bottom-left: binary split at AUC = 0.5 (all neurons)
    m_lo = auc <= 0.5
    m_hi = auc > 0.5
    ax_bl.scatter(
        embedding[m_lo, 0], embedding[m_lo, 1],
        c="#3498db", alpha=0.45, s=6,
        label=f"AUC ≤ 0.5  (n={n_auc_lo:,})",
        rasterized=True,
    )
    ax_bl.scatter(
        embedding[m_hi, 0], embedding[m_hi, 1],
        c="#e74c3c", alpha=0.45, s=6,
        label=f"AUC > 0.5  (n={n_auc_hi:,})",
        rasterized=True,
    )
    ax_bl.set_title(
        f"Binary AUC split at 0.5 ({embed_name}) [{model}]",
        fontsize=12,
        fontweight="bold",
    )
    ax_bl.set_xlabel(axis_x, fontsize=11)
    ax_bl.set_ylabel(axis_y, fontsize=11)
    ax_bl.legend(fontsize=9, loc="best", framealpha=0.9)

    # (4) Bottom-right: statistically discriminative neurons only (original panel)
    non_mask = ~is_disc
    ai_mask = is_disc & (auc > 0.5)
    hu_mask = is_disc & (auc <= 0.5)

    ax_br.scatter(embedding[non_mask, 0], embedding[non_mask, 1],
                  c="#cccccc", alpha=0.25, s=5,
                  label=f"Non-discriminative  (n={n_non:,})",
                  rasterized=True)
    ax_br.scatter(embedding[ai_mask, 0], embedding[ai_mask, 1],
                  c="#e74c3c", alpha=0.85, s=18, edgecolors="none",
                  label=f"AI-preferring  (n={n_ai:,})")
    ax_br.scatter(embedding[hu_mask, 0], embedding[hu_mask, 1],
                  c="#3498db", alpha=0.85, s=18, edgecolors="none",
                  label=f"Human-preferring  (n={n_hu:,})")
    ax_br.set_title(
        f"Discriminative highlighted ({embed_name}) [{model}]",
        fontsize=12,
        fontweight="bold",
    )
    ax_br.set_xlabel(axis_x, fontsize=11)
    ax_br.set_ylabel(axis_y, fontsize=11)
    ax_br.legend(fontsize=9, loc="best", framealpha=0.9,
                 title=f"Total discriminative: {n_disc:,} / {len(neurons_df):,}")

    fig.tight_layout()
    return fig


def fig_full_space_clustering(
    embedding: np.ndarray,
    neurons_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    model: str,
    method: str,
    axis_x: str,
    axis_y: str,
) -> plt.Figure:
    is_disc = neurons_df["discriminative"].values
    auc = neurons_df["auc"].values
    auc_deviation = neurons_df["auc_deviation"].values
    layers = neurons_df["layer"].values

    ai_mask = is_disc & (auc > 0.5)
    hu_mask = is_disc & (auc <= 0.5)
    non_disc_mask = ~is_disc

    unique_sorted = sorted(set(cluster_labels))
    has_noise = -1 in unique_sorted
    real_ids = [c for c in unique_sorted if c >= 0]
    n_real = len(real_ids)

    cmap_clust = cm.get_cmap("tab10", max(n_real, 1))
    cluster_colors = {c: cmap_clust(i) for i, c in enumerate(real_ids)}
    if has_noise:
        cluster_colors[-1] = (0.75, 0.75, 0.75, 0.25)

    def layer_group(l):
        return 0 if l <= 4 else (1 if l <= 8 else 2)

    lg = np.vectorize(layer_group)(layers)
    lg_labels = ["Early (1-4)", "Mid (5-8)", "Late (9-12)"]
    lg_colors = ["#8e44ad", "#f39c12", "#16a085"]

    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.1, 0.8], hspace=0.28, wspace=0.25)
    ax_clust = fig.add_subplot(gs[0, 0])
    ax_auc = fig.add_subplot(gs[0, 1])
    ax_disc = fig.add_subplot(gs[0, 2])
    ax_layer = fig.add_subplot(gs[1, 0])
    gs_right = gs[1, 1:].subgridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.42)
    ax_strength = fig.add_subplot(gs_right[0, 0])
    ax_dir = fig.add_subplot(gs_right[1, 0])

    point_colors = np.array([cluster_colors[k] for k in cluster_labels])

    if has_noise:
        noise_m = cluster_labels == -1
        ax_clust.scatter(embedding[noise_m, 0], embedding[noise_m, 1],
                         c=[(0.75, 0.75, 0.75)], alpha=0.15, s=4, rasterized=True)
    for c in real_ids:
        m = cluster_labels == c
        ax_clust.scatter(embedding[m, 0], embedding[m, 1],
                         c=[cluster_colors[c]], alpha=0.45, s=6, rasterized=True)

    handles = [mpatches.Patch(facecolor=cluster_colors[c], label=f"Cluster {c}")
               for c in real_ids]
    if has_noise:
        n_noise = int((cluster_labels == -1).sum())
        handles.append(mpatches.Patch(facecolor=(0.75, 0.75, 0.75),
                                       label=f"Noise (n={n_noise:,})"))
    ax_clust.legend(handles=handles, fontsize=9, loc="best",
                    framealpha=0.9, title=f"{method}  K={n_real}")
    ax_clust.set_title(f"Full Neuron Space — {method} [{model}]",
                       fontsize=13, fontweight="bold")
    ax_clust.set_xlabel(axis_x, fontsize=11)
    ax_clust.set_ylabel(axis_y, fontsize=11)

    sc = ax_auc.scatter(
        embedding[:, 0], embedding[:, 1],
        c=auc, cmap="RdBu_r", vmin=0, vmax=1,
        alpha=0.5, s=6, rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax_auc, fraction=0.046, pad=0.04)
    cb.set_label("AUC  (0 = Human, 1 = AI)", fontsize=10)
    cb.ax.axhline(0.5, color="black", linewidth=1.5, linestyle="--")
    cb.ax.axhline(AUC_HIGH, color="black", linewidth=0.8, linestyle=":")
    cb.ax.axhline(AUC_LOW, color="black", linewidth=0.8, linestyle=":")
    ax_auc.set_title(f"Signed AUC — {method} [{model}]",
                     fontsize=13, fontweight="bold")
    ax_auc.set_xlabel(axis_x, fontsize=11)
    ax_auc.set_ylabel(axis_y, fontsize=11)

    ax_disc.scatter(embedding[:, 0], embedding[:, 1],
                    c=point_colors, alpha=0.12, s=5, rasterized=True)
    non_mask = ~is_disc
    ax_disc.scatter(embedding[non_mask, 0], embedding[non_mask, 1],
                    c="#aaaaaa", alpha=0.2, s=5, rasterized=True,
                    label=f"Non-discriminative (n={non_mask.sum():,})")
    ax_disc.scatter(embedding[ai_mask, 0], embedding[ai_mask, 1],
                    c="#e74c3c", alpha=0.9, s=22, edgecolors="none",
                    label=f"AI-preferring (n={ai_mask.sum():,})")
    ax_disc.scatter(embedding[hu_mask, 0], embedding[hu_mask, 1],
                    c="#3498db", alpha=0.9, s=22, edgecolors="none",
                    label=f"Human-preferring (n={hu_mask.sum():,})")
    ax_disc.set_title(f"Discriminative on {method} Map [{model}]",
                      fontsize=13, fontweight="bold")
    ax_disc.set_xlabel(axis_x, fontsize=11)
    ax_disc.set_ylabel(axis_y, fontsize=11)
    ax_disc.legend(fontsize=9, loc="best", framealpha=0.9)

    n_bars = n_real
    bar_width = 0.65
    bottoms = np.zeros(n_bars)
    x_pos = np.arange(n_bars)
    for g_idx, (g_label, g_col) in enumerate(zip(lg_labels, lg_colors)):
        counts = np.array([((cluster_labels == c) & (lg == g_idx)).sum()
                           for c in real_ids])
        totals = np.array([(cluster_labels == c).sum() for c in real_ids])
        fracs = counts / np.maximum(totals, 1)
        ax_layer.bar(x_pos, fracs, bar_width, bottom=bottoms,
                     color=g_col, label=g_label, edgecolor="white", linewidth=0.5)
        bottoms += fracs

    ax_layer.set_xticks(x_pos)
    ax_layer.set_xticklabels([f"Cluster {c}" for c in real_ids], fontsize=10)
    ax_layer.set_ylabel("Fraction of neurons", fontsize=11)
    ax_layer.set_title(f"Layer Composition — {method} [{model}]",
                       fontsize=13, fontweight="bold")
    ax_layer.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax_layer.set_ylim(0, 1.05)
    ax_layer.grid(axis="y", alpha=0.3)

    bar_cols = [cluster_colors[c] for c in real_ids]
    mean_strength = []
    for c in real_ids:
        mask_c = cluster_labels == c
        disc_in_c = mask_c & is_disc
        if disc_in_c.any():
            mean_strength.append(float(auc_deviation[disc_in_c].mean()))
        else:
            mean_strength.append(0.0)
    bars_s = ax_strength.bar(
        x_pos, mean_strength, bar_width,
        color=bar_cols, edgecolor="black", linewidth=0.7,
    )
    ymax_s = max(mean_strength) if mean_strength else 0.0
    y_top = max(0.06, ymax_s * 1.2) if ymax_s > 0 else 0.06
    for bar, val in zip(bars_s, mean_strength):
        if val > 1e-9:
            pad = 0.02 * y_top
            ax_strength.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
    ax_strength.set_xticks(x_pos)
    ax_strength.set_xticklabels([f"Cluster {c}" for c in real_ids], fontsize=10)
    ax_strength.set_ylabel("Mean AUC deviation from 0.5\n(discriminative neurons only)",
                           fontsize=10)
    ax_strength.set_title(
        f"Discriminative Strength by Cluster — {method} [{model}]",
        fontsize=13, fontweight="bold",
    )
    ax_strength.grid(axis="y", alpha=0.3)
    ax_strength.set_ylim(0, y_top)

    dir_labels = ["Non-discriminative", "Human-preferring", "AI-preferring"]
    dir_colors = ["#aaaaaa", "#3498db", "#e74c3c"]
    dir_masks = [non_disc_mask, hu_mask, ai_mask]
    bottoms_dir = np.zeros(n_bars)
    for d_label, d_col, d_mask in zip(dir_labels, dir_colors, dir_masks):
        fracs_d = np.array(
            [((cluster_labels == c) & d_mask).sum() / max((cluster_labels == c).sum(), 1)
             for c in real_ids]
        )
        ax_dir.bar(
            x_pos, fracs_d, bar_width, bottom=bottoms_dir,
            color=d_col, label=d_label, edgecolor="white", linewidth=0.5,
        )
        bottoms_dir += fracs_d
    ax_dir.set_xticks(x_pos)
    ax_dir.set_xticklabels([f"Cluster {c}" for c in real_ids], fontsize=10)
    ax_dir.set_ylabel("Fraction of neurons in cluster", fontsize=11)
    ax_dir.set_title(
        f"Discriminative Direction Mix — {method} [{model}]",
        fontsize=13, fontweight="bold",
    )
    ax_dir.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax_dir.set_ylim(0, 1.05)
    ax_dir.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{method} Cluster Analysis — {model}  (K={n_real})",
                 fontsize=15, fontweight="bold", y=1.01)
    return fig
