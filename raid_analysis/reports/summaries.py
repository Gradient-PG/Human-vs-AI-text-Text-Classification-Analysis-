"""Text reports: neuron stats and clustering summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import ALPHA, AUC_HIGH, AUC_LOW
from ..data.activations import load_activation_column


def neuron_summary_text(
    neurons_df: pd.DataFrame, disc_df: pd.DataFrame, results_path: Path, model: str
) -> str:
    n_total = len(neurons_df)
    n_disc = len(disc_df)
    n_ai = (disc_df["auc"] > AUC_HIGH).sum()
    n_hu = (disc_df["auc"] < AUC_LOW).sum()
    layer_counts = disc_df.groupby("layer").size()
    peak_layer = layer_counts.idxmax() if not layer_counts.empty else "N/A"
    early = disc_df[disc_df["layer"] <= 4]
    middle = disc_df[(disc_df["layer"] >= 5) & (disc_df["layer"] <= 8)]
    late = disc_df[disc_df["layer"] >= 9]

    lines = [
        "=" * 60,
        f"DISCRIMINATIVE NEURON SUMMARY  [{model}]",
        "=" * 60,
        f"{'Total neurons analyzed':<40} {n_total:>15,}",
        f"{'Discriminative neurons':<40} {n_disc:>15,}",
        f"{'Percentage of total':<40} {n_disc/n_total*100:>14.1f}%",
        "-" * 60,
        f"{'AI-preferring (AUC > 0.7)':<40} {n_ai:>15,}",
        f"{'Human-preferring (AUC < 0.3)':<40} {n_hu:>15,}",
        f"{'AI:Human ratio':<40} {n_ai/max(n_hu, 1):>14.2f}:1",
        f"{'Mean AUC deviation from 0.5':<40} {disc_df['auc_deviation'].mean():>15.3f}",
        "-" * 60,
        f"{'Early layers (1-4)':<40} {len(early):>15,}",
        f"{'Middle layers (5-8)':<40} {len(middle):>15,}",
        f"{'Late layers (9-12)':<40} {len(late):>15,}",
        f"{'Peak layer':<40} {'Layer ' + str(peak_layer):>15}",
        "=" * 60,
    ]

    if n_disc > 0:
        top20 = disc_df.nlargest(20, "auc_deviation")[
            ["layer", "neuron_idx", "auc", "cohens_d", "p_value", "direction"]
        ].copy()
        top20.insert(0, "rank", range(1, len(top20) + 1))
        lines += ["", "TOP-20 DISCRIMINATIVE NEURONS", "=" * 85, top20.to_string(index=False), "=" * 85]

    if n_disc == 0:
        lines.append("\n(No discriminative neurons found — increase --samples to get more data.)")
        return "\n".join(lines)

    lines += [
        "",
        "ACTIVATION PATTERN",
        "=" * 50,
        f"  Higher for AI:    {(disc_df['mean_diff'] > 0).sum()}",
        f"  Higher for Human: {(disc_df['mean_diff'] < 0).sum()}",
        f"  Mean |Cohen d|:   {disc_df['cohens_d'].abs().mean():.3f}",
        f"  Median |Cohen d|: {disc_df['cohens_d'].abs().median():.3f}",
        f"  Large effects (|d| > 0.8): {(disc_df['cohens_d'].abs() > 0.8).sum()}",
        f"  Mean variance ratio (AI/Human): {disc_df['variance_ratio'].mean():.3f}",
    ]

    n_tests = n_total
    bonferroni_alpha = ALPHA / n_tests
    n_survive = (disc_df["p_value"] < bonferroni_alpha).sum()
    survive_pct = n_survive / n_disc * 100

    lines += [
        "",
        "STATISTICAL VALIDATION",
        "=" * 70,
        f"{'Bonferroni correction survived':<45} {survive_pct:.0f}%",
        f"{'Large effect sizes (|d| > 0.8)':<45} {(disc_df['cohens_d'].abs() > 0.8).sum() / n_disc * 100:.1f}%",
        f"{'Median p-value':<45} {disc_df['p_value'].median():.2e}",
        f"{'Max p-value':<45} {disc_df['p_value'].max():.2e}",
    ]

    top50 = disc_df.nlargest(min(50, n_disc), "auc_deviation")
    if len(top50) >= 2:
        act_mat = np.array([
            load_activation_column(results_path, int(r["layer"]), int(r["neuron_idx"]))
            for _, r in top50.iterrows()
        ])
        corr = np.corrcoef(act_mat)
        tri = corr[np.triu_indices_from(corr, k=1)]
        mean_corr = float(np.abs(tri).mean())
        high_corr_pct = float((np.abs(tri) > 0.7).sum() / len(tri) * 100)
        lines += [
            f"{'Mean pairwise correlation (top 50)':<45} {mean_corr:.3f}",
            f"{'Highly correlated pairs (|r| > 0.7)':<45} {high_corr_pct:.1f}%",
        ]

    lines.append("=" * 70)
    return "\n".join(lines)


def clustering_summary_text(
    neurons_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    model: str,
    method: str = "Ward",
    extra_header_lines: list | None = None,
) -> str:
    is_disc = neurons_df["discriminative"].values
    global_disc_frac = float(is_disc.mean()) if len(is_disc) else 0.0
    n_total = len(neurons_df)
    n_disc = int(is_disc.sum())

    unique_sorted = sorted(set(cluster_labels))
    real_ids = [c for c in unique_sorted if c >= 0]
    has_noise = -1 in unique_sorted
    n_noise = int((cluster_labels == -1).sum()) if has_noise else 0

    lines = [
        "=" * 80,
        f"{method.upper()} CLUSTER ANALYSIS  [{model}]",
        "=" * 80,
        f"Total neurons:                {n_total}",
        f"Discriminative neurons:       {n_disc}  ({n_disc/n_total*100:.1f}%)",
        f"Clusters found:               {len(real_ids)}",
    ]
    if has_noise:
        lines.append(f"Noise points:                 {n_noise}")
    if extra_header_lines:
        lines.extend(extra_header_lines)

    for cid in real_ids:
        mask_c = cluster_labels == cid
        cn = neurons_df[mask_c]
        cn_disc = cn[cn["discriminative"]]
        n_cn = len(cn)
        n_cn_disc = len(cn_disc)
        disc_frac = n_cn_disc / n_cn if n_cn > 0 else 0.0
        enrichment = disc_frac / global_disc_frac if global_disc_frac > 0 else 0.0

        lines += [
            "",
            f"Cluster {cid}: {n_cn} neurons  ({n_cn_disc} discriminative, "
            f"enrichment = {enrichment:.2f}x)",
            "-" * 80,
            "  Layer distribution:",
        ]
        for layer, count in cn["layer"].value_counts().sort_index().items():
            lines.append(f"    Layer {layer:2d}: {count:3d} ({count/n_cn*100:.1f}%)")

        if n_cn_disc > 0:
            lines.append("  Discriminative neuron direction:")
            for direction, count in cn_disc["direction"].value_counts().items():
                lines.append(f"    {direction}: {count:3d} ({count/n_cn_disc*100:.1f}%)")
            lines += [
                f"  Mean AUC (disc. only):      {cn_disc['auc'].mean():.4f}  "
                f"(std: {cn_disc['auc'].std():.4f})",
                f"  Mean AUC deviation:         {cn_disc['auc_deviation'].mean():.4f}",
            ]
        else:
            lines.append("  No discriminative neurons in this cluster.")

    if has_noise and n_noise > 0:
        noise_disc = neurons_df[cluster_labels == -1]
        n_noise_disc = noise_disc["discriminative"].sum()
        lines += [
            "",
            f"Noise: {n_noise} neurons  ({n_noise_disc} discriminative)",
        ]

    lines += [
        "",
        "=" * 80,
        "OVERALL DISCRIMINATIVE DISTRIBUTION",
        "=" * 80,
        "Direction:",
        neurons_df[neurons_df["discriminative"]]["direction"].value_counts().to_string(),
        "",
        "Layer (discriminative only):",
        neurons_df[neurons_df["discriminative"]]["layer"].value_counts().sort_index().to_string(),
    ]
    return "\n".join(lines)
