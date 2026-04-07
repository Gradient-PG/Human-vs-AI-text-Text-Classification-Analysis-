"""Shared constants for RAID multi-model neuron analysis."""

ALL_LAYERS = list(range(1, 13))

ALPHA = 0.001
AUC_LOW = 0.3
AUC_HIGH = 0.7

# Total number of BERT-base-uncased neurons across all layers (12 × 768).
# Used for global Bonferroni correction across all 9,216 hypothesis tests.
N_BERT_NEURONS = len(ALL_LAYERS) * 768  # 9,216

CLUSTER_PALETTE = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
