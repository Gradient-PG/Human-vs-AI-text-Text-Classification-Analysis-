"""
K-Means Clustering Reference Code
==================================
This code was used for initial exploration but replaced with Hierarchical Clustering
for the final paper. Kept here for reference.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Assuming X_neurons and embedding are already defined

# Test different K values and evaluate using silhouette score
k_range = range(2, 11)  # Test K from 2 to 10
silhouette_scores = []
inertias = []

print("Evaluating different K values for neuron clusters...")
print("-" * 60)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_neurons)
    
    # Calculate silhouette score
    score = silhouette_score(X_neurons, cluster_labels)
    silhouette_scores.append(score)
    inertias.append(kmeans.inertia_)
    
    print(f"K={k}: Silhouette Score = {score:.4f}")

# Find optimal K
optimal_k = k_range[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)

print("-" * 60)
print(f"✓ Optimal K: {optimal_k} (Silhouette Score: {best_silhouette:.4f})")

# Visualize the scores
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Silhouette scores
axes[0].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(optimal_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal K={optimal_k}')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Silhouette Score', fontsize=12)
axes[0].set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Elbow plot (inertia)
axes[1].plot(k_range, inertias, 'go-', linewidth=2, markersize=8)
axes[1].axvline(optimal_k, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Optimal K={optimal_k}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
axes[1].set_title('Elbow Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('../results/figures/neuron_optimal_k_selection.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ K selection plot saved to results/figures/neuron_optimal_k_selection.png")

# Apply K-means with optimal K to cluster neurons
n_clusters = optimal_k
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
neuron_clusters = kmeans.fit_predict(X_neurons)

print(f"\nFinal clustering with K={n_clusters}")
print("=" * 60)
print(f"Neuron cluster distribution:")
for i in range(n_clusters):
    n_neurons = np.sum(neuron_clusters == i)
    print(f"  Cluster {i}: {n_neurons} neurons")

