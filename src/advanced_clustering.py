import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                             davies_bouldin_score, adjusted_rand_score)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ADVANCED CLUSTERING WITH MULTIPLE ALGORITHMS")
print("=" * 70)

# Load Conv-VAE latent features
print("\nLoading Conv-VAE latent features...")
latent_features = np.load('results/conv_latent_features.npy')
print(f"Latent features shape: {latent_features.shape}")

# Load original data
df = pd.read_csv('results/audio_features.csv')
original_features = df.drop('file_name', axis=1).values

# Apply PCA for baseline
pca = PCA(n_components=5)
pca_features = pca.fit_transform(original_features)

n_clusters = 5

# =============================================================================
# CLUSTERING METHOD 1: K-MEANS
# =============================================================================
print("\n" + "-" * 70)
print("METHOD 1: K-MEANS CLUSTERING")
print("-" * 70)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(latent_features)

silhouette_kmeans = silhouette_score(latent_features, labels_kmeans)
calinski_kmeans = calinski_harabasz_score(latent_features, labels_kmeans)
davies_kmeans = davies_bouldin_score(latent_features, labels_kmeans)

print(f"\nK-Means Results:")
print(f"  Silhouette Score: {silhouette_kmeans:.4f}")
print(f"  Calinski-Harabasz Index: {calinski_kmeans:.4f}")
print(f"  Davies-Bouldin Index: {davies_kmeans:.4f}")
print(f"  Cluster distribution: {np.bincount(labels_kmeans)}")

# =============================================================================
# CLUSTERING METHOD 2: AGGLOMERATIVE CLUSTERING (Hierarchical)
# =============================================================================
print("\n" + "-" * 70)
print("METHOD 2: AGGLOMERATIVE CLUSTERING (Hierarchical)")
print("-" * 70)

agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels_agg = agg.fit_predict(latent_features)

silhouette_agg = silhouette_score(latent_features, labels_agg)
calinski_agg = calinski_harabasz_score(latent_features, labels_agg)
davies_agg = davies_bouldin_score(latent_features, labels_agg)

print(f"\nAgglomerative Clustering Results:")
print(f"  Silhouette Score: {silhouette_agg:.4f}")
print(f"  Calinski-Harabasz Index: {calinski_agg:.4f}")
print(f"  Davies-Bouldin Index: {davies_agg:.4f}")
print(f"  Cluster distribution: {np.bincount(labels_agg)}")

# =============================================================================
# CLUSTERING METHOD 3: DBSCAN
# =============================================================================
print("\n" + "-" * 70)
print("METHOD 3: DBSCAN (Density-Based)")
print("-" * 70)

# Find good eps value using k-distance graph (k=4)
from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(latent_features)
distances, indices = neighbors_fit.kneighbors(latent_features)
distances = np.sort(distances[:, -1], axis=0)
eps_value = np.percentile(distances, 70)

dbscan = DBSCAN(eps=eps_value, min_samples=3)
labels_dbscan = dbscan.fit_predict(latent_features)

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise = list(labels_dbscan).count(-1)

print(f"\nDBSCAN Results:")
print(f"  Number of clusters found: {n_clusters_dbscan}")
print(f"  Number of noise points: {n_noise}")

if n_clusters_dbscan > 1:
    # Remove noise points for metric calculation
    mask = labels_dbscan != -1
    if np.sum(mask) > 1:
        silhouette_dbscan = silhouette_score(latent_features[mask], labels_dbscan[mask])
        calinski_dbscan = calinski_harabasz_score(latent_features[mask], labels_dbscan[mask])
        davies_dbscan = davies_bouldin_score(latent_features[mask], labels_dbscan[mask])
        
        print(f"  Silhouette Score: {silhouette_dbscan:.4f} (excluding noise)")
        print(f"  Calinski-Harabasz Index: {calinski_dbscan:.4f}")
        print(f"  Davies-Bouldin Index: {davies_dbscan:.4f}")
    else:
        print(f"  Not enough clustered points for metrics")
else:
    print(f"  DBSCAN found only 1 cluster (increase eps or decrease min_samples)")

# =============================================================================
# COMPARISON TABLE
# =============================================================================
print("\n" + "=" * 70)
print("CLUSTERING COMPARISON TABLE")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Method': ['K-Means', 'Agglomerative', 'DBSCAN*', 'PCA + K-Means (Baseline)'],
    'Silhouette Score': [
        silhouette_kmeans,
        silhouette_agg,
        silhouette_dbscan if n_clusters_dbscan > 1 else np.nan,
        silhouette_score(pca_features, KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(pca_features))
    ],
    'Calinski-Harabasz': [
        calinski_kmeans,
        calinski_agg,
        calinski_dbscan if n_clusters_dbscan > 1 else np.nan,
        calinski_harabasz_score(pca_features, KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(pca_features))
    ],
    'Davies-Bouldin': [
        davies_kmeans,
        davies_agg,
        davies_dbscan if n_clusters_dbscan > 1 else np.nan,
        davies_bouldin_score(pca_features, KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(pca_features))
    ]
})

print("\n", comparison_df.to_string(index=False))
comparison_df.to_csv('results/advanced_clustering_metrics.csv', index=False)
print("\n✓ Saved to results/advanced_clustering_metrics.csv")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\nGenerating visualizations...")

# Prepare 2D visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
latent_2d = tsne.fit_transform(latent_features)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: K-Means
scatter1 = axes[0, 0].scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels_kmeans,
                              cmap='viridis', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0, 0].set_title(f'K-Means (Silhouette: {silhouette_kmeans:.3f})', fontweight='bold')
axes[0, 0].set_xlabel('t-SNE 1')
axes[0, 0].set_ylabel('t-SNE 2')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0, 0], label='Cluster')

# Plot 2: Agglomerative
scatter2 = axes[0, 1].scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels_agg,
                              cmap='viridis', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[0, 1].set_title(f'Agglomerative (Silhouette: {silhouette_agg:.3f})', fontweight='bold')
axes[0, 1].set_xlabel('t-SNE 1')
axes[0, 1].set_ylabel('t-SNE 2')
axes[0, 1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[0, 1], label='Cluster')

# Plot 3: DBSCAN
colors = labels_dbscan.copy().astype(float)
colors[colors == -1] = -1  # Noise points
scatter3 = axes[1, 0].scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors,
                              cmap='viridis', s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
axes[1, 0].set_title(f'DBSCAN (Clusters: {n_clusters_dbscan}, Noise: {n_noise})', fontweight='bold')
axes[1, 0].set_xlabel('t-SNE 1')
axes[1, 0].set_ylabel('t-SNE 2')
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=axes[1, 0], label='Cluster')

# Plot 4: Metrics comparison
methods = ['K-Means', 'Agglomerative', 'PCA+KM']
silhouette_scores = [silhouette_kmeans, silhouette_agg, 
                     silhouette_score(pca_features, KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(pca_features))]

x_pos = np.arange(len(methods))
bars = axes[1, 1].bar(x_pos, silhouette_scores, color=['steelblue', 'coral', 'lightgreen'], alpha=0.7)
axes[1, 1].set_ylabel('Silhouette Score')
axes[1, 1].set_title('Clustering Quality Comparison', fontweight='bold')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(methods)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('results/advanced_clustering_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved to results/advanced_clustering_visualization.png")
plt.close()

# =============================================================================
# SAVE CLUSTER ASSIGNMENTS
# =============================================================================
results_df = pd.DataFrame({
    'file_name': df['file_name'],
    'KMeans_Cluster': labels_kmeans,
    'Agglomerative_Cluster': labels_agg,
    'DBSCAN_Cluster': labels_dbscan,
})

results_df.to_csv('results/advanced_cluster_assignments.csv', index=False)
print("✓ Saved to results/advanced_cluster_assignments.csv")

print("\n" + "=" * 70)
print("ADVANCED CLUSTERING COMPLETE!")
print("=" * 70)
print("\nBest performing method:")
best_method = comparison_df.loc[comparison_df['Silhouette Score'].idxmax()]
print(f"  {best_method['Method']}: {best_method['Silhouette Score']:.4f}")