import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CLUSTERING LATENT FEATURES")
print("=" * 70)

# Load latent features from trained VAE
print("\nLoading latent features...")
latent_features = np.load('results/latent_features.npy')
print(f"Latent features shape: {latent_features.shape}")

# Load original features for baseline comparison
df = pd.read_csv('results/audio_features.csv')
original_features = df.drop('file_name', axis=1).values
print(f"Original features shape: {original_features.shape}")

# =============================================================================
# CLUSTERING WITH K-MEANS ON VAE LATENT FEATURES
# =============================================================================

print("\n" + "-" * 70)
print("METHOD 1: K-MEANS ON VAE LATENT FEATURES")
print("-" * 70)

n_clusters = 5
kmeans_vae = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_vae = kmeans_vae.fit_predict(latent_features)

print(f"\nK-Means with {n_clusters} clusters")
print(f"Cluster distribution: {np.bincount(labels_vae)}")

# =============================================================================
# BASELINE: PCA + K-MEANS ON ORIGINAL FEATURES
# =============================================================================

print("\n" + "-" * 70)
print("BASELINE: PCA + K-MEANS ON ORIGINAL FEATURES")
print("-" * 70)

# Apply PCA
pca = PCA(n_components=5)
pca_features = pca.fit_transform(original_features)
print(f"\nPCA features shape: {pca_features.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")

# K-Means on PCA features
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(pca_features)

print(f"\nCluster distribution: {np.bincount(labels_pca)}")

# =============================================================================
# EVALUATION METRICS
# =============================================================================

print("\n" + "=" * 70)
print("CLUSTERING QUALITY METRICS")
print("=" * 70)

# Metrics for VAE + K-Means
silhouette_vae = silhouette_score(latent_features, labels_vae)
calinski_vae = calinski_harabasz_score(latent_features, labels_vae)
davies_vae = davies_bouldin_score(latent_features, labels_vae)

print("\nVAE + K-Means:")
print(f"  Silhouette Score: {silhouette_vae:.4f} (range: -1 to 1, higher is better)")
print(f"  Calinski-Harabasz Index: {calinski_vae:.4f} (higher is better)")
print(f"  Davies-Bouldin Index: {davies_vae:.4f} (lower is better)")

# Metrics for PCA + K-Means
silhouette_pca = silhouette_score(pca_features, labels_pca)
calinski_pca = calinski_harabasz_score(pca_features, labels_pca)
davies_pca = davies_bouldin_score(pca_features, labels_pca)

print("\nPCA + K-Means (Baseline):")
print(f"  Silhouette Score: {silhouette_pca:.4f}")
print(f"  Calinski-Harabasz Index: {calinski_pca:.4f}")
print(f"  Davies-Bouldin Index: {davies_pca:.4f}")

# Compare
print("\n" + "-" * 70)
print("COMPARISON: VAE vs PCA")
print("-" * 70)
print(f"\nSilhouette Score:")
print(f"  VAE: {silhouette_vae:.4f}")
print(f"  PCA: {silhouette_pca:.4f}")
print(f"  Improvement: {((silhouette_vae - silhouette_pca) / abs(silhouette_pca) * 100):.2f}%")

print(f"\nCalinski-Harabasz Index:")
print(f"  VAE: {calinski_vae:.4f}")
print(f"  PCA: {calinski_pca:.4f}")
improvement = ((calinski_vae - calinski_pca) / calinski_pca * 100)
print(f"  Improvement: {improvement:.2f}%")

# =============================================================================
# SAVE METRICS TO CSV
# =============================================================================

metrics_df = pd.DataFrame({
    'Method': ['VAE + K-Means', 'PCA + K-Means'],
    'Silhouette Score': [silhouette_vae, silhouette_pca],
    'Calinski-Harabasz Index': [calinski_vae, calinski_pca],
    'Davies-Bouldin Index': [davies_vae, davies_pca]
})

metrics_df.to_csv('results/clustering_metrics.csv', index=False)
print("\n✓ Metrics saved to results/clustering_metrics.csv")

# =============================================================================
# VISUALIZATION 1: T-SNE OF VAE LATENT SPACE
# =============================================================================

print("\nGenerating t-SNE visualization...")
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
latent_2d = tsne.fit_transform(latent_features)

plt.figure(figsize=(12, 5))

# Plot 1: VAE + K-Means
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels_vae, 
                       cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter1, label='Cluster')
plt.title('VAE Latent Space (t-SNE)\nK-Means Clustering', fontsize=12, fontweight='bold')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True, alpha=0.3)

# Plot 2: PCA Features
pca_2d = PCA(n_components=2)
original_2d = pca_2d.fit_transform(original_features)

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(original_2d[:, 0], original_2d[:, 1], c=labels_pca, 
                       cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter2, label='Cluster')
plt.title('PCA Latent Space (2D)\nK-Means Clustering (Baseline)', fontsize=12, fontweight='bold')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/clustering_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved to results/clustering_comparison.png")
plt.close()

# =============================================================================
# VISUALIZATION 2: CLUSTER SIZE DISTRIBUTION
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# VAE clusters
cluster_sizes_vae = np.bincount(labels_vae)
axes[0].bar(range(len(cluster_sizes_vae)), cluster_sizes_vae, color='steelblue', alpha=0.7)
axes[0].set_title('VAE + K-Means\nCluster Distribution', fontweight='bold')
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('Number of Songs')
axes[0].grid(True, alpha=0.3, axis='y')

# PCA clusters
cluster_sizes_pca = np.bincount(labels_pca)
axes[1].bar(range(len(cluster_sizes_pca)), cluster_sizes_pca, color='coral', alpha=0.7)
axes[1].set_title('PCA + K-Means\nCluster Distribution', fontweight='bold')
axes[1].set_xlabel('Cluster')
axes[1].set_ylabel('Number of Songs')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/cluster_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved to results/cluster_distribution.png")
plt.close()

# =============================================================================
# SAVE CLUSTER ASSIGNMENTS
# =============================================================================

results_df = pd.DataFrame({
    'file_name': df['file_name'],
    'VAE_Cluster': labels_vae,
    'PCA_Cluster': labels_pca
})

results_df.to_csv('results/cluster_assignments.csv', index=False)
print("✓ Saved to results/cluster_assignments.csv")

print("\n" + "=" * 70)
print("CLUSTERING COMPLETE!")
print("=" * 70)
print("\nGenerated files:")
print("  - results/clustering_metrics.csv (metrics comparison)")
print("  - results/clustering_comparison.png (visualization)")
print("  - results/cluster_distribution.png (cluster sizes)")
print("  - results/cluster_assignments.csv (cluster assignments)")