import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HARD TASK: ADVANCED EVALUATION & MULTI-MODAL ANALYSIS")
print("=" * 70)

# Load all latent features from different methods
print("\nLoading all latent representations...")
vae_latent = np.load('results/latent_features.npy')
conv_latent = np.load('results/conv_latent_features.npy')
cvae_latent = np.load('results/cvae_latent_features.npy')

# Load conditions (simulating genre/language)
conditions = np.load('results/conditions.npy')

print(f"VAE latent shape: {vae_latent.shape}")
print(f"Conv-VAE latent shape: {conv_latent.shape}")
print(f"CVAE latent shape: {cvae_latent.shape}")
print(f"Condition labels: {np.unique(conditions)} (5 simulated genres)")

n_clusters = 5
methods_data = {
    'VAE': vae_latent,
    'Conv-VAE': conv_latent,
    'CVAE (Disentangled)': cvae_latent,
}

# =============================================================================
# PURITY METRIC
# =============================================================================

def cluster_purity(labels_pred, labels_true):
    """Calculate cluster purity"""
    purity = 0.0
    n = len(labels_pred)
    
    unique_predicted = np.unique(labels_pred)
    
    for cluster_id in unique_predicted:
        cluster_mask = (labels_pred == cluster_id)
        true_labels_in_cluster = labels_true[cluster_mask]
        
        if len(true_labels_in_cluster) > 0:
            most_common_count = np.bincount(true_labels_in_cluster).max()
            purity += most_common_count
    
    return purity / n


# =============================================================================
# EVALUATE ALL METHODS
# =============================================================================

print("\n" + "=" * 70)
print("EVALUATING ALL METHODS WITH HARD TASK METRICS")
print("=" * 70)

results = {}

for method_name, latent_features in methods_data.items():
    print(f"\n{'-' * 70}")
    print(f"Evaluating: {method_name}")
    print(f"{'-' * 70}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latent_features)
    
    silhouette = silhouette_score(latent_features, labels)
    calinski = calinski_harabasz_score(latent_features, labels)
    davies = davies_bouldin_score(latent_features, labels)
    
    nmi = normalized_mutual_info_score(conditions, labels)
    ari = adjusted_rand_score(conditions, labels)
    purity = cluster_purity(labels, conditions)
    
    print(f"\nClustering Metrics:")
    print(f"  Silhouette Score: {silhouette:.4f}")
    print(f"  Calinski-Harabasz Index: {calinski:.4f}")
    print(f"  Davies-Bouldin Index: {davies:.4f}")
    
    print(f"\nLabel-Based Metrics:")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Cluster Purity: {purity:.4f}")
    
    print(f"\nCluster Distribution: {np.bincount(labels)}")
    
    results[method_name] = {
        'labels': labels,
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies,
        'nmi': nmi,
        'ari': ari,
        'purity': purity,
        'features': latent_features
    }

# =============================================================================
# METRICS TABLE
# =============================================================================

print("\n" + "=" * 70)
print("COMPREHENSIVE METRICS COMPARISON")
print("=" * 70)

metrics_table = pd.DataFrame({
    'Method': list(results.keys()),
    'Silhouette': [results[m]['silhouette'] for m in results.keys()],
    'Calinski-Harabasz': [results[m]['calinski'] for m in results.keys()],
    'Davies-Bouldin': [results[m]['davies'] for m in results.keys()],
    'NMI': [results[m]['nmi'] for m in results.keys()],
    'ARI': [results[m]['ari'] for m in results.keys()],
    'Purity': [results[m]['purity'] for m in results.keys()],
})

print("\n", metrics_table.to_string(index=False))

metrics_table.to_csv('results/hard_task_metrics.csv', index=False)
print("\n✓ Saved to results/hard_task_metrics.csv")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\nGenerating advanced visualizations...")

fig, axes = plt.subplots(3, 3, figsize=(16, 14))

tsne = TSNE(n_components=2, random_state=42, perplexity=20)

for row_idx, (method_name, method_data) in enumerate(results.items()):
    latent_2d = tsne.fit_transform(method_data['features'])
    labels = method_data['labels']
    
    # Plot 1: Predicted clusters
    scatter1 = axes[row_idx, 0].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                        c=labels, cmap='viridis', s=80, 
                                        alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[row_idx, 0].set_title(f'{method_name}\nClusters (Silhouette: {method_data["silhouette"]:.3f})', 
                                fontweight='bold')
    axes[row_idx, 0].set_xlabel('t-SNE 1')
    axes[row_idx, 0].set_ylabel('t-SNE 2')
    axes[row_idx, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[row_idx, 0], label='Cluster')
    
    # Plot 2: Ground truth
    scatter2 = axes[row_idx, 1].scatter(latent_2d[:, 0], latent_2d[:, 1], 
                                        c=conditions, cmap='plasma', s=80, 
                                        alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[row_idx, 1].set_title(f'{method_name}\nGround Truth (NMI: {method_data["nmi"]:.3f})', 
                                fontweight='bold')
    axes[row_idx, 1].set_xlabel('t-SNE 1')
    axes[row_idx, 1].set_ylabel('t-SNE 2')
    axes[row_idx, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[row_idx, 1], label='Class')
    
    # Plot 3: Metrics
    metric_names = ['Silhouette', 'NMI', 'ARI', 'Purity']
    metric_values = [method_data['silhouette'], method_data['nmi'], 
                     method_data['ari'], method_data['purity']]
    
    bars = axes[row_idx, 2].bar(metric_names, metric_values, color=['steelblue', 'coral', 'lightgreen', 'gold'], alpha=0.7)
    axes[row_idx, 2].set_title(f'{method_name}\nMetrics', fontweight='bold')
    axes[row_idx, 2].set_ylim([0, 1.0])
    axes[row_idx, 2].grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        axes[row_idx, 2].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/hard_task_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved to results/hard_task_analysis.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("HARD TASK SUMMARY")
print("=" * 70)

best_silhouette = metrics_table.loc[metrics_table['Silhouette'].idxmax()]
best_nmi = metrics_table.loc[metrics_table['NMI'].idxmax()]
best_purity = metrics_table.loc[metrics_table['Purity'].idxmax()]

print(f"\nBest by Silhouette Score:")
print(f"  {best_silhouette['Method']}: {best_silhouette['Silhouette']:.4f}")

print(f"\nBest by NMI (Label Alignment):")
print(f"  {best_nmi['Method']}: {best_nmi['NMI']:.4f}")

print(f"\nBest by Purity:")
print(f"  {best_purity['Method']}: {best_purity['Purity']:.4f}")

print("\n" + "=" * 70)
print("HARD TASK COMPLETE!")
print("=" * 70)