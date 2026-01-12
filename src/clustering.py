"""
Clustering algorithms and evaluation
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
from pathlib import Path

class ClusteringPipeline:
    """Pipeline for clustering experiments"""
    
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
    
    def fit_kmeans(self, features):
        """Fit K-Means clustering"""
        features_scaled = self.scaler.fit_transform(features)
        labels = self.kmeans.fit_predict(features_scaled)
        return labels
    
    def fit_agglomerative(self, features):
        """Fit Agglomerative clustering"""
        features_scaled = self.scaler.fit_transform(features)
        labels = self.agglomerative.fit_predict(features_scaled)
        return labels
    
    def fit_dbscan(self, features):
        """Fit DBSCAN clustering"""
        features_scaled = self.scaler.fit_transform(features)
        labels = self.dbscan.fit_predict(features_scaled)
        return labels

def evaluate_clustering(features, pred_labels, true_labels=None):
    """Evaluate clustering results"""
    
    metrics = {}
    
    # Internal metrics
    if len(np.unique(pred_labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(features, pred_labels)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(features, pred_labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(features, pred_labels)
    else:
        metrics['silhouette_score'] = 0
        metrics['calinski_harabasz_score'] = 0
        metrics['davies_bouldin_score'] = float('inf')
    
    # External metrics if true labels available
    if true_labels is not None:
        metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, pred_labels)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, pred_labels)
        
        # Cluster purity
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, pred_labels)
        purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
        metrics['cluster_purity'] = purity
    
    return metrics

def run_baseline_experiment(features, true_labels=None, n_clusters=10):
    """Run baseline clustering experiments"""
    
    pipeline = ClusteringPipeline(n_clusters=n_clusters)
    
    results = {}
    
    # K-Means
    kmeans_labels = pipeline.fit_kmeans(features)
    results['kmeans'] = evaluate_clustering(features, kmeans_labels, true_labels)
    
    # Agglomerative
    agg_labels = pipeline.fit_agglomerative(features)
    results['agglomerative'] = evaluate_clustering(features, agg_labels, true_labels)
    
    # PCA + K-Means baseline
    pca = PCA(n_components=50, random_state=42)
    n_components = min(32, features.shape[1])  # Use 32 or fewer components
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    pca_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pca_labels = pca_kmeans.fit_predict(features_pca)
    results['pca_kmeans'] = evaluate_clustering(features, pca_labels, true_labels)
    
    return results

def print_metrics(metrics_dict, title="Clustering Results"):
    """Print metrics in formatted way"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    for metric, value in metrics_dict.items():
        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:<25}: {value:.4f}")
    
    print(f"{'='*60}")

def save_results(results, output_dir):
    """Save clustering results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(results).T
    
    # Save to CSV
    metrics_df.to_csv(output_dir / 'clustering_metrics.csv')
    
    print(f"Results saved to {output_dir}/clustering_metrics.csv")
    return metrics_df