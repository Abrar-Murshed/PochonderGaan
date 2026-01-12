"""
Visualization and evaluation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_tsne(features, labels, true_labels=None, save_path=None, title="t-SNE Visualization"):
    """Create t-SNE plot"""
    
    print("Creating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, 
                            figsize=(15, 6) if true_labels is not None else (10, 8))
    
    if true_labels is not None:
        # Predicted clusters
        scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=labels, cmap='tab10', alpha=0.6, s=50)
        axes[0].set_title(f'{title} - Predicted Clusters')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # True labels
        scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=true_labels, cmap='tab10', alpha=0.6, s=50)
        axes[1].set_title(f'{title} - True Labels')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=axes[1], label='Genre')
    else:
        scatter = axes.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=labels, cmap='tab10', alpha=0.6, s=50)
        axes.set_title(title)
        axes.set_xlabel('t-SNE 1')
        axes.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes, label='Cluster')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    plt.show()
    return features_2d

def plot_umap(features, labels, true_labels=None, save_path=None):
    """Create UMAP plot"""
    
    print("Creating UMAP visualization...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    features_2d = reducer.fit_transform(features)
    
    fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, 
                            figsize=(15, 6) if true_labels is not None else (10, 8))
    
    if true_labels is not None:
        # Predicted clusters
        scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=labels, cmap='tab10', alpha=0.6, s=50)
        axes[0].set_title('UMAP - Predicted Clusters')
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        plt.colorbar(scatter1, ax=axes[0], label='Cluster')
        
        # True labels
        scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=true_labels, cmap='tab10', alpha=0.6, s=50)
        axes[1].set_title('UMAP - True Labels')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        plt.colorbar(scatter2, ax=axes[1], label='Genre')
    else:
        scatter = axes.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=labels, cmap='tab10', alpha=0.6, s=50)
        axes.set_title('UMAP Visualization')
        axes.set_xlabel('UMAP 1')
        axes.set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=axes, label='Cluster')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved UMAP plot to {save_path}")
    
    plt.show()
    return features_2d

def plot_metrics_comparison(results_dict, save_path=None):
    """Plot comparison of metrics across methods"""
    
    methods = list(results_dict.keys())
    
    metrics_to_plot = ['silhouette_score', 'calinski_harabasz_score', 
                      'davies_bouldin_score', 'adjusted_rand_score',
                      'normalized_mutual_info', 'cluster_purity']
    
    available_metrics = []
    for metric in metrics_to_plot:
        if all(metric in results_dict[m] for m in methods):
            available_metrics.append(metric)
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = plt.cm.Set3.colors
    
    for idx, metric in enumerate(available_metrics):
        values = [results_dict[m][metric] for m in methods]
        
        bars = axes[idx].bar(range(len(methods)), values, color=colors[:len(methods)])
        axes[idx].set_xticks(range(len(methods)))
        axes[idx].set_xticklabels(methods, rotation=45, ha='right')
        axes[idx].set_ylabel('Score')
        axes[idx].set_title(metric.replace('_', ' ').title())
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Highlight best value
        if metric == 'davies_bouldin_score':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_facecolor('gold')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Clustering Methods Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics comparison plot to {save_path}")
    
    plt.show()

def create_visualization_report(features, pred_labels, true_labels, 
                               results_dict, output_dir='results/visualization'):
    """Create comprehensive visualization report"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating visualization report...")
    
    # t-SNE plot
    plot_tsne(features, pred_labels, true_labels, 
             save_path=output_dir / 'tsne_visualization.png')
    
    # UMAP plot
    plot_umap(features, pred_labels, true_labels,
             save_path=output_dir / 'umap_visualization.png')
    
    # Metrics comparison
    plot_metrics_comparison(results_dict,
                           save_path=output_dir / 'metrics_comparison.png')
    
    print(f"Visualization report saved to {output_dir}")