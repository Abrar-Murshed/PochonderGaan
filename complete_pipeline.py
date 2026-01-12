"""
Complete Pipeline for VAE Music Clustering
Runs all tasks: Easy, Medium, Hard
"""

import sys
import os
sys.path.append('src')

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse
import json
import pandas as pd

def load_processed_data():
    """Load processed data"""
    data_dir = Path('data/processed')
    
    # Load GTZAN audio data
    audio_features = np.load(data_dir / 'gtzan_features.npy')
    audio_labels = np.load(data_dir / 'gtzan_labels.npy')
    
    # Load lyrics embeddings if available
    lyrics_embeddings = None
    lyrics_path = data_dir / 'lyrics_embeddings.npy'
    if lyrics_path.exists():
        lyrics_embeddings = np.load(lyrics_path)
    
    return audio_features, audio_labels, lyrics_embeddings

def easy_task():
    """Easy Task: Basic clustering on audio features"""
    print("="*80)
    print("EASY TASK: BASIC AUDIO CLUSTERING")
    print("="*80)
    
    # Load data
    audio_features, audio_labels, _ = load_processed_data()
    
    print(f"Audio features shape: {audio_features.shape}")
    print(f"Labels shape: {audio_labels.shape}")
    
    # Run baseline clustering
    from src.clustering import run_baseline_experiment, print_metrics, save_results
    
    results = run_baseline_experiment(audio_features, audio_labels, n_clusters=10)
    
    # Print results
    for method, metrics in results.items():
        print_metrics(metrics, f"{method.upper()} Results")
    
    # Save results
    output_dir = Path('results/easy')
    save_results(results, output_dir)
    
    # Create visualizations
    from src.clustering import ClusteringPipeline
    pipeline = ClusteringPipeline(n_clusters=10)
    pred_labels = pipeline.fit_kmeans(audio_features)
    
    from src.evaluation import create_visualization_report
    create_visualization_report(audio_features, pred_labels, audio_labels,
                               results, output_dir / 'visualization')
    
    print(f"\n✓ Easy task completed! Results saved to {output_dir}")
    return results

def medium_task():
    """Medium Task: VAE + Clustering"""
    print("="*80)
    print("MEDIUM TASK: VAE FEATURE EXTRACTION + CLUSTERING")
    print("="*80)
    
    # Load data
    audio_features, audio_labels, lyrics_embeddings = load_processed_data()
    
    print(f"Audio features shape: {audio_features.shape}")
    print(f"Lyrics embeddings: {'Available' if lyrics_embeddings is not None else 'Not available'}")
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(audio_features)
    
    # Prepare data loader
    dataset = TensorDataset(torch.FloatTensor(features_normalized))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train VAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from src.vae import BaseVAE, train_vae, extract_latent_features
    
    input_dim = audio_features.shape[1]
    model = BaseVAE(input_dim=input_dim, latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining VAE...")
    train_losses = train_vae(model, train_loader, optimizer, device, n_epochs=50)
    
    # Extract latent features
    print("\nExtracting latent features...")
    latent_features = extract_latent_features(model, train_loader, device)
    print(f"Latent features shape: {latent_features.shape}")
    
    # Run clustering on latent features
    from src.clustering import run_baseline_experiment, save_results
    
    results = run_baseline_experiment(latent_features, audio_labels, n_clusters=10)
    
    # Compare with baseline
    from src.clustering import run_baseline_experiment as run_baseline
    baseline_results = run_baseline(audio_features, audio_labels, n_clusters=10)
    
    # Combine results
    all_results = {
        'baseline_raw': baseline_results['kmeans'],
        'vae_latent': results['kmeans']
    }
    
    # Save results
    output_dir = Path('results/medium')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    pd.DataFrame(all_results).T.to_csv(output_dir / 'comparison_metrics.csv')
    
    # Save latent features
    np.save(output_dir / 'latent_features.npy', latent_features)
    
    # Save model
    torch.save(model.state_dict(), output_dir / 'vae_model.pth')
    
    # Create visualization
    from src.clustering import ClusteringPipeline
    pipeline = ClusteringPipeline(n_clusters=10)
    pred_labels = pipeline.fit_kmeans(latent_features)
    
    from src.evaluation import create_visualization_report
    create_visualization_report(latent_features, pred_labels, audio_labels,
                               {'vae_kmeans': results['kmeans']}, 
                               output_dir / 'visualization')
    
    print(f"\n✓ Medium task completed! Results saved to {output_dir}")
    return all_results

def hard_task():
    """Hard Task: Advanced VAE + Hybrid Features"""
    print("="*80)
    print("HARD TASK: ADVANCED VAE + HYBRID FEATURES")
    print("="*80)
    
    # Load all data
    audio_features, audio_labels, lyrics_embeddings = load_processed_data()
    
    print(f"Audio features shape: {audio_features.shape}")
    
    if lyrics_embeddings is None:
        print("Warning: Lyrics embeddings not found. Using audio only.")
        features = audio_features
    else:
        # Align sizes (take minimum)
        min_size = min(len(audio_features), len(lyrics_embeddings))
        audio_features = audio_features[:min_size]
        lyrics_embeddings = lyrics_embeddings[:min_size]
        audio_labels = audio_labels[:min_size]
        
        # Combine features
        features = np.concatenate([audio_features, lyrics_embeddings], axis=1)
        print(f"Hybrid features shape: {features.shape}")
    
    # Normalize
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Train multiple VAEs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from src.vae import BaseVAE, BetaVAE, train_vae, extract_latent_features
    
    input_dim = features.shape[1]
    
    # Prepare data loader
    dataset = TensorDataset(torch.FloatTensor(features_normalized))
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    results_all = {}
    
    # 1. Base VAE
    print("\n1. Training Base VAE...")
    base_vae = BaseVAE(input_dim=input_dim, latent_dim=32).to(device)
    optimizer_base = optim.Adam(base_vae.parameters(), lr=0.001)
    train_vae(base_vae, train_loader, optimizer_base, device, n_epochs=50)
    base_features = extract_latent_features(base_vae, train_loader, device)
    
    # 2. Beta VAE
    print("\n2. Training Beta VAE...")
    beta_vae = BetaVAE(input_dim=input_dim, latent_dim=64).to(device)
    optimizer_beta = optim.Adam(beta_vae.parameters(), lr=0.001)
    train_vae(beta_vae, train_loader, optimizer_beta, device, n_epochs=50)
    beta_features = extract_latent_features(beta_vae, train_loader, device)
    
    # Cluster and evaluate each
    from src.clustering import ClusteringPipeline, evaluate_clustering
    
    pipeline = ClusteringPipeline(n_clusters=10)
    
    # Base VAE results
    base_labels = pipeline.fit_kmeans(base_features)
    results_all['base_vae'] = evaluate_clustering(base_features, base_labels, audio_labels)
    
    # Beta VAE results
    beta_labels = pipeline.fit_kmeans(beta_features)
    results_all['beta_vae'] = evaluate_clustering(beta_features, beta_labels, audio_labels)
    
    # Raw features baseline
    raw_labels = pipeline.fit_kmeans(features)
    results_all['raw_features'] = evaluate_clustering(features, raw_labels, audio_labels)
    
    # Save results
    output_dir = Path('results/hard')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(results_all).T.to_csv(output_dir / 'all_results.csv')
    
    # Save models and features
    torch.save(base_vae.state_dict(), output_dir / 'base_vae.pth')
    torch.save(beta_vae.state_dict(), output_dir / 'beta_vae.pth')
    np.save(output_dir / 'base_vae_features.npy', base_features)
    np.save(output_dir / 'beta_vae_features.npy', beta_features)
    
    # Create comprehensive visualization
    from src.evaluation import plot_metrics_comparison
    
    plot_metrics_comparison(results_all, 
                           save_path=output_dir / 'methods_comparison.png')
    
    # Create t-SNE for best method
    best_method = max(results_all.items(), key=lambda x: x[1]['silhouette_score'])[0]
    
    if best_method == 'base_vae':
        best_features = base_features
        best_labels = base_labels
    elif best_method == 'beta_vae':
        best_features = beta_features
        best_labels = beta_labels
    else:
        best_features = features
        best_labels = raw_labels
    
    from src.evaluation import plot_tsne
    plot_tsne(best_features, best_labels, audio_labels,
             save_path=output_dir / f'best_method_{best_method}_tsne.png',
             title=f'Best Method: {best_method.upper()}')
    
    print(f"\nBest method: {best_method}")
    print(f"Silhouette Score: {results_all[best_method]['silhouette_score']:.4f}")
    print(f"Adjusted Rand Index: {results_all[best_method]['adjusted_rand_score']:.4f}")
    
    print(f"\n✓ Hard task completed! Results saved to {output_dir}")
    return results_all

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='VAE Music Clustering Pipeline')
    parser.add_argument('--task', type=str, required=True,
                       choices=['easy', 'medium', 'hard', 'all'],
                       help='Task to run')
    
    args = parser.parse_args()
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    if args.task == 'easy':
        results = easy_task()
    elif args.task == 'medium':
        results = medium_task()
    elif args.task == 'hard':
        results = hard_task()
    elif args.task == 'all':
        print("Running all tasks sequentially...\n")
        easy_task()
        print("\n" + "="*80 + "\n")
        medium_task()
        print("\n" + "="*80 + "\n")
        hard_task()
        print("\n" + "="*80)
        print("ALL TASKS COMPLETED!")
        print("="*80)
        return
    
    print("\n" + "="*80)
    print(f"TASK {args.task.upper()} COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()