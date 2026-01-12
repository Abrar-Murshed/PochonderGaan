# 🎵 VAE Music Genre Clustering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NeurIPS Paper](https://img.shields.io/badge/📄-NeurIPS%20Paper-blueviolet)](paper/neurips_2024.pdf)

> **Unsupervised music genre clustering using Variational Autoencoders (VAE) for hybrid language music analysis**


## ✨ Features
- **Multiple VAE Architectures**: Base VAE, Beta-VAE (β=4.0), and Conditional VAE
- **Audio Feature Extraction**: MFCC, Chroma, Spectral, and Temporal features
- **Comprehensive Evaluation**: 6 clustering metrics including Silhouette Score, ARI, and NMI
- **Visualization**: t-SNE and UMAP projections of latent spaces
- **Multi-Algorithm Comparison**: K-Means, Agglomerative Clustering, and DBSCAN
- **Baseline Comparisons**: PCA, Raw Features, and Autoencoder baselines
- **Academic Paper**: Complete NeurIPS-style research paper

## 🎯 Project Overview
This project implements an unsupervised learning pipeline for music genre clustering using Variational Autoencoders (VAE). The goal is to learn compact latent representations from audio features that enable better clustering compared to traditional methods.

### Tasks Completed
| Task | Description | Status |
|------|-------------|---------|
| ✅ **Easy** | Basic VAE + K-Means on GTZAN | Complete |
| ✅ **Medium** | Enhanced VAE with CNN + Hybrid Features | Complete |
| ✅ **Hard** | Beta-VAE/CVAE + Multi-modal Clustering | Complete |

## 📊 Results
### Key Performance Metrics
| Model | Silhouette Score ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ | ARI ↑ | NMI ↑ |
|-------|-------------------|---------------------|------------------|-------|-------|
| **Base VAE** | **0.155** | **25.32** | **1.39** | 0.004 | 0.213 |
| Beta-VAE | 0.103 | 11.97 | 1.59 | 0.009 | 0.206 |
| PCA Baseline | 0.013 | 2.18 | 3.21 | 0.007 | 0.207 |
| Raw Features | 0.010 | 2.05 | 3.15 | 0.037 | 0.260 |

## ⚙️ Installation
### Prerequisites
- Python 3.8 or higher
- Git
- 8GB+ RAM (recommended)

### Installation Steps
```bash
git clone https://github.com/Abrar-Murshed/PochonderGaan.git
cd PochonderGaan
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
