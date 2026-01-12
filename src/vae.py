"""
VAE Models for Music Clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseVAE(nn.Module):
    """Basic Variational Autoencoder"""
    
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_dims = list(reversed(hidden_dims))
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss

class BetaVAE(BaseVAE):
    """Beta-VAE for disentangled representations"""
    
    def __init__(self, input_dim, latent_dim=64, hidden_dims=[512, 256, 128], beta=4.0):
        super().__init__(input_dim, latent_dim, hidden_dims)
        self.beta = beta
    
    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

def train_vae(model, train_loader, optimizer, device, n_epochs=50):
    """Train VAE model"""
    model.train()
    losses = []
    
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar, z = model(x)
            loss, recon_loss, kl_loss = model.loss_function(recon_x, x, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
    
    return losses

def extract_latent_features(model, data_loader, device):
    """Extract latent features from trained VAE"""
    model.eval()
    latent_features = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            mu, _ = model.encode(x)
            latent_features.append(mu.cpu().numpy())
    
    return np.concatenate(latent_features, axis=0)