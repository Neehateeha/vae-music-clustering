import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle

# =============================================================================
# CONVOLUTIONAL VARIATIONAL AUTOENCODER (Conv-VAE)
# =============================================================================

class ConvVAE(nn.Module):
    """
    Convolutional VAE for better feature extraction
    
    Better than basic VAE because:
    - Convolutional layers capture local patterns
    - Better for sequential/temporal data like audio features
    """
    
    def __init__(self, input_dim=13, latent_dim=5):
        super(ConvVAE, self).__init__()
        
        # Reshape input to (batch_size, 1, input_dim) for Conv1d
        self.input_dim = input_dim
        
        # ENCODER: Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),  # 1 -> 32 channels
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),  # 32 -> 64 channels
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1), # 64 -> 128 channels
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling -> (batch, 128, 1)
        )
        
        # Flatten to (batch, 128)
        self.fc_encode = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # DECODER
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # Decoder: Transpose convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
        )
    
    def encode(self, x):
        """Encode to latent space"""
        # x shape: (batch, 13) -> reshape to (batch, 1, 13)
        x = x.unsqueeze(1)
        h = self.encoder(x)  # (batch, 128, 1)
        h = h.squeeze(-1)    # (batch, 128)
        h = self.fc_encode(h)  # (batch, 64)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode from latent space"""
        h = self.fc_decode(z)  # (batch, 128)
        h = h.unsqueeze(-1)    # (batch, 128, 1)
        # Expand to match input size
        h = h.expand(-1, -1, self.input_dim)  # (batch, 128, 13)
        
        x_recon = self.decoder(h)  # (batch, 1, 13)
        x_recon = x_recon.squeeze(1)  # (batch, 13)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# =============================================================================
# VAE LOSS WITH BETA WEIGHTING (Beta-VAE)
# =============================================================================

def beta_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE Loss with beta weighting
    beta > 1: Encourages more disentangled representations
    """
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + beta * kl_loss


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_vae(model, train_loader, epochs=50, learning_rate=1e-3, beta=1.0, device='cpu'):
    """Train Conv-VAE with beta weighting"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = beta_vae_loss(recon_x, x, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    print(f"Training completed! Final Loss: {losses[-1]:.4f}")
    return losses


# =============================================================================
# EXTRACT LATENT FEATURES
# =============================================================================

def get_latent_features(model, data_loader, device='cpu'):
    """Extract latent representations"""
    model.to(device)
    model.eval()
    
    latent_features_list = []
    
    with torch.no_grad():
        for batch_idx, (x,) in enumerate(data_loader):
            x = x.to(device)
            mu, _ = model.encode(x)
            latent_features_list.append(mu.cpu().numpy())
    
    return np.concatenate(latent_features_list, axis=0)


# =============================================================================
# MAIN: TRAIN CONV-VAE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING CONVOLUTIONAL VAE (MEDIUM TASK)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading audio features...")
    df = pd.read_csv('results/audio_features.csv')
    file_names = df['file_name'].values
    features = df.drop('file_name', axis=1).values
    
    print(f"Loaded {len(file_names)} songs")
    print(f"Features shape: {features.shape}")
    
    # Normalize
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Create dataset
    X = torch.FloatTensor(features_normalized)
    dataset = TensorDataset(X)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create and train Conv-VAE
    input_dim = features.shape[1]
    latent_dim = 5
    
    print(f"\nCreating Convolutional VAE...")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Latent dimensions: {latent_dim}")
    
    model = ConvVAE(input_dim=input_dim, latent_dim=latent_dim)
    
    print(f"\nTraining Conv-VAE...")
    print("-" * 70)
    losses = train_vae(model, train_loader, epochs=50, learning_rate=1e-3, beta=1.0, device=device)
    print("-" * 70)
    
    # Extract latent features
    print(f"\nExtracting latent features from Conv-VAE...")
    latent_features = get_latent_features(model, train_loader, device=device)
    print(f"Latent features shape: {latent_features.shape}")
    
    # Save results
    torch.save(model.state_dict(), 'results/conv_vae_model.pth')
    print(f"\n✓ Conv-VAE model saved to results/conv_vae_model.pth")
    
    np.save('results/conv_latent_features.npy', latent_features)
    print(f"✓ Latent features saved to results/conv_latent_features.npy")
    
    with open('results/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n" + "=" * 70)
    print("CONVOLUTIONAL VAE TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext: Run advanced clustering with Conv-VAE features")