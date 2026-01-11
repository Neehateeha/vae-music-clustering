import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# VARIATIONAL AUTOENCODER (VAE) MODEL
# =============================================================================

class VAE(nn.Module):
    """
    Variational Autoencoder for music feature compression and clustering
    
    Structure:
    - Encoder: Compresses 13 MFCC features into 5-dimensional latent space
    - Decoder: Reconstructs original features from latent space
    - KL Divergence: Ensures latent space is well-structured
    """
    
    def __init__(self, input_dim=13, latent_dim=5):
        """
        Args:
            input_dim: Number of input features (13 MFCC)
            latent_dim: Size of latent space (smaller = more compression)
        """
        super(VAE, self).__init__()
        
        # ENCODER: Compress input to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),      # 13 -> 64
            nn.ReLU(),
            nn.Linear(64, 32),             # 64 -> 32
            nn.ReLU(),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(32, latent_dim)           # Mean of latent distribution
        self.fc_logvar = nn.Linear(32, latent_dim)       # Log variance
        
        # DECODER: Reconstruct from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),     # 5 -> 32
            nn.ReLU(),
            nn.Linear(32, 64),             # 32 -> 64
            nn.ReLU(),
            nn.Linear(64, input_dim),      # 64 -> 13 (reconstruct original)
        )
    
    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: Sample from latent distribution
        Allows backpropagation through sampling
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruct input"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full VAE forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# =============================================================================
# VAE LOSS FUNCTION
# =============================================================================

def vae_loss(recon_x, x, mu, logvar):
    """
    VAE Loss = Reconstruction Loss + KL Divergence
    
    Reconstruction Loss: How well can we reconstruct the input?
    KL Divergence: How close is latent distribution to standard normal?
    """
    # Reconstruction loss (how well we reconstruct)
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence (regularization term)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return mse_loss + kl_loss


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_vae(model, train_loader, epochs=50, learning_rate=1e-3, device='cpu'):
    """
    Train the VAE model
    
    Args:
        model: VAE model
        train_loader: DataLoader with training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: 'cpu' or 'cuda' (GPU)
    
    Returns:
        losses: List of loss values per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Final Loss: {losses[-1]:.4f}")
    
    return losses


# =============================================================================
# EXTRACT LATENT FEATURES
# =============================================================================

def get_latent_features(model, data_loader, device='cpu'):
    """
    Extract latent representations from trained VAE
    These are the compressed features we'll use for clustering
    
    Args:
        model: Trained VAE model
        data_loader: DataLoader with data
        device: 'cpu' or 'cuda'
    
    Returns:
        latent_features: numpy array of shape (num_samples, latent_dim)
        file_names: List of file names
    """
    model.to(device)
    model.eval()
    
    latent_features_list = []
    file_names_list = []
    
    with torch.no_grad():
        for batch_idx, (x,) in enumerate(data_loader):
            x = x.to(device)
            mu, _ = model.encode(x)
            latent_features_list.append(mu.cpu().numpy())
    
    latent_features = np.concatenate(latent_features_list, axis=0)
    
    return latent_features, file_names_list


# =============================================================================
# MAIN: TRAIN VAE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING VARIATIONAL AUTOENCODER (VAE)")
    print("=" * 70)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\nLoading audio features from results/audio_features.csv...")
    df = pd.read_csv('results/audio_features.csv')
    
    file_names = df['file_name'].values
    features = df.drop('file_name', axis=1).values
    
    print(f"Loaded {len(file_names)} songs")
    print(f"Features shape: {features.shape}")
    
    # Normalize features (important!)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(features_normalized)
    file_names_tensor = file_names  # Keep as list
    
    # Create simple dataset (just features)
    dataset = TensorDataset(X)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create VAE model
    input_dim = features.shape[1]  # 13 MFCC features
    latent_dim = 5                 # Compress to 5 dimensions
    
    print(f"\nCreating VAE model...")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Latent dimensions: {latent_dim}")
    
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    
    # Train VAE
    print(f"\nStarting training for 50 epochs...")
    print("-" * 70)
    losses = train_vae(model, train_loader, epochs=50, learning_rate=1e-3, device=device)
    print("-" * 70)
    
    # Extract latent features
    print(f"\nExtracting latent representations...")
    latent_features, _ = get_latent_features(model, train_loader, device=device)
    
    print(f"Latent features shape: {latent_features.shape}")
    
    # Save trained model
    torch.save(model.state_dict(), 'results/vae_model.pth')
    print(f"\n✓ Model saved to results/vae_model.pth")
    
    # Save latent features
    np.save('results/latent_features.npy', latent_features)
    print(f"✓ Latent features saved to results/latent_features.npy")
    
    # Save scaler for later use
    import pickle
    with open('results/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to results/scaler.pkl")
    
    print("\n" + "=" * 70)
    print("VAE TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Cluster the latent features (K-Means)")
    print("2. Visualize with t-SNE")
    print("3. Compare with baseline (PCA + K-Means)")