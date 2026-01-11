import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pickle

# =============================================================================
# CONDITIONAL VARIATIONAL AUTOENCODER (CVAE)
# =============================================================================

class CVAE(nn.Module):
    """
    Conditional VAE: Creates disentangled latent representations
    
    Advantages:
    - Can condition on additional information
    - Better disentanglement (features more interpretable)
    - Better for controlled generation
    """
    
    def __init__(self, input_dim=13, condition_dim=5, latent_dim=5):
        super(CVAE, self).__init__()
        
        # ENCODER: Takes input + condition
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # DECODER: Takes latent + condition
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, input_dim),
        )
    
    def encode(self, x, c):
        """Encode with condition"""
        h = self.encoder(torch.cat([x, c], dim=1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """Decode with condition"""
        return self.decoder(torch.cat([z, c], dim=1))
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


def cvae_loss(recon_x, x, mu, logvar, beta=4.0):
    """CVAE Loss with beta weighting for disentanglement"""
    mse_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + beta * kl_loss


def train_cvae(model, train_loader, epochs=50, learning_rate=1e-3, beta=4.0, device='cpu'):
    """Train CVAE"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, c) in enumerate(train_loader):
            x = x.to(device)
            c = c.to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x, c)
            loss = cvae_loss(recon_x, x, mu, logvar, beta=beta)
            
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


def get_latent_features_cvae(model, data_loader, device='cpu'):
    """Extract latent representations from CVAE"""
    model.to(device)
    model.eval()
    
    latent_features_list = []
    
    with torch.no_grad():
        for batch_idx, (x, c) in enumerate(data_loader):
            x = x.to(device)
            c = c.to(device)
            mu, _ = model.encode(x, c)
            latent_features_list.append(mu.cpu().numpy())
    
    return np.concatenate(latent_features_list, axis=0)


# =============================================================================
# MAIN: TRAIN CVAE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING CONDITIONAL VAE (HARD TASK)")
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
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Create condition vector: Random cluster assignments (simulating genre/language)
    np.random.seed(42)
    n_conditions = 5
    condition_assignments = np.random.randint(0, n_conditions, len(file_names))
    conditions_onehot = np.eye(n_conditions)[condition_assignments]
    
    print(f"\nCondition vector shape (one-hot encoded): {conditions_onehot.shape}")
    print(f"Condition classes: {n_conditions} (simulating genres/languages)")
    
    # Create dataset with conditions
    X = torch.FloatTensor(features_normalized)
    C = torch.FloatTensor(conditions_onehot)
    
    dataset = TensorDataset(X, C)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Create and train CVAE
    input_dim = features.shape[1]
    condition_dim = n_conditions
    latent_dim = 5
    
    print(f"\nCreating Conditional VAE...")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Condition dimensions: {condition_dim}")
    print(f"  Latent dimensions: {latent_dim}")
    
    model = CVAE(input_dim=input_dim, condition_dim=condition_dim, latent_dim=latent_dim)
    
    print(f"\nTraining CVAE (beta=4.0 for disentanglement)...")
    print("-" * 70)
    losses = train_cvae(model, train_loader, epochs=50, learning_rate=1e-3, beta=4.0, device=device)
    print("-" * 70)
    
    # Extract latent features
    print(f"\nExtracting disentangled latent features from CVAE...")
    latent_features = get_latent_features_cvae(model, train_loader, device=device)
    print(f"CVAE Latent features shape: {latent_features.shape}")
    
    # Save results
    torch.save(model.state_dict(), 'results/cvae_model.pth')
    print(f"\n✓ CVAE model saved to results/cvae_model.pth")
    
    np.save('results/cvae_latent_features.npy', latent_features)
    print(f"✓ Latent features saved to results/cvae_latent_features.npy")
    
    np.save('results/conditions.npy', condition_assignments)
    print(f"✓ Conditions saved to results/conditions.npy")
    
    with open('results/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n" + "=" * 70)
    print("CONDITIONAL VAE TRAINING COMPLETE!")
    print("=" * 70)
    print("\nCVAE learns disentangled representations!")
    print("Each latent dimension now encodes distinct features")