import numpy as np
import pandas as pd
import os

os.makedirs('results', exist_ok=True)

# Generate 100 synthetic songs with 13 MFCC features each
num_songs = 100
num_features = 13

np.random.seed(42)
features = np.random.randn(num_songs, num_features) * 10 - 20

song_names = [f'song_{i:03d}.wav' for i in range(1, num_songs + 1)]

df = pd.DataFrame(features, columns=[str(i) for i in range(num_features)])
df.insert(0, 'file_name', song_names)

df.to_csv('results/audio_features.csv', index=False)

print("=" * 60)
print("✓ 100 SYNTHETIC SONGS GENERATED!")
print("=" * 60)
print(f"✓ Saved to results/audio_features.csv")
print(f"✓ Ready for VAE training!")
print("\nNote: You can replace with real data later")
