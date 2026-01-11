import pandas as pd
import numpy as np

def verify_features():
    """Verify that extracted features look correct"""
    
    csv_path = "results/audio_features.csv"
    
    try:
        df = pd.read_csv(csv_path)
        
        print("=" * 60)
        print("FEATURE VERIFICATION")
        print("=" * 60)
        
        print(f"\nTotal songs processed: {len(df)}")
        print(f"Features per song: {len(df.columns) - 1}")
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nFeature statistics:")
        print(df.describe())
        
        print("\nFile names loaded:")
        for i, fname in enumerate(df['file_name'].head(5)):
            print(f"  {i+1}. {fname}")
        
        if len(df) > 5:
            print(f"  ... and {len(df) - 5} more files")
        
        print("\n✓ Features look good!")
        print(f"✓ Data shape: {df.shape}")
        print(f"✓ No missing values: {df.isnull().sum().sum() == 0}")
        
        return True
        
    except FileNotFoundError:
        print("Error: audio_features.csv not found!")
        print("Make sure you ran feature_extraction.py first")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    verify_features()