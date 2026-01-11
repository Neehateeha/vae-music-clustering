import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path

def extract_mfcc(audio_path, n_mfcc=13, sr=22050):
    """
    Extract MFCC (Mel-Frequency Cepstral Coefficients) from audio file
    
    Parameters:
    - audio_path: path to audio file (.wav, .mp3, etc)
    - n_mfcc: number of MFCC coefficients (default 13)
    - sr: sample rate (default 22050 Hz)
    
    Returns:
    - numpy array of shape (n_mfcc,) - averaged across time
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def extract_spectrogram(audio_path, sr=22050):
    """
    Extract Mel-Spectrogram features from audio file
    
    Parameters:
    - audio_path: path to audio file
    - sr: sample rate
    
    Returns:
    - numpy array of shape (128,) - averaged across time
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_mean = np.mean(spec_db, axis=1)
        return spec_mean
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def extract_combined_features(audio_path, n_mfcc=13, sr=22050):
    """
    Extract both MFCC and Spectrogram features and combine them
    
    Returns:
    - numpy array combining both features (13 + 128 = 141 features)
    """
    mfcc = extract_mfcc(audio_path, n_mfcc, sr)
    spec = extract_spectrogram(audio_path, sr)
    
    if mfcc is not None and spec is not None:
        combined = np.concatenate([mfcc, spec])
        return combined
    return None


def process_audio_folder(audio_dir, feature_type='mfcc', n_mfcc=13):
    """
    Process all audio files in a folder and extract features
    
    Parameters:
    - audio_dir: path to folder containing audio files
    - feature_type: 'mfcc', 'spectrogram', or 'combined'
    - n_mfcc: number of MFCC coefficients
    
    Returns:
    - features: numpy array of shape (num_files, num_features)
    - file_names: list of corresponding file names
    """
    features = []
    file_names = []
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        
        if os.path.isfile(file_path) and Path(file_path).suffix.lower() in audio_extensions:
            print(f"Processing: {file}")
            
            if feature_type == 'mfcc':
                feat = extract_mfcc(file_path, n_mfcc=n_mfcc)
            elif feature_type == 'spectrogram':
                feat = extract_spectrogram(file_path)
            elif feature_type == 'combined':
                feat = extract_combined_features(file_path, n_mfcc=n_mfcc)
            else:
                print("Unknown feature type")
                continue
            
            if feat is not None:
                features.append(feat)
                file_names.append(file)
    
    if len(features) == 0:
        print("No audio files found!")
        return None, None
    
    return np.array(features), file_names


def save_features_csv(features, file_names, output_path):
    """
    Save extracted features to CSV file for later use
    
    Parameters:
    - features: numpy array
    - file_names: list of file names
    - output_path: where to save the CSV
    """
    df = pd.DataFrame(features)
    df.insert(0, 'file_name', file_names)
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")


def load_features_csv(csv_path):
    """
    Load previously saved features from CSV
    
    Returns:
    - features: numpy array
    - file_names: list of file names
    """
    df = pd.read_csv(csv_path)
    file_names = df['file_name'].values.tolist()
    features = df.drop('file_name', axis=1).values
    
    return features, file_names


if __name__ == "__main__":
    audio_folder = "data/audio"
    output_csv = "results/audio_features.csv"
    
    if not os.path.exists(audio_folder):
        print(f"Error: {audio_folder} folder not found!")
        print("Please make sure you have audio files in data/audio/")
    else:
        print("=" * 50)
        print("EXTRACTING AUDIO FEATURES")
        print("=" * 50)
        
        features, file_names = process_audio_folder(audio_folder, feature_type='mfcc', n_mfcc=13)
        
        if features is not None:
            print(f"\nSuccess! Extracted {len(file_names)} audio files")
            print(f"Feature shape: {features.shape}")
            print(f"Features per file: {features.shape[1]}")
            
            save_features_csv(features, file_names, output_csv)
            print("\nFeatures saved for later use!")
        else:
            print("Failed to extract features!")