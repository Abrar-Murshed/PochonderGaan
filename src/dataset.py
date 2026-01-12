import numpy as np
import pandas as pd
import os
from pathlib import Path
import librosa
import librosa.display
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def extract_audio_features(audio_path, sr=22050, duration=30):
    """
    Extract features from an audio file.
    Returns: MFCCs, chroma, melspectrogram, spectral features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Extract features
        features = []
        
        # MFCCs (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        features.extend(mfcc_mean)
        
        # Chroma features (12)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean)
        
        # Melspectrogram (128)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel, axis=1)
        # Take first 20 for dimensionality reduction
        features.extend(mel_mean[:20])
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        features.extend([spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate])
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        return np.array(features)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_audio_data():
    """Process all audio files from genre folders"""
    print("Processing audio data...")
    
    # Define paths
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Genre folders (based on your structure)
    genre_folders = [
        "blues", "classical", "country", "disco", 
        "hiphop", "jazz", "metal", "pop", "reggae", "rock"
    ]
    
    all_features = []
    all_labels = []
    all_metadata = []
    
    # Process each genre
    for genre_idx, genre in enumerate(genre_folders):
        genre_path = data_dir / genre
        
        # Check if genre folder exists
        if not genre_path.exists():
            print(f"Warning: {genre_path} not found. Trying audio/{genre}...")
            genre_path = data_dir / "audio" / genre
        
        if not genre_path.exists():
            print(f"Warning: {genre} folder not found. Skipping...")
            continue
        
        # Get audio files
        audio_files = list(genre_path.glob("*.wav")) + list(genre_path.glob("*.mp3"))
        
        if not audio_files:
            print(f"No audio files found in {genre_path}")
            continue
        
        print(f"Processing {genre} ({len(audio_files)} files)...")
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc=genre):
            # Extract features
            features = extract_audio_features(str(audio_file))
            
            if features is not None:
                all_features.append(features)
                all_labels.append(genre_idx)
                
                # Store metadata
                all_metadata.append({
                    'filename': audio_file.name,
                    'filepath': str(audio_file),
                    'genre': genre,
                    'genre_id': genre_idx,
                    'duration': librosa.get_duration(filename=str(audio_file)) if hasattr(librosa, 'get_duration') else 0
                })
    
    # Convert to arrays
    if not all_features:
        print("Error: No audio features extracted!")
        return False
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(all_metadata)
    
    # Save processed data
    print(f"\nSaving processed data...")
    print(f"Features shape: {all_features.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Metadata shape: {metadata_df.shape}")
    
    np.save(processed_dir / "gtzan_features.npy", all_features)
    np.save(processed_dir / "gtzan_labels.npy", all_labels)
    metadata_df.to_csv(processed_dir / "gtzan_metadata.csv", index=False)
    
    print(f"Saved to {processed_dir}/")
    
    return True

def process_lyrics_data():
    """Process lyrics data if available"""
    print("\nProcessing lyrics data...")
    
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    
    # Check for lyrics CSV files
    lyrics_files = list(data_dir.glob("*lyrics*.csv")) + list(data_dir.glob("lyrics*"))
    
    for lyrics_file in lyrics_files:
        try:
            print(f"Found lyrics file: {lyrics_file}")
            lyrics_df = pd.read_csv(lyrics_file)
            
            # Basic cleaning
            lyrics_df_clean = lyrics_df.dropna(subset=['lyrics'] if 'lyrics' in lyrics_df.columns else [])
            
            # Save cleaned lyrics
            output_path = processed_dir / f"{lyrics_file.stem}_cleaned.csv"
            lyrics_df_clean.to_csv(output_path, index=False)
            print(f"Saved cleaned lyrics to {output_path}")
            
        except Exception as e:
            print(f"Error processing {lyrics_file}: {e}")
    
    # Check for artists data
    artists_files = list(data_dir.glob("*artists*.csv")) + list(data_dir.glob("artists*"))
    
    for artists_file in artists_files:
        try:
            print(f"Found artists file: {artists_file}")
            artists_df = pd.read_csv(artists_file)
            
            # Save artists data
            output_path = processed_dir / f"{artists_file.stem}_cleaned.csv"
            artists_df.to_csv(output_path, index=False)
            print(f"Saved artists data to {output_path}")
            
        except Exception as e:
            print(f"Error processing {artists_file}: {e}")
    
    return True

def main():
    """Main function to process all data"""
    print("=" * 60)
    print("DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Process audio data
    audio_success = process_audio_data()
    
    if not audio_success:
        print("Audio processing failed. Creating dummy data for testing...")
        create_dummy_data()
    
    # Process lyrics data
    process_lyrics_data()
    
    print("\n" + "=" * 60)
    print("DATA PROCESSING COMPLETE!")
    print("=" * 60)
    
    # Show what was created
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        print("\nCreated files:")
        for file in processed_dir.glob("*"):
            print(f"  • {file.name}")

def create_dummy_data():
    """Create dummy data if no audio files are found (for testing)"""
    print("Creating dummy data for testing...")
    
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Create dummy features (100 samples, 50 features)
    dummy_features = np.random.randn(100, 50)
    
    # Create dummy labels (10 genres, 10 samples each)
    dummy_labels = np.array([i // 10 for i in range(100)])
    
    # Create dummy metadata
    genres = ["blues", "classical", "country", "disco", "hiphop", 
              "jazz", "metal", "pop", "reggae", "rock"]
    
    metadata = []
    for i in range(100):
        genre_id = i // 10
        metadata.append({
            'filename': f'sample_{i:03d}.wav',
            'filepath': f'data/{genres[genre_id]}/sample_{i:03d}.wav',
            'genre': genres[genre_id],
            'genre_id': genre_id,
            'duration': 30.0
        })
    
    metadata_df = pd.DataFrame(metadata)
    
    # Save dummy data
    np.save(processed_dir / "gtzan_features.npy", dummy_features)
    np.save(processed_dir / "gtzan_labels.npy", dummy_labels)
    metadata_df.to_csv(processed_dir / "gtzan_metadata.csv", index=False)
    
    print("Dummy data created. Note: This is for testing only!")

if __name__ == "__main__":
    main()