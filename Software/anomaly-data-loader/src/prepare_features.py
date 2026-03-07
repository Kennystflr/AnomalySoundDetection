from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import os
from tqdm import tqdm
from extractor import PerchExtractor

# =====================
# Configuration
# =====================
SR = 16000  # for mel-spectrogram
CHUNK_SEC = 5.0

# Mel parameters
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

RAW_DATA_DIR = Path("/mnt/elephant-seals/cetaceans")
AUDIO_CHUNK_DIR = Path("Software/anomaly-data-loader/data/audio_chunks")
MEL_DIR = Path("Software/anomaly-data-loader/data/mels")
EMBEDDING_DIR = Path("Software/anomaly-data-loader/data/embeddings")

METADATA_CSV = Path("Software/anomaly-data-loader/data/metadata.csv")
REFERENCE_EMB = Path("Software/anomaly-data-loader/data/reference_embedding.npy")
REFERENCE_STD = Path("Software/anomaly-data-loader/data/reference_std.npy")

# Anomaly detection threshold
ANOMALY_THRESHOLD = 0.25

# =====================
# Feature extraction
# =====================
def extract_logmel(y):
    """Extract log-mel spectrogram from audio."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    return librosa.power_to_db(mel, ref=np.max)

# =====================
# Chunking with Perch 2.0 embeddings
# =====================
def chunk_wav_file(wav_path, perch_extractor, metadata_rows):
    """
    Chunk a long audio file into 5s segments.
    Extract mel-spectrograms and Perch 2.0 embeddings.
    
    Args:
        wav_path: Path to input WAV file
        perch_extractor: PerchExtractor instance
        metadata_rows: List to append metadata rows to
    """
    # Load at 16kHz for mel-spectrogram
    y, _ = librosa.load(wav_path, sr=SR)

    samples_per_chunk = int(CHUNK_SEC * SR)
    total_chunks = len(y) // samples_per_chunk

    for idx in range(total_chunks):
        base_name = f"{wav_path.stem}_chunk{idx:04d}"

        audio_out = AUDIO_CHUNK_DIR / f"{base_name}.wav"
        mel_out = MEL_DIR / f"{base_name}.npy"
        emb_out = EMBEDDING_DIR / f"{base_name}.npy"

        # Skip if already processed
        if audio_out.exists() and mel_out.exists() and emb_out.exists():
            metadata_rows.append({
                "chunk_id": base_name,
                "audio_path": str(audio_out),
                "mel_path": str(mel_out),
                "embedding_path": str(emb_out),
                "label": -1,  # unknown
                "distance_to_ref": -1.0
            })
            continue

        start = idx * samples_per_chunk
        end = start + samples_per_chunk
        y_chunk = y[start:end]

        # Create directories
        audio_out.parent.mkdir(parents=True, exist_ok=True)
        mel_out.parent.mkdir(parents=True, exist_ok=True)
        emb_out.parent.mkdir(parents=True, exist_ok=True)

        # Save WAV chunk (16kHz for compatibility)
        sf.write(audio_out, y_chunk, SR)

        # Save mel-spectrogram
        mel = extract_logmel(y_chunk)
        np.save(mel_out, mel)

        # Extract Perch 2.0 embedding from the saved WAV file
        try:
            embedding = perch_extractor.extract_embedding(str(audio_out))
            np.save(emb_out, embedding)
            
            # Compute distance to reference if available
            distance = -1.0
            if REFERENCE_EMB.exists():
                ref_emb = np.load(REFERENCE_EMB)
                distance = perch_extractor.compute_distance(embedding, ref_emb)
                label = 1 if distance > ANOMALY_THRESHOLD else 0
            else:
                label = -1  # unknown
            
            metadata_rows.append({
                "chunk_id": base_name,
                "audio_path": str(audio_out),
                "mel_path": str(mel_out),
                "embedding_path": str(emb_out),
                "label": label,
                "distance_to_ref": round(distance, 4)
            })
        except Exception as e:
            print(f"Error extracting embedding for {base_name}: {e}")
            metadata_rows.append({
                "chunk_id": base_name,
                "audio_path": str(audio_out),
                "mel_path": str(mel_out),
                "embedding_path": str(emb_out),
                "label": -1,
                "distance_to_ref": -1.0
            })

# =====================
# Build reference from normal audio
# =====================
def build_reference_from_folder(folder_path, perch_extractor):
    """
    Build reference embedding from a folder of normal audio files.
    
    Args:
        folder_path: Path to folder containing normal audio chunks
        perch_extractor: PerchExtractor instance
    """
    if not REFERENCE_EMB.exists():
        print(f"Building reference embedding from {folder_path}...")
        
        audio_files = list(Path(folder_path).glob("*.wav"))
        if not audio_files:
            print(f"No audio files found in {folder_path}")
            return
        
        try:
            ref_emb, ref_std = perch_extractor.build_reference([str(f) for f in audio_files])
            np.save(REFERENCE_EMB, ref_emb)
            np.save(REFERENCE_STD, ref_std)
            print(f"Reference embedding saved. Stability: {np.mean(ref_std):.4f}")
        except Exception as e:
            print(f"Error building reference: {e}")

# =====================
# Main
# =====================
def main():
    print("=== Anomaly Sound Detection - Data Preparation ===\n")
    
    # Initialize Perch extractor
    try:
        perch = PerchExtractor()
        print("Perch 2.0 model loaded successfully\n")
    except Exception as e:
        print(f"Error loading Perch model: {e}")
        return
    
    # Create output directories
    AUDIO_CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    MEL_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Chunking RAW WAV files ===")
    metadata_rows = []
    
    wav_files = list(RAW_DATA_DIR.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {RAW_DATA_DIR}")
        return
    
    for wav_path in tqdm(wav_files, desc="Processing audio files"):
        chunk_wav_file(wav_path, perch, metadata_rows)
    
    # Save metadata CSV
    print(f"\n=== Step 2: Saving metadata ===")
    df_metadata = pd.DataFrame(metadata_rows)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_metadata.to_csv(METADATA_CSV, index=False)
    print(f"Metadata saved: {METADATA_CSV}")
    print(f"   Total chunks: {len(df_metadata)}")
    
    # Summary statistics
    if "label" in df_metadata.columns:
        labeled = df_metadata[df_metadata["label"] >= 0]
        if len(labeled) > 0:
            anomalies = (labeled["label"] == 1).sum()
            normal = (labeled["label"] == 0).sum()
            print(f"   Normal chunks: {normal}")
            print(f"   Anomaly chunks: {anomalies}")
    
    print("\n=== Complete ===")

# =====================
if __name__ == "__main__":
    main()