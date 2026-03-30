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

METADATA_CSV = Path("Software/anomaly-data-loader/data/metadata.csv")
REFERENCE_EMB = Path("/opt/venv/AnomalySoundDetection/Software/anomaly-data-loader/data/reference/reference_signature.npy")
REFERENCE_STD = Path("/opt/venv/AnomalySoundDetection/Software/anomaly-data-loader/data/reference/reference_std.npy")

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
SR_PERCH = 32000  # Perch requires 32kHz

def chunk_wav_file(wav_path, perch_extractor, base_path, metadata_rows, threshold=0.316, batch_size=8):
    """
    Chunk a long audio file into 5s segments.
    Extract mel-spectrograms and Perch 2.0 embeddings (batched).

    Args:
        wav_path: Path to input WAV file
        perch_extractor: PerchExtractor instance
        base_path: Base path for saving features
        metadata_rows: List to append metadata rows to
        batch_size: Number of chunks to embed in one ONNX forward pass
    """
    samples_per_chunk_mel = int(CHUNK_SEC * SR)
    samples_per_chunk_perch = int(CHUNK_SEC * SR_PERCH)

    # Load audio once at each required sample rate
    y_mel, _ = librosa.load(wav_path, sr=SR)
    y_perch, _ = librosa.load(wav_path, sr=SR_PERCH)

    total_chunks = len(y_mel) // samples_per_chunk_mel

    # Load reference once
    ref_emb = np.load(REFERENCE_EMB) if REFERENCE_EMB.exists() else None

    # Create output directories once
    (base_path / "audio_chunks").mkdir(parents=True, exist_ok=True)
    (base_path / "mels").mkdir(parents=True, exist_ok=True)
    (base_path / "embeddings").mkdir(parents=True, exist_ok=True)

    # Identify which chunks still need embedding extraction
    pending_indices = []
    for idx in range(total_chunks):
        base_name = f"{wav_path.stem}_chunk{idx:04d}"
        audio_out = base_path / "audio_chunks" / f"{base_name}.wav"
        mel_out   = base_path / "mels"         / f"{base_name}.npy"
        emb_out   = base_path / "embeddings"   / f"{base_name}.npy"

        if audio_out.exists() and mel_out.exists() and emb_out.exists():
            metadata_rows.append({
                "chunk_id": base_name,
                "audio_path": str(audio_out),
                "mel_path": str(mel_out),
                "embedding_path": str(emb_out),
                "label": -1,
                "distance_to_ref": -1.0
            })
            continue

        # Save WAV and mel now; collect index for batch embedding
        start_mel = idx * samples_per_chunk_mel
        y_chunk = y_mel[start_mel : start_mel + samples_per_chunk_mel]

        sf.write(audio_out, y_chunk, SR)
        np.save(mel_out, extract_logmel(y_chunk))

        pending_indices.append(idx)

    # Run embedding inference in batches
    for batch_start in range(0, len(pending_indices), batch_size):
        batch_indices = pending_indices[batch_start : batch_start + batch_size]

        # Build (N, samples_per_chunk_perch) batch from in-memory 32kHz audio
        audio_batch = np.stack([
            y_perch[idx * samples_per_chunk_perch : (idx + 1) * samples_per_chunk_perch]
            for idx in batch_indices
        ])

        try:
            embeddings = perch_extractor.extract_embeddings_batch(audio_batch)  # (N, dim)
        except Exception as e:
            print(f"Error running batch inference for {wav_path.stem}: {e}")
            embeddings = None

        for i, idx in enumerate(batch_indices):
            base_name = f"{wav_path.stem}_chunk{idx:04d}"
            audio_out = base_path / "audio_chunks" / f"{base_name}.wav"
            mel_out   = base_path / "mels"         / f"{base_name}.npy"
            emb_out   = base_path / "embeddings"   / f"{base_name}.npy"

            if embeddings is not None:
                embedding = embeddings[i]
                np.save(emb_out, embedding)

                distance = -1.0
                label = -1
                if ref_emb is not None:
                    distance = perch_extractor.compute_distance(embedding, ref_emb)
                    label = 1 if distance > threshold else 0
            else:
                label = -1
                distance = -1.0

            metadata_rows.append({
                "chunk_id": base_name,
                "audio_path": str(audio_out),
                "mel_path": str(mel_out),
                "embedding_path": str(emb_out),
                "label": label,
                "distance_to_ref": round(distance, 4)
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
def main(input_folder=Path("/mnt/gpu_storage/cs_courses/cetaceans/drift_dives_SES/samples"), output_base_path=Path("Software/anomaly-data-loader/data"), threshold=0.316):
    print("=== Anomaly Sound Detection - Data Preparation ===\n")
    
    # Initialize Perch extractor
    try:
        perch = PerchExtractor()
        print("Perch 2.0 model loaded successfully\n")
    except Exception as e:
        print(f"Error loading Perch model: {e}")
        return
    
    # Create output directories
    output_base_path.mkdir(parents=True, exist_ok=True)
    (output_base_path / "audio_chunks").mkdir(parents=True, exist_ok=True)
    (output_base_path / "mels").mkdir(parents=True, exist_ok=True)
    (output_base_path / "embeddings").mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Chunking RAW WAV files ===")
    metadata_rows = []
    
    wav_files = list(input_folder.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {input_folder}")
        return
    
    for wav_path in tqdm(wav_files, desc="Processing audio files"):
        chunk_wav_file(wav_path, perch, output_base_path, metadata_rows, threshold=threshold,batch_size=64)
    
    # Save metadata CSV
    print(f"\n=== Step 2: Saving metadata ===")
    df_metadata = pd.DataFrame(metadata_rows)
    data_path = METADATA_CSV.parent
    metadata_name = input_folder.name+"_metadata.csv"
    data_path.mkdir(parents=True, exist_ok=True)
    df_metadata.to_csv(data_path/metadata_name, index=False)
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