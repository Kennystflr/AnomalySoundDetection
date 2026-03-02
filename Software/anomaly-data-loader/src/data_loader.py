"""
PyTorch Dataset loaders for anomalous sound detection.
Supports multi-modal data: audio, mel-spectrogram, and Perch 2.0 embeddings.
"""
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

class AudioChunkDataset(Dataset):
    """Load audio chunks with metadata from CSV file."""
    
    def __init__(self, metadata_csv, load_mel=True, load_embedding=True, load_audio=True):
        """
        Args:
            metadata_csv: Path to metadata CSV with columns: chunk_id, audio_path, mel_path, 
                         embedding_path, label, distance_to_ref
            load_mel: Load mel-spectrograms
            load_embedding: Load Perch 2.0 embeddings
            load_audio: Load audio waveforms
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.base_dir = Path(metadata_csv).parent.parent
        self.load_mel = load_mel
        self.load_embedding = load_embedding
        self.load_audio = load_audio

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        chunk_id = row["chunk_id"]
        label = int(row["label"]) if pd.notna(row["label"]) else -1
        
        data = {"chunk_id": chunk_id, "label": label}
        
        # Load audio
        if self.load_audio:
            audio_path = self.base_dir / row["audio_path"]
            try:
                audio, sr = sf.read(audio_path)
                audio = audio.astype(np.float32)
                data["audio"] = torch.tensor(audio, dtype=torch.float32)
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}")
                data["audio"] = None
        
        # Load mel-spectrogram
        if self.load_mel:
            mel_path = self.base_dir / row["mel_path"]
            try:
                mel = np.load(mel_path)
                data["mel"] = torch.tensor(mel, dtype=torch.float32)
            except Exception as e:
                print(f"Error loading mel {mel_path}: {e}")
                data["mel"] = None
        
        # Load embedding
        if self.load_embedding:
            emb_path = self.base_dir / row["embedding_path"]
            try:
                embedding = np.load(emb_path)
                data["embedding"] = torch.tensor(embedding, dtype=torch.float32)
            except Exception as e:
                print(f"Error loading embedding {emb_path}: {e}")
                data["embedding"] = None
        
        return data


class MetaModalDataset(Dataset):
    """Multi-modal dataset returning all three modalities: audio, mel, embedding."""
    
    def __init__(
        self,
        metadata_csv,
        load_all=True,
        use_anomaly_labels=True
    ):
        """
        Args:
            metadata_csv: Path to metadata CSV
            load_all: If True, return all modalities; if False, skip None values
            use_anomaly_labels: Use distance_to_ref for labels if available
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.base_dir = Path(metadata_csv).parent.parent
        self.load_all = load_all
        self.use_anomaly_labels = use_anomaly_labels
        
        # Filter out rows with missing data if load_all=False
        if not load_all:
            self.metadata = self.metadata.dropna(subset=["audio_path", "mel_path", "embedding_path"])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        chunk_id = row["chunk_id"]
        
        # Get label
        if self.use_anomaly_labels and pd.notna(row.get("distance_to_ref")):
            # Use distance if available
            label = torch.tensor(row["label"], dtype=torch.long)
        else:
            label = torch.tensor(int(row["label"]) if pd.notna(row["label"]) else -1, dtype=torch.long)
        
        # Load audio
        audio_path = self.base_dir / row["audio_path"]
        audio, sr = sf.read(audio_path)
        audio = torch.tensor(audio.astype(np.float32), dtype=torch.float32)
        
        # Load mel
        mel_path = self.base_dir / row["mel_path"]
        mel = np.load(mel_path)
        mel = torch.tensor(mel, dtype=torch.float32)
        
        # Load embedding
        emb_path = self.base_dir / row["embedding_path"]
        embedding = np.load(emb_path)
        embedding = torch.tensor(embedding, dtype=torch.float32)
        
        return {
            "chunk_id": chunk_id,
            "audio": audio,
            "mel": mel,
            "embedding": embedding,
            "label": label,
            "distance": torch.tensor(row.get("distance_to_ref", -1.0), dtype=torch.float32)
        }


def extract_mel(audio, sr=16000, n_fft=1024, hop_length=512, n_mels=64):
    """Extract log-mel spectrogram from audio."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    return librosa.power_to_db(mel, ref=np.max)

def save_embedding(embedding, path):
    """Save embedding to disk."""
    np.save(path, embedding)