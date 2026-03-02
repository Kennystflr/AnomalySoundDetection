from pathlib import Path
import numpy as np
import soundfile as sf

# =====================
# Configuration
# =====================
DATA_DIR = Path("data")
AUDIO_CHUNKS_DIR = DATA_DIR / "audio_chunks"
MELS_DIR = DATA_DIR / "mels"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# =====================
# Saving Functions
# =====================
def save_audio_chunk(chunk, filename):
    audio_out = AUDIO_CHUNKS_DIR / filename
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_out, chunk, 16000)

def save_mel_spectrogram(mel, filename):
    mel_out = MELS_DIR / filename
    mel_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(mel_out, mel)

def save_embedding(embedding, filename):
    embedding_out = EMBEDDINGS_DIR / filename
    embedding_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_out, embedding)

def save_features(chunk, mel, embedding, base_name):
    save_audio_chunk(chunk, f"{base_name}.wav")
    save_mel_spectrogram(mel, f"{base_name}.npy")
    save_embedding(embedding, f"{base_name}_embedding.npy")