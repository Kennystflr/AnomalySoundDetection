from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import shutil
import soundfile as sf

# =====================
# Configuration
# =====================
SR = 16000
CHUNK_SEC = 3.0

# Mel parameters
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

WAV_DIR = Path("wav")
FEATURE_ROOT = Path("features")

AUDIO_ROOT = FEATURE_ROOT / "audio"
MEL_ROOT = FEATURE_ROOT / "mel"

UNLABELED_AUDIO = AUDIO_ROOT / "unlabeled"
UNLABELED_MEL = MEL_ROOT / "unlabeled"

NORMAL_AUDIO = AUDIO_ROOT / "normal"
ANOMALY_AUDIO = AUDIO_ROOT / "anomaly"

NORMAL_MEL = MEL_ROOT / "normal"
ANOMALY_MEL = MEL_ROOT / "anomaly"

METADATA_PATH = Path("metadata.csv")

# =====================
# Feature extraction
# =====================
def extract_logmel(y):
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
# Chunking
# =====================
def chunk_wav_file(wav_path):
    y, _ = librosa.load(wav_path, sr=SR)

    samples_per_chunk = int(CHUNK_SEC * SR)
    total_chunks = len(y) // samples_per_chunk

    for idx in range(total_chunks):
        base_name = f"{wav_path.stem}_chunk{idx:04d}"

        audio_out = UNLABELED_AUDIO / f"{base_name}.wav"
        mel_out = UNLABELED_MEL / f"{base_name}.npy"

        if audio_out.exists() and mel_out.exists():
            continue

        start = idx * samples_per_chunk
        end = start + samples_per_chunk
        y_chunk = y[start:end]

        # Save WAV chunk (PERCH compatible)
        audio_out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_out, y_chunk, SR)

        # Save mel-spectrogram (ML models)
        mel = extract_logmel(y_chunk)
        mel_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(mel_out, mel)

# =====================
# Sorting (optional)
# =====================
def sort_by_metadata():
    print("metadata.csv found → sorting chunks")

    df = pd.read_csv(METADATA_PATH)

    for d in [NORMAL_AUDIO, ANOMALY_AUDIO, NORMAL_MEL, ANOMALY_MEL]:
        d.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        chunk = Path(row["chunk"]).stem
        label = int(row["label"])

        src_audio = UNLABELED_AUDIO / f"{chunk}.wav"
        src_mel = UNLABELED_MEL / f"{chunk}.npy"

        if not src_audio.exists():
            print(f"Missing chunk: {chunk}")
            continue

        if label == 0:
            dst_audio, dst_mel = NORMAL_AUDIO, NORMAL_MEL
        else:
            dst_audio, dst_mel = ANOMALY_AUDIO, ANOMALY_MEL

        shutil.move(src_audio, dst_audio / src_audio.name)
        shutil.move(src_mel, dst_mel / src_mel.name)

# =====================
# Main
# =====================
def main():
    UNLABELED_AUDIO.mkdir(parents=True, exist_ok=True)
    UNLABELED_MEL.mkdir(parents=True, exist_ok=True)

    print("=== Chunking WAV files ===")
    for wav in WAV_DIR.glob("*.wav"):
        print(f"Processing {wav.name}")
        chunk_wav_file(wav)

    if METADATA_PATH.exists():
        print("=== Sorting labeled chunks ===")
        sort_by_metadata()
    else:
        print("No metadata.csv found → skipping sorting")

# =====================
if __name__ == "__main__":
    main()
