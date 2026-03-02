import csv
import os
import time
import re
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import onnxruntime as ort
from scipy.spatial.distance import cosine
from tqdm import tqdm


# ─── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = "Software/Perch2.0/perch_v2.onnx"
SR = 32000
INPUT_LENGTH = 160000  # 5 seconds @ 32 kHz

FOLDER_NORMAL = "neutre_sounds_5sec"
FOLDER_TEST   = "Sound2test"

THRESHOLD_COSINE = 0.25
BATCH_SIZE = 8          # adjust depending on your hardware
NUM_WORKERS = 4

REF_EMB_FILE   = "reference_embedding.npy"
REF_STD_FILE   = "reference_std.npy"
REPORT_CSV     = "anomaly_report.csv"


# ─── ONNX Session (global) ─────────────────────────────────────────────────────
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])


def extract_perch_embedding(audio: np.ndarray) -> np.ndarray:
    """
    Extract 1536-dim embedding from Perch 2.0 (mean-pooled).
    Audio should be mono, 32kHz, up to 5 seconds.
    """
    input_tensor = np.zeros((1, INPUT_LENGTH), dtype=np.float32)
    n = min(len(audio), INPUT_LENGTH)
    input_tensor[0, :n] = audio[:n]

    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    # Perch 2.0 typically returns [logits, embedding, spatial_embedding]
    # we take the mean-pooled embedding
    embedding = outputs[1]          # shape (1, 1536)
    return embedding.flatten()      # (1536,)


# ─── Dataset ────────────────────────────────────────────────────────────────────

class AudioEmbeddingDataset(Dataset):
    """
    Loads audio files and computes Perch embeddings on-the-fly.
    Can be restricted to normal samples only via NormalOnly wrapper.
    """
    def __init__(
        self,
        folder: str,
        extensions=('.wav', '.mp3', '.flac', '.ogg'),
        cache_embeddings: bool = False,
        cache_dir: Optional[str] = None
    ):
        self.folder = folder
        self.files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(extensions)
        ])
        self.cache_embeddings = cache_embeddings
        self.cache_dir = cache_dir

        if cache_embeddings and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str, Optional[int]]:
        filename = self.files[idx]
        path = os.path.join(self.folder, filename)

        # Try to load cached embedding
        if self.cache_embeddings and self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{filename}.npy")
            if os.path.exists(cache_path):
                emb = np.load(cache_path)
                return torch.from_numpy(emb), filename, idx

        # Load & resample audio
        audio, sr = librosa.load(path, sr=SR, mono=True)
        if len(audio) == 0:
            # fallback – silent
            audio = np.zeros(INPUT_LENGTH, dtype=np.float32)

        emb_np = extract_perch_embedding(audio)

        if self.cache_embeddings and self.cache_dir:
            np.save(os.path.join(self.cache_dir, f"{filename}.npy"), emb_np)

        return torch.from_numpy(emb_np), filename, idx


class NormalOnly(Dataset):
    """Filter to normal (label=0) samples — here we assume whole folder is normal"""
    def __init__(self, base_dataset: AudioEmbeddingDataset):
        self.base = base_dataset
        # In your current setup all samples in FOLDER_NORMAL are "normal"
        self.indices = list(range(len(base_dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]


# ─── Reference computation ──────────────────────────────────────────────────────

def build_reference(
    normal_folder: str,
    force_recompute: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    if not force_recompute and os.path.exists(REF_EMB_FILE):
        print("Loading cached reference embedding...")
        ref = np.load(REF_EMB_FILE)
        std = np.load(REF_STD_FILE) if os.path.exists(REF_STD_FILE) else None
        return ref, std

    print(f"Building reference from folder: {normal_folder}")
    ds_normal = AudioEmbeddingDataset(normal_folder)
    ds_normal_only = NormalOnly(ds_normal)  # redundant now, but future-proof

    loader = DataLoader(
        ds_normal_only,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    embeddings = []
    for batch_emb, _, _ in tqdm(loader, desc="Computing normal embeddings"):
        embeddings.append(batch_emb.numpy())

    if not embeddings:
        raise ValueError("No normal samples found!")

    emb_matrix = np.concatenate(embeddings, axis=0)   # (N, 1536)
    ref = np.mean(emb_matrix, axis=0)
    std = np.std(emb_matrix, axis=0)

    overall_stability = np.mean(std)
    print(f"Reference stability (lower is better): {overall_stability:.4f}")

    np.save(REF_EMB_FILE, ref)
    np.save(REF_STD_FILE, std)
    print("Reference saved.")

    return ref, std


# ─── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_test_set(
    ref_embedding: np.ndarray,
    test_folder: str,
    threshold: float = THRESHOLD_COSINE
) -> List[Dict]:
    ds_test = AudioEmbeddingDataset(test_folder)

    loader = DataLoader(
        ds_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    results = []

    print(f"Evaluating {len(ds_test)} test segments...")

    for batch_emb, batch_filenames, _ in tqdm(loader):
        batch_emb = batch_emb.numpy()  # (bs, 1536)

        for emb, fname in zip(batch_emb, batch_filenames):
            dist = cosine(ref_embedding, emb)

            # Try to parse part index for timestamp
            match = re.search(r'part(\d+)', fname)
            part_idx = int(match.group(1)) if match else 0
            start_sec = part_idx * 5
            min_ = int(start_sec // 60)
            sec = int(start_sec % 60)
            timestamp = f"{min_}:{sec:02d}"

            status = "ANOMALIE" if dist > threshold else "RAS"

            results.append({
                "Source Audio": fname,
                "Part": part_idx,
                "Start (sec)": start_sec,
                "Start (min:sec)": timestamp,
                "Cosine Distance": round(float(dist), 4),
                "Status": status
            })

    return results


def save_report(results: List[Dict], csv_path: str = REPORT_CSV):
    if not results:
        print("No results to save.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("Cosine Distance", ascending=False)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Report saved → {csv_path}")
    print(f"Total segments analyzed: {len(df)}")

    print("\nTop 5 potential anomalies:")
    anomalies = df[df["Status"] == "ANOMALIE"].head(5)
    if not anomalies.empty:
        for _, row in anomalies.iterrows():
            print(f"  ⚠️ {row['Source Audio']}  dist = {row['Cosine Distance']:.4f}")
    else:
        print("No anomalies detected above threshold.")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    start_time = time.perf_counter()

    ref_emb, _ = build_reference(
        normal_folder=FOLDER_NORMAL,
        force_recompute=False   # set True to recompute
    )

    if ref_emb is None:
        print("Failed to obtain reference embedding. Exiting.")
        return

    results = evaluate_test_set(
        ref_embedding=ref_emb,
        test_folder=FOLDER_TEST,
        threshold=THRESHOLD_COSINE
    )

    save_report(results)

    total_time = time.perf_counter() - start_time
    print(f"\n{'='*50}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()