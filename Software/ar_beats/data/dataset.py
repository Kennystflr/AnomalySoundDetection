"""
data/dataset.py
---------------
Dataset class for 5-second underwater audio clips.

KEY DESIGN DECISION:
    BEATs runs its OWN internal mel-spectrogram preprocessing via
    torchaudio.compliance.kaldi.fbank at 16000 Hz.
    Do NOT pre-compute spectrograms here — return raw waveforms only.
    Normalization is applied at the token-embedding level instead
    (see compute_token_mean below).

Expected metadata CSV format:
    filename,label
    clip_001.wav,0      # 0 = normal
    clip_002.wav,1      # 1 = anomalous

During training, only normal clips (label=0) are used.
At evaluation time, all clips are used.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile
import torchaudio.transforms as T

# BEATs expects raw audio at 16000 Hz — fixed
BEATS_SAMPLE_RATE = 16000


class UnderwaterAudioDataset(Dataset):
    """
    Loads 5-second audio clips and returns raw waveforms resampled to
    16000 Hz (BEATs' native rate).

    Args:
        metadata_file (str): Path to CSV with columns [filename, label].
        root_dir (str):      Root directory containing the audio files.
        config (dict):       Full config dict.
        split (str):         'train' | 'val' | 'test'.
        token_mean (np.ndarray | None): Per-dimension mean of BEATs token
            embeddings, shape (D,). Applied inside the encoder after
            extraction. Pass None during the mean-computation pass.
    """

    def __init__(self, metadata_file, root_dir, config, split, token_mean=None):
        self.root_dir = root_dir
        self.config = config
        self.split = split
        self.token_mean = token_mean  # stored but applied in encoder, not here

        df = pd.read_csv(metadata_file)

        normal = df[df["label"] == 0].reset_index(drop=True)
        anomalous = df[df["label"] == 1].reset_index(drop=True)

        n = len(normal)
        n_train = int(n * config["data"]["train_split"])
        n_val = int(n * config["data"]["val_split"])

        if split == "train":
            self.df = normal.iloc[:n_train]
        elif split == "val":
            self.df = normal.iloc[n_train: n_train + n_val]
        elif split == "test":
            remaining_normal = normal.iloc[n_train + n_val:]
            self.df = pd.concat([remaining_normal, anomalous], ignore_index=True)
        else:
            raise ValueError(f"Unknown split: {split}")

        self.df = self.df.reset_index(drop=True)

        self.resample_cache = {}
        self.clip_samples = int(config["data"]["clip_duration"] * BEATS_SAMPLE_RATE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.root_dir, row["filename"])
        label = int(row["label"])

        audio, sr = soundfile.read(filepath, always_2d=True)
        waveform = torch.from_numpy(audio.T).float()  # (channels, samples)

        # Resample to BEATs native rate
        if sr != BEATS_SAMPLE_RATE:
            if sr not in self.resample_cache:
                self.resample_cache[sr] = T.Resample(sr, BEATS_SAMPLE_RATE)
            waveform = self.resample_cache[sr](waveform)

        # Mono: (1, T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or trim to exact clip length
        n = waveform.shape[-1]
        if n < self.clip_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.clip_samples - n))
        else:
            waveform = waveform[..., :self.clip_samples]

        # Squeeze to (T,) — BEATs expects (batch, time), batching by DataLoader
        waveform = waveform.squeeze(0)  # (T,)

        return {
            "waveform": waveform,        # (T,) float32 at 16000 Hz
            "label": label,
            "filename": row["filename"],
        }


def compute_token_mean(train_dataset, encoder, device, batch_size=16):
    """
    Compute per-dimension mean over BEATs token embeddings on the training set.

    Since BEATs handles its own mel preprocessing internally, normalization
    is applied at the token level rather than the spectrogram level.

    Returns np.ndarray of shape (D,) where D = BEATs embed_dim (768).
    """
    from torch.utils.data import DataLoader

    print("Computing per-dimension token mean over training set...")
    loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    encoder.eval()
    all_means = []

    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)       # (B, T)
            E = encoder(waveform)                          # (B, H_p, W_p, D)
            B = E.shape[0]
            # Spatial mean per clip → (B, D)
            clip_mean = E.view(B, -1, E.shape[-1]).mean(dim=1)
            all_means.append(clip_mean.cpu().numpy())

    all_means = np.concatenate(all_means, axis=0)   # (N, D)
    token_mean = all_means.mean(axis=0)              # (D,)
    print(f"  Token mean computed — {len(all_means)} clips, D={token_mean.shape[0]}")
    return token_mean