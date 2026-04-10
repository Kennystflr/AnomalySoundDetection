"""
data/dataset.py
---------------
Dataset class for 5-second underwater audio clips.

KEY DESIGN DECISIONS:

  1. RECORDING-STRATIFIED SPLITS
     Clips are not split randomly. All clips from a given recording
     (identified by the filename prefix before '_chunk') are assigned
     to the same partition. This prevents the model from being tested
     on acoustic environments (recordings) it has never encountered
     during training, which would cause distribution shift unrelated to
     the presence of animal vocalizations.

     Without this fix, a clip-level random split sends chunks of the
     same recording to different partitions, causing near-perfect train
     leakage and artificially inflated or deflated test scores depending
     on which recordings happen to be anomalous-only in the metadata.

  2. RAW WAVEFORMS ONLY
     BEATs runs its own internal mel-spectrogram preprocessing via
     torchaudio.compliance.kaldi.fbank at 16000 Hz.
     Do NOT pre-compute spectrograms — return raw waveforms only.
     Normalization is applied at the token-embedding level instead.

Expected metadata CSV format:
    filename,label
    ml19_292b_0027_chunk0052.wav,0   # 0 = normal
    ml19_292b_0065_chunk0025.wav,1   # 1 = anomalous

During training, only normal clips (label=0) are used.
At evaluation time, all clips are used.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
import soundfile as sf
from torch.utils.data import Dataset

# BEATs expects raw audio at 16000 Hz
BEATS_SAMPLE_RATE = 16000


def _extract_recording_id(filename: str) -> str:
    """
    Extract the recording-level identifier from a clip filename.

    Assumes filenames follow the pattern:
        <recording_id>_chunk<N>.wav
    e.g. 'ml19_292b_0027_chunk0052.wav' → 'ml19_292b_0027'

    Falls back to the full stem if no '_chunk' is found.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"(.+)_chunk\d+$", stem)
    return match.group(1) if match else stem


def make_recording_splits(df: pd.DataFrame, train_frac: float, val_frac: float,
                           seed: int = 42):
    """
    Split recordings (not clips) into train / val / test partitions.

    Strategy:
      - Only NORMAL (label=0) recordings are eligible for train and val.
        All clips from these recordings will be exclusively normal.
      - Recordings that contain ANY anomalous clip are sent to test only,
        regardless of what fraction of their clips are anomalous.
      - Remaining normal recordings are split: train_frac / val_frac /
        (1 - train_frac - val_frac) by recording count.

    This guarantees:
      - No recording appears in more than one partition.
      - The model never trains on acoustic environments seen only in
        anomalous test recordings.
      - Test set = held-out normal recordings + all anomalous recordings.

    Returns:
        train_df, val_df, test_df  — DataFrames of clips.
    """
    df = df.copy()
    df["recording_id"] = df["filename"].apply(_extract_recording_id)

    # Recordings that contain at least one anomalous clip → test only
    rec_has_anomaly = df.groupby("recording_id")["label"].max()
    anomalous_recs = rec_has_anomaly[rec_has_anomaly == 1].index
    normal_only_recs = rec_has_anomaly[rec_has_anomaly == 0].index

    # Shuffle normal-only recordings deterministically
    rng = np.random.default_rng(seed)
    normal_recs = rng.permutation(normal_only_recs.tolist()).tolist()

    n = len(normal_recs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_recs = set(normal_recs[:n_train])
    val_recs   = set(normal_recs[n_train: n_train + n_val])
    # Remaining normal-only recs go to test
    test_normal_recs = set(normal_recs[n_train + n_val:])
    test_recs = test_normal_recs | set(anomalous_recs)

    train_df = df[df["recording_id"].isin(train_recs)].copy()
    val_df   = df[df["recording_id"].isin(val_recs)].copy()
    test_df  = df[df["recording_id"].isin(test_recs)].copy()

    # Sanity checks
    assert len(train_df) + len(val_df) + len(test_df) == len(df), \
        "Split sizes don't sum to total — check for recording_id extraction issues."
    assert set(train_df["recording_id"]) & set(test_df["recording_id"]) == set(), \
        "Recording leakage: same recording in train and test."
    assert set(val_df["recording_id"]) & set(test_df["recording_id"]) == set(), \
        "Recording leakage: same recording in val and test."
    assert (train_df["label"] == 1).sum() == 0, \
        "Anomalous clips found in training split."
    assert (val_df["label"] == 1).sum() == 0, \
        "Anomalous clips found in validation split."

    print(f"\nRecording-stratified split:")
    print(f"  Train : {train_df['recording_id'].nunique():3d} recordings, "
          f"{len(train_df):4d} clips  (all normal)")
    print(f"  Val   : {val_df['recording_id'].nunique():3d} recordings, "
          f"{len(val_df):4d} clips  (all normal)")
    print(f"  Test  : {test_df['recording_id'].nunique():3d} recordings, "
          f"{len(test_df):4d} clips  "
          f"({(test_df['label']==1).sum()} anomalous / "
          f"{(test_df['label']==0).sum()} normal)")

    return train_df, val_df, test_df


class UnderwaterAudioDataset(Dataset):
    """
    Loads 5-second audio clips and returns raw waveforms at 16000 Hz.

    Args:
        df (pd.DataFrame):   Pre-split DataFrame with columns
                             [filename, label, recording_id].
                             Build via make_recording_splits().
        root_dir (str):      Root directory containing the audio files.
        config (dict):       Full config dict.
    """

    def __init__(self, df: pd.DataFrame, root_dir: str, config: dict):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.config = config
        self.clip_samples = int(
            config["data"]["clip_duration"] * BEATS_SAMPLE_RATE
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.root_dir, row["filename"])
        label = int(row["label"])

        # soundfile.read returns:
        #   data: np.ndarray, shape (T,) for mono or (T, C) for multichannel
        #   samplerate: int
        # dtype='float32' avoids a separate cast step.
        data, sr = sf.read(filepath, dtype="float32", always_2d=True)
        # data is now always (T, C) — transpose to (C, T) to match torch convention
        waveform = torch.from_numpy(data.T)  # (C, T)

        # Resample to BEATs native rate using torchaudio functional API
        # (no backend required — operates directly on tensors)
        if sr != BEATS_SAMPLE_RATE:
            waveform = TAF.resample(waveform, orig_freq=sr,
                                    new_freq=BEATS_SAMPLE_RATE)

        # Mono: average across channels → (1, T)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or trim to exact clip length
        n = waveform.shape[-1]
        if n < self.clip_samples:
            waveform = F.pad(waveform, (0, self.clip_samples - n))
        else:
            waveform = waveform[..., :self.clip_samples]

        waveform = waveform.squeeze(0)  # (T,) — BEATs expects (B, T)

        return {
            "waveform": waveform,
            "label": label,
            "filename": row["filename"],
        }


def compute_token_mean(train_dataset, encoder, device, batch_size=16):
    """
    Compute per-dimension mean of BEATs token embeddings on the training set.
    Used for token-level normalization (subtracted inside the encoder).

    Returns np.ndarray of shape (D,) where D = 768.
    """
    from torch.utils.data import DataLoader

    print("Computing per-dimension token mean over training set...")
    loader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    encoder.eval()
    all_means = []

    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            E = encoder(waveform)                   # (B, H_p, W_p, D)
            B = E.shape[0]
            clip_mean = E.view(B, -1, E.shape[-1]).mean(dim=1)
            all_means.append(clip_mean.cpu().numpy())

    all_means = np.concatenate(all_means, axis=0)
    token_mean = all_means.mean(axis=0)
    print(f"  Token mean computed — {len(all_means)} clips, D={token_mean.shape[0]}")
    return token_mean