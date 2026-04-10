"""
utils/recording_analysis.py
----------------------------
DSP-based analysis of acoustic distance between recordings.

Computes a set of spectral and energy features per recording from raw audio,
then measures how different each recording is from the others. Intended to be
run after LORO to explain why certain folds perform better or worse.

Features computed per recording (averaged over all clips):
  - noise_floor_db   : median spectral energy (dB), proxy for ambient noise level
  - spectral_centroid: energy-weighted mean frequency (Hz)
  - spectral_flatness: geometric/arithmetic mean ratio (0=tonal, 1=white noise)
  - lf_energy_ratio  : fraction of energy below 500 Hz (shipping, swell)
  - rms_db           : mean clip loudness (dB)
  - rms_std_db       : std of clip loudness — measures noise floor stability
  - dominant_freq    : frequency of peak spectral energy (Hz)

Usage:
    # Standalone analysis
    python utils/recording_analysis.py --config configs/config.yaml

    # Import and use in LORO results analysis
    from utils.recording_analysis import compute_recording_features, plot_distance_vs_auroc
"""

import argparse
import os
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio.functional as TAF
import torch
from scipy.signal import welch
from scipy.stats import gmean



BEATS_SAMPLE_RATE = 16000
LOW_FREQ_CUTOFF   = 500   # Hz — threshold for low-frequency energy ratio


# ---------------------------------------------------------------
# Per-clip DSP features
# ---------------------------------------------------------------

def load_clip(filepath: str, target_sr: int = BEATS_SAMPLE_RATE,
              clip_samples: int = None) -> np.ndarray:
    """
    Load and resample a single clip to target_sr.
    Returns mono float32 array of shape (T,).
    """
    data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)  # (C, T)

    if sr != target_sr:
        waveform = TAF.resample(waveform, orig_freq=sr, new_freq=target_sr)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    wave = waveform.squeeze(0).numpy()  # (T,)

    if clip_samples is not None:
        if len(wave) < clip_samples:
            wave = np.pad(wave, (0, clip_samples - len(wave)))
        else:
            wave = wave[:clip_samples]

    return wave


def clip_features(wave: np.ndarray, sr: int = BEATS_SAMPLE_RATE) -> dict:
    """
    Compute DSP features for a single waveform clip.

    Returns a dict of scalar features.
    """
    # Power spectral density via Welch's method
    nperseg = min(512, len(wave) // 4)
    freqs, psd = welch(wave, fs=sr, nperseg=nperseg)

    eps = 1e-12
    psd = np.maximum(psd, eps)

    # RMS energy
    rms = float(np.sqrt(np.mean(wave ** 2)))
    rms_db = float(20 * np.log10(rms + eps))

    # Noise floor: median PSD in dB
    noise_floor_db = float(10 * np.log10(np.median(psd)))

    # Spectral centroid
    total_power = psd.sum()
    spectral_centroid = float((freqs * psd).sum() / (total_power + eps))

    # Spectral flatness: geometric mean / arithmetic mean of PSD
    # Use log-space geometric mean to avoid underflow
    log_psd = np.log(psd)
    geom_mean = float(np.exp(log_psd.mean()))
    arith_mean = float(psd.mean())
    spectral_flatness = float(geom_mean / (arith_mean + eps))

    # Low-frequency energy ratio (below LOW_FREQ_CUTOFF Hz)
    lf_mask = freqs <= LOW_FREQ_CUTOFF
    lf_energy_ratio = float(psd[lf_mask].sum() / (total_power + eps))

    # Dominant frequency
    dominant_freq = float(freqs[np.argmax(psd)])

    return {
        "rms_db":            rms_db,
        "noise_floor_db":    noise_floor_db,
        "spectral_centroid": spectral_centroid,
        "spectral_flatness": spectral_flatness,
        "lf_energy_ratio":   lf_energy_ratio,
        "dominant_freq":     dominant_freq,
    }


# ---------------------------------------------------------------
# Per-recording aggregation
# ---------------------------------------------------------------

def compute_recording_features(df: pd.DataFrame, root_dir: str,
                                clip_duration: float = 5.0,
                                max_clips_per_rec: int = 30,
                                verbose: bool = True) -> pd.DataFrame:
    """
    Compute DSP feature statistics for each recording.

    Averages clip-level features across all clips in a recording.
    Also computes rms_std_db (std of clip RMS) as a measure of
    noise floor stability.

    Args:
        df:                Full metadata DataFrame (filename, label, recording_id).
        root_dir:          Root audio directory.
        clip_duration:     Clip length in seconds.
        max_clips_per_rec: Maximum clips to sample per recording (for speed).
        verbose:           Print progress.

    Returns:
        DataFrame with one row per recording and mean feature values.
    """
    clip_samples = int(clip_duration * BEATS_SAMPLE_RATE)
    feature_cols = ["rms_db", "noise_floor_db", "spectral_centroid",
                    "spectral_flatness", "lf_energy_ratio", "dominant_freq"]

    recs = df["recording_id"].unique()
    rows = []

    for i, rec_id in enumerate(sorted(recs)):
        if verbose:
            print(f"  [{i+1}/{len(recs)}] {rec_id}")

        rec_df = df[df["recording_id"] == rec_id]
        # Sample up to max_clips_per_rec clips for speed
        if len(rec_df) > max_clips_per_rec:
            rec_df = rec_df.sample(max_clips_per_rec, random_state=42)

        clip_feats = []
        for _, row in rec_df.iterrows():
            filepath = os.path.join(root_dir, row["filename"])
            try:
                wave = load_clip(filepath, clip_samples=clip_samples)
                feats = clip_features(wave)
                clip_feats.append(feats)
            except Exception as e:
                if verbose:
                    print(f"    Warning: failed to load {row['filename']}: {e}")
                continue

        if not clip_feats:
            continue

        feat_df = pd.DataFrame(clip_feats)
        rec_row = {"recording_id": rec_id, "n_clips": len(rec_df),
                   "label": rec_df["label"].max()}  # 1 if any anomalous clip

        for col in feature_cols:
            rec_row[f"mean_{col}"] = float(feat_df[col].mean())

        # RMS stability: std of per-clip RMS across the recording
        rec_row["rms_std_db"] = float(feat_df["rms_db"].std())

        rows.append(rec_row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Distance from training set
# ---------------------------------------------------------------

def compute_distance_from_training(features_df: pd.DataFrame,
                                   training_rec_ids: list) -> pd.DataFrame:
    """
    For each recording, compute its Euclidean distance from the centroid
    of the training set recordings in feature space.

    Features are z-scored before distance computation so all features
    contribute equally regardless of scale.

    Returns features_df with an added 'dist_from_train' column.
    """
    feat_cols = [c for c in features_df.columns
                 if c.startswith("mean_") or c == "rms_std_db"]

    X = features_df[feat_cols].values.astype(float)

    # Z-score normalise across all recordings
    mu    = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    X_norm = (X - mu) / sigma

    train_mask = features_df["recording_id"].isin(training_rec_ids)
    train_centroid = X_norm[train_mask].mean(axis=0)

    dists = np.linalg.norm(X_norm - train_centroid, axis=1)
    features_df = features_df.copy()
    features_df["dist_from_train"] = dists

    return features_df


# ---------------------------------------------------------------
# Merge with LORO results and plot
# ---------------------------------------------------------------

def merge_with_loro(features_df: pd.DataFrame,
                    loro_results_path: str) -> pd.DataFrame:
    """
    Merge recording features with LORO per-fold metrics.

    Returns a DataFrame with both DSP features and AUROC/AP/F1 per recording.
    """
    loro_df = pd.read_csv(loro_results_path)
    merged  = loro_df.merge(features_df, left_on="held_out_rec",
                            right_on="recording_id", how="left")
    return merged


def plot_distance_vs_auroc(merged_df: pd.DataFrame, results_dir: str):
    """
    Scatter plot: acoustic distance from training set vs per-fold AUROC and AP.
    Helps identify whether poor LORO folds correspond to acoustically distant recordings.
    """
    import matplotlib.pyplot as plt

    if "dist_from_train" not in merged_df.columns:
        print("dist_from_train column missing — run compute_distance_from_training first.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric in zip(axes, ["auroc", "ap"]):
        valid = merged_df.dropna(subset=["dist_from_train", metric])
        ax.scatter(valid["dist_from_train"], valid[metric],
                   s=60, color="#378ADD", zorder=3)

        for _, row in valid.iterrows():
            ax.annotate(row["held_out_rec"].split("_")[-1],
                        (row["dist_from_train"], row[metric]),
                        fontsize=8, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

        # Correlation
        if len(valid) > 2:
            corr = float(valid["dist_from_train"].corr(valid[metric]))
            ax.set_title(f"{metric.upper()} vs acoustic distance  (r={corr:.2f})")
        else:
            ax.set_title(f"{metric.upper()} vs acoustic distance")

        ax.set_xlabel("Distance from training centroid (z-scored features)")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(results_dir, "loro_distance_vs_metrics.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Distance vs metrics plot saved to {path}")


def print_feature_summary(features_df: pd.DataFrame):
    """Print a human-readable summary of per-recording DSP features."""
    feat_cols = [c for c in features_df.columns
                 if c.startswith("mean_") or c == "rms_std_db"
                 or c == "dist_from_train"]

    print(f"\n{'='*80}")
    print(f"  Recording acoustic feature summary")
    print(f"{'='*80}")
    header = f"  {'Recording':<25} {'N':>4} {'Lbl':>4}"
    for col in feat_cols:
        short = col.replace("mean_", "").replace("_", " ")[:10]
        header += f"  {short:>10}"
    print(header)
    print(f"  {'-'*76}")

    for _, row in features_df.sort_values("recording_id").iterrows():
        line = (f"  {row['recording_id']:<25} "
                f"{int(row['n_clips']):>4} "
                f"{int(row['label']):>4}")
        for col in feat_cols:
            if col in row:
                line += f"  {row[col]:>10.3f}"
        print(line)
    print()


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from data.dataset import _extract_recording_id

    parser = argparse.ArgumentParser(
        description="DSP-based recording distance analysis"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--loro_results", type=str,
                        default="results/loro_results.csv",
                        help="Path to LORO per-fold results CSV")
    parser.add_argument("--max_clips", type=int, default=30,
                        help="Max clips to sample per recording (for speed)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = config["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(config["data"]["metadata_file"])
    df["recording_id"] = df["filename"].apply(_extract_recording_id)

    print("Computing DSP features per recording...")
    features_df = compute_recording_features(
        df,
        root_dir=config["data"]["root_dir"],
        clip_duration=config["data"]["clip_duration"],
        max_clips_per_rec=args.max_clips,
    )

    # Identify training recordings (normal-only, enough clips)
    from evaluate_loro import discover_recordings
    qualifying, excluded, anomalous_df, _ = discover_recordings(df)

    # For distance: use all qualifying normal recordings as "training context"
    features_df = compute_distance_from_training(features_df, qualifying)

    print_feature_summary(features_df)

    # Save features
    feat_path = os.path.join(results_dir, "recording_features.csv")
    features_df.to_csv(feat_path, index=False)
    print(f"Recording features saved to {feat_path}")

    # Merge with LORO if available
    if os.path.exists(args.loro_results):
        merged = merge_with_loro(features_df, args.loro_results)
        plot_distance_vs_auroc(merged, results_dir)
        merged_path = os.path.join(results_dir, "loro_with_features.csv")
        merged.to_csv(merged_path, index=False)
        print(f"Merged results saved to {merged_path}")
    else:
        print(f"LORO results not found at {args.loro_results} — "
              f"run evaluate_loro.py first, then re-run this script.")
