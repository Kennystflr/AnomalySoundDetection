"""
within_recording_detector.py
-----------------------------
Within-recording anomaly detection using BEATs embeddings + classical density estimation.

No training, no cross-recording assumptions, no labels required.

Pipeline:
  1. Extract BEATs embeddings for every clip in a recording
     (mean-pool the 248 tokens → one 768-dim vector per clip)
  2. Fit a density model on those embeddings:
       - LOF (Local Outlier Factor): compares local density around each
         point to its neighbours — robust, non-parametric, handles
         multi-modal normal distributions
       - Gaussian (one-class): fits mean + diagonal covariance, scores
         by Mahalanobis distance — fast, interpretable, assumes unimodal
  3. Score each clip by its anomaly score under the density model
  4. Flag clips beyond a data-driven threshold

Assumptions:
  - Anomalies are RARE within the recording (< ~20% of clips).
    If a recording is half anomalous, neither method will work well.
  - All clips come from the same acoustic environment (same microphone,
    depth, equipment) — which is exactly the within-recording guarantee.

Usage:
    # Single recording — all clips in a folder
    python within_recording_detector.py \\
        --config configs/config.yaml \\
        --recording_id ml17_280a_0060

    # With ground truth labels for evaluation
    python within_recording_detector.py \\
        --config configs/config.yaml \\
        --recording_id ml17_280a_0060 \\
        --metadata data/metadata.csv

    # All recordings in the dataset (one detector per recording)
    python within_recording_detector.py \\
        --config configs/config.yaml \\
        --metadata data/metadata.csv \\
        --all_recordings
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, f1_score,
                              precision_score, recall_score)
import matplotlib.pyplot as plt
import soundfile as sf
import torchaudio.functional as TAF
from pathlib import Path

from models.beats_encoder import BEATsEncoder
from data.dataset import _extract_recording_id, BEATS_SAMPLE_RATE


# ---------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------

def load_and_embed(filepaths: list, encoder: BEATsEncoder,
                   clip_duration: float, device: str,
                   batch_size: int = 16) -> np.ndarray:
    """
    Load clips and extract mean-pooled BEATs embeddings.

    Returns:
        embeddings: np.ndarray (N, D) — one 768-dim vector per clip
    """
    clip_samples = int(clip_duration * BEATS_SAMPLE_RATE)
    embeddings   = []

    for i in range(0, len(filepaths), batch_size):
        batch_paths = filepaths[i: i + batch_size]
        batch_waves = []

        for path in batch_paths:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            wave = torch.from_numpy(data.T)  # (C, T)
            if sr != BEATS_SAMPLE_RATE:
                wave = TAF.resample(wave, orig_freq=sr,
                                    new_freq=BEATS_SAMPLE_RATE)
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            wave = wave.squeeze(0)  # (T,)
            n = wave.shape[-1]
            if n < clip_samples:
                wave = torch.nn.functional.pad(wave, (0, clip_samples - n))
            else:
                wave = wave[:clip_samples]
            batch_waves.append(wave)

        batch_tensor = torch.stack(batch_waves).to(device)  # (B, T)

        with torch.no_grad():
            E = encoder(batch_tensor)   # (B, H_p, W_p, D)
            B, H_p, W_p, D = E.shape
            # Mean-pool over spatial grid → (B, D)
            emb = E.view(B, H_p * W_p, D).mean(dim=1)
            embeddings.append(emb.cpu().numpy())

    return np.concatenate(embeddings, axis=0)  # (N, D)


# ---------------------------------------------------------------
# Density models
# ---------------------------------------------------------------

def fit_lof(embeddings: np.ndarray,
            n_neighbors: int = 20,
            contamination: float = 0.1) -> np.ndarray:
    """
    Local Outlier Factor — non-parametric, no Gaussian assumption.

    contamination: expected fraction of anomalies (used for threshold only,
    not for scoring). Default 0.1 = assume ~10% anomalous.

    Returns anomaly scores (higher = more anomalous).
    LOF returns negative scores — we negate so higher = more anomalous.
    """
    scaler   = StandardScaler()
    X        = scaler.fit_transform(embeddings)
    n_neighbors = min(n_neighbors, len(X) - 1)
    lof      = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
    )
    lof.fit(X)
    # negative_outlier_factor_: more negative = more anomalous
    # Negate so higher = more anomalous
    scores = -lof.negative_outlier_factor_
    return scores, scaler, lof


def fit_gaussian(embeddings: np.ndarray) -> np.ndarray:
    """
    One-class diagonal Gaussian — fast, interpretable.

    Fits mean and per-dimension variance on all clips.
    Scores by Mahalanobis distance (diagonal covariance).
    Anomalous clips = large distance from the recording centroid.

    Returns anomaly scores (higher = more anomalous).
    """
    mu    = embeddings.mean(axis=0)          # (D,)
    var   = embeddings.var(axis=0) + 1e-8   # (D,) — diagonal covariance
    diff  = embeddings - mu                  # (N, D)
    scores = (diff ** 2 / var).sum(axis=1)   # (N,) — Mahalanobis² distance
    return scores, mu, var


# ---------------------------------------------------------------
# Threshold strategies
# ---------------------------------------------------------------

def threshold_percentile(scores: np.ndarray, p: float) -> float:
    """Flag top-p% of clips as anomalous (p=90 → top 10% flagged)."""
    return float(np.percentile(scores, p))


def threshold_mean_std(scores: np.ndarray, k: float = 2.0) -> float:
    """Flag clips more than k std above the mean score."""
    return float(scores.mean() + k * scores.std())


def evaluate_threshold(scores, labels, tau):
    if labels is None:
        preds = (scores >= tau).astype(int)
        return {"n_flagged": int(preds.sum()),
                "flag_rate": float(preds.mean()),
                "tau": tau}
    preds = (scores >= tau).astype(int)
    if preds.sum() == 0 or labels.sum() == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "n_flagged": 0, "flag_rate": 0.0, "tau": tau}
    return {
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "n_flagged": int(preds.sum()),
        "flag_rate": float(preds.mean()),
        "tau":       float(tau),
    }


def oracle_threshold(scores, labels):
    p, r, th = precision_recall_curve(labels, scores)
    f1s = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-8)
    bi  = np.argmax(f1s)
    return float(th[bi]), float(f1s[bi])


# ---------------------------------------------------------------
# Single recording detector
# ---------------------------------------------------------------

def detect_recording(recording_id: str, df: pd.DataFrame,
                     config: dict, encoder: BEATsEncoder,
                     device: str, results_dir: str,
                     contamination: float = 0.1,
                     verbose: bool = True) -> dict:
    """
    Run within-recording anomaly detection on a single recording.

    Args:
        recording_id: Recording identifier string.
        df:           Metadata DataFrame (must have recording_id column).
        config:       Full config dict.
        encoder:      BEATsEncoder (frozen, no token_mean — fit per-recording).
        device:       torch device string.
        results_dir:  Output directory.
        contamination: Expected anomaly fraction (for LOF).
        verbose:      Print results.

    Returns:
        dict with per-clip scores and metrics (if labels available).
    """
    rec_df = df[df["recording_id"] == recording_id].copy().reset_index(drop=True)
    n_clips = len(rec_df)

    if n_clips < 5:
        print(f"  Skipping {recording_id}: only {n_clips} clips "
              f"(minimum 5 required)")
        return {}

    has_labels = "label" in rec_df.columns
    labels = rec_df["label"].values.astype(int) if has_labels else None
    filepaths = [os.path.join(config["data"]["root_dir"], f)
                 for f in rec_df["filename"]]

    if verbose:
        anom_str = (f"  {labels.sum()} anomalous / {n_clips} total"
                    if has_labels else f"  {n_clips} clips (no labels)")
        print(f"\n  Recording: {recording_id}  —  {anom_str}")

    # ---- Extract embeddings ----
    embeddings = load_and_embed(
        filepaths, encoder,
        clip_duration=config["data"]["clip_duration"],
        device=device,
    )  # (N, 768)

    # ---- Per-recording z-score normalisation of embeddings ----
    # (not scores — we normalise the embedding dimensions before fitting)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    results = {"recording_id": recording_id, "n_clips": n_clips}

    # ---- LOF ----
    lof_scores, _, _ = fit_lof(
        emb_scaled, n_neighbors=min(20, n_clips - 1),
        contamination=contamination,
    )

    # ---- Gaussian ----
    gauss_scores, _, _ = fit_gaussian(emb_scaled)

    # ---- Thresholds and evaluation ----
    for method, scores in [("lof", lof_scores), ("gaussian", gauss_scores)]:
        tau_p90  = threshold_percentile(scores, 90)
        tau_p95  = threshold_percentile(scores, 95)
        tau_ms2  = threshold_mean_std(scores, k=2.0)

        results[f"{method}_scores"] = scores.tolist()

        if has_labels and labels.sum() > 0:
            # Ranking metrics
            auroc = roc_auc_score(labels, scores)
            ap    = average_precision_score(labels, scores)
            tau_oracle, f1_oracle = oracle_threshold(scores, labels)

            results[f"{method}_auroc"]     = auroc
            results[f"{method}_ap"]        = ap
            results[f"{method}_f1_oracle"] = f1_oracle

            m_p90 = evaluate_threshold(scores, labels, tau_p90)
            m_p95 = evaluate_threshold(scores, labels, tau_p95)
            m_ms2 = evaluate_threshold(scores, labels, tau_ms2)

            results[f"{method}_f1_p90"]   = m_p90.get("f1", 0)
            results[f"{method}_f1_p95"]   = m_p95.get("f1", 0)
            results[f"{method}_f1_ms2"]   = m_ms2.get("f1", 0)

            if verbose:
                print(f"    {method.upper():10s}  "
                      f"AUROC={auroc:.3f}  AP={ap:.3f}  "
                      f"F1-oracle={f1_oracle:.3f}  "
                      f"F1-p90={m_p90.get('f1',0):.3f}  "
                      f"F1-ms2={m_ms2.get('f1',0):.3f}")
        else:
            m_p90 = evaluate_threshold(scores, labels, tau_p90)
            if verbose:
                print(f"    {method.upper():10s}  "
                      f"flagged={m_p90['n_flagged']}/{n_clips} "
                      f"at p90 threshold (no labels for evaluation)")

    # ---- Save per-clip predictions ----
    os.makedirs(results_dir, exist_ok=True)
    out = rec_df[["filename"]].copy()

    # Extract chunk index from filename and compute timestamps
    # Filenames follow pattern: <recording_id>_chunk<N>.wav
    import re as _re
    def _chunk_idx(fname):
        m = _re.search(r"_chunk(\d+)", os.path.basename(fname))
        return int(m.group(1)) if m else -1

    clip_duration = config["data"]["clip_duration"]
    out["chunk_idx"]      = out["filename"].apply(_chunk_idx)
    out["time_start_s"]   = out["chunk_idx"] * clip_duration
    out["time_end_s"]     = out["time_start_s"] + clip_duration
    out["time_start_min"] = (out["time_start_s"] / 60).round(3)
    out["time_end_min"]   = (out["time_end_s"]   / 60).round(3)

    if has_labels:
        out["label"] = labels
    out["score_lof"]      = lof_scores
    out["score_gaussian"] = gauss_scores
    out["pred_lof_p90"]   = (lof_scores   >= threshold_percentile(lof_scores,   90)).astype(int)
    out["pred_gauss_p90"] = (gauss_scores >= threshold_percentile(gauss_scores, 90)).astype(int)
    out["pred_lof_ms2"]   = (lof_scores   >= threshold_mean_std(lof_scores,   2.0)).astype(int)
    out["pred_gauss_ms2"] = (gauss_scores >= threshold_mean_std(gauss_scores, 2.0)).astype(int)

    # Sort by time for readability
    out = out.sort_values("time_start_s").reset_index(drop=True)

    out_path = os.path.join(results_dir, f"within_rec_{recording_id}.csv")
    out.to_csv(out_path, index=False)

    # ---- Plot ----
    _plot_recording(recording_id, rec_df, lof_scores, gauss_scores,
                    labels, results_dir,
                    clip_duration=config["data"]["clip_duration"])

    results["output_path"] = out_path
    return results


# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------

def _plot_recording(recording_id, rec_df, lof_scores, gauss_scores,
                    labels, results_dir, clip_duration=5.0):
    import re as _re

    def _chunk_idx(fname):
        m = _re.search(r"_chunk(\d+)", os.path.basename(fname))
        return int(m.group(1)) if m else -1

    chunk_indices = rec_df["filename"].apply(_chunk_idx).values
    time_minutes  = chunk_indices * clip_duration / 60.0   # actual timestamps

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for ax, scores, name, color in zip(
        axes,
        [lof_scores, gauss_scores],
        ["LOF score", "Gaussian (Mahalanobis²)"],
        ["#378ADD", "#D85A30"],
    ):
        ax.plot(time_minutes, scores, color=color, lw=1.2, alpha=0.8, label=name)
        tau_p90 = threshold_percentile(scores, 90)
        tau_ms2 = threshold_mean_std(scores, 2.0)
        ax.axhline(tau_p90, color="gray", linestyle="--", lw=1,
                   label=f"p90 threshold ({tau_p90:.1f})")
        ax.axhline(tau_ms2, color="gray", linestyle=":", lw=1,
                   label=f"mean+2σ ({tau_ms2:.1f})")

        if labels is not None:
            anom_times = time_minutes[labels == 1]
            if len(anom_times) > 0:
                ax.scatter(anom_times, scores[labels == 1],
                           color="#D85A30", s=30, zorder=5,
                           label="anomalous clips")

        ax.set_ylabel(name)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time in recording (minutes)")
    n = len(lof_scores)
    fig.suptitle(
        f"Within-recording anomaly detection — {recording_id}\n"
        f"{n} clips  |  "
        f"{'labels available' if labels is not None else 'unsupervised'}"
    )
    plt.tight_layout()
    path = os.path.join(results_dir, f"within_rec_{recording_id}.png")
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results_dir = os.path.join(config["evaluation"]["results_dir"],
                               "within_recording")
    os.makedirs(results_dir, exist_ok=True)

    # Load encoder — no token_mean (fit per recording instead)
    encoder = BEATsEncoder(
        model_path=config["beats"]["model_path"],
        device=device,
        token_mean=None,
    )
    encoder.eval()

    # Load metadata
    if args.metadata:
        df = pd.read_csv(args.metadata)
        df["recording_id"] = df["filename"].apply(_extract_recording_id)
    else:
        # No metadata — discover clips from root_dir
        root = config["data"]["root_dir"]
        wavs = sorted(Path(root).glob("**/*.wav"))
        df = pd.DataFrame({
            "filename":     [str(w.relative_to(root)) for w in wavs],
            "recording_id": [_extract_recording_id(str(w)) for w in wavs],
        })

    # Determine which recordings to run
    if args.all_recordings:
        recording_ids = sorted(df["recording_id"].unique())
    elif args.recording_id:
        recording_ids = [args.recording_id]
    else:
        print("Specify --recording_id <id> or --all_recordings")
        return

    print(f"\nRunning within-recording detection on "
          f"{len(recording_ids)} recording(s)...\n")

    all_results = []
    for rec_id in recording_ids:
        r = detect_recording(
            recording_id=rec_id,
            df=df,
            config=config,
            encoder=encoder,
            device=device,
            results_dir=results_dir,
            contamination=args.contamination,
            verbose=True,
        )
        if r:
            all_results.append(r)

    # Aggregate summary if labels available and multiple recordings
    if len(all_results) > 1:
        summary_rows = []
        for r in all_results:
            row = {"recording_id": r["recording_id"],
                   "n_clips": r["n_clips"]}
            for key in ["lof_auroc", "lof_ap", "lof_f1_oracle",
                        "lof_f1_p90", "gaussian_auroc", "gaussian_ap",
                        "gaussian_f1_oracle", "gaussian_f1_p90"]:
                row[key] = r.get(key, float("nan"))
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(results_dir, "within_rec_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*60}")
        print(f"  Within-Recording Detection Summary")
        print(f"{'='*60}")
        for col in ["lof_auroc", "lof_ap", "lof_f1_p90",
                    "gaussian_auroc", "gaussian_ap", "gaussian_f1_p90"]:
            vals = summary_df[col].dropna()
            if len(vals):
                print(f"  {col:<25} {vals.mean():.4f} ± {vals.std():.4f}")
        print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Within-recording anomaly detection (no training required)"
    )
    parser.add_argument("--config", type=str,
                        default="configs/config.yaml")
    parser.add_argument("--metadata", type=str,
                        default=None,
                        help="Path to metadata CSV. If omitted, discovers "
                             "all wavs in data.root_dir.")
    parser.add_argument("--recording_id", type=str,
                        default=None,
                        help="Single recording ID to analyse.")
    parser.add_argument("--all_recordings", action="store_true",
                        help="Run on every recording in the metadata.")
    parser.add_argument("--contamination", type=float,
                        default=0.1,
                        help="Expected anomaly fraction for LOF (default 0.1).")
    args = parser.parse_args()
    main(args)