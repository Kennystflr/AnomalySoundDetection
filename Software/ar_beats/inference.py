"""
inference.py
------------
Run AR-BEATs anomaly detection on a folder of 5-second audio chunks.

Loads a trained checkpoint and scores every .wav file in the input
folder, then saves a ranked CSV and prints flagged clips.

Usage:
    python inference.py --chunks path/to/folder --checkpoint checkpoints/best.pt
    python inference.py --chunks path/to/folder --checkpoint checkpoints/best.pt --threshold -411.47
    python inference.py --chunks path/to/folder --checkpoint checkpoints/best.pt --percentile 90
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio.functional as TAF
from pathlib import Path

from models.beats_encoder import BEATsEncoder
from models.ar_cnn import ARCNN
from evaluation.evaluate import agg_mean_std

BEATS_SR     = 16000
CLIP_SAMPLES = 5 * BEATS_SR  # 5 seconds


# ---------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------

def load_clip(path: str) -> torch.Tensor:
    """Load a wav file, resample to 16kHz mono, pad/trim to 5 s."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(data.T)               # (C, T)
    if sr != BEATS_SR:
        wave = TAF.resample(wave, orig_freq=sr, new_freq=BEATS_SR)
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    wave = wave.squeeze(0)                         # (T,)
    if wave.shape[-1] < CLIP_SAMPLES:
        wave = torch.nn.functional.pad(wave, (0, CLIP_SAMPLES - wave.shape[-1]))
    else:
        wave = wave[:CLIP_SAMPLES]
    return wave                                    # (T,)


# ---------------------------------------------------------------
# Inference
# ---------------------------------------------------------------

def score_folder(chunks_dir: str, encoder: BEATsEncoder,
                 ar_model: ARCNN, device: str) -> pd.DataFrame:
    """
    Score every .wav in chunks_dir.
    Returns a DataFrame with filename and anomaly score, sorted
    descending by score.
    """
    paths = sorted(Path(chunks_dir).glob("**/*.wav"))
    if not paths:
        raise FileNotFoundError(f"No .wav files found in {chunks_dir}")

    print(f"Found {len(paths)} clips — running inference...")

    ar_model.eval()
    encoder.eval()

    rows = []
    with torch.no_grad():
        for i, path in enumerate(paths, 1):
            try:
                wave = load_clip(str(path)).unsqueeze(0).to(device)  # (1, T)
                E            = encoder(wave)                          # (1, H_p, W_p, D)
                token_scores = ar_model.nll(E).squeeze(0).cpu().numpy()  # (H_p, W_p)
                score        = agg_mean_std(token_scores)
                rows.append({"filename": path.name, "score": score})
            except Exception as e:
                print(f"  [skip] {path.name}: {e}")

            if i % 50 == 0:
                print(f"  {i}/{len(paths)} done...")

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------
# Threshold strategies
# ---------------------------------------------------------------

def apply_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["prediction"] = (df["score"] >= threshold).astype(int)
    df["label"]      = df["prediction"].map({1: "ANOMALY", 0: "normal"})
    return df


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]

    token_mean_path = os.path.join(
        os.path.dirname(args.checkpoint), "token_mean.npy"
    )
    if not os.path.exists(token_mean_path):
        raise FileNotFoundError(
            f"token_mean.npy not found at {token_mean_path}. "
            "It should be saved alongside the checkpoint."
        )
    token_mean = np.load(token_mean_path)

    encoder = BEATsEncoder(
        model_path=config["beats"]["model_path"],
        device=device,
        token_mean=token_mean,
    )

    ar_cfg   = config["ar_cnn"]
    ar_model = ARCNN(
        embed_dim=config["beats"]["embed_dim"],
        hidden_dim=ar_cfg["hidden_dim"],
        n_layers=ar_cfg["n_layers"],
        dilation=ar_cfg["dilation"],
    ).to(device)
    ar_model.load_state_dict(ckpt["model_state_dict"])

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val NLL = {ckpt.get('val_nll', '?'):.4f})")

    # ---- Score folder ----
    df = score_folder(args.chunks, encoder, ar_model, device)

    # ---- Determine threshold ----
    if args.threshold is not None:
        tau = args.threshold
        print(f"\nUsing fixed threshold τ = {tau:.4f}")
    elif args.percentile is not None:
        tau = float(np.percentile(df["score"], args.percentile))
        print(f"\nUsing p{args.percentile} threshold τ = {tau:.4f} "
              f"(computed over ALL {len(df)} clips — use with caution "
              f"if anomalies are frequent)")
    else:
        tau = None

    # ---- Apply threshold if provided ----
    if tau is not None:
        df = apply_threshold(df, tau)
        n_flagged = (df["prediction"] == 1).sum()
        print(f"Flagged : {n_flagged} / {len(df)} clips "
              f"({n_flagged/len(df):.1%})")
    else:
        df["prediction"] = None
        df["label"]      = None
        print("\nNo threshold set — scores only. Use --threshold or --percentile "
              "to get binary predictions.")

    # ---- Print top anomalous clips ----
    n_show = min(20, len(df))
    print(f"\nTop {n_show} clips by anomaly score:")
    print(f"  {'Rank':<5} {'Filename':<45} {'Score':>10}  {'Label'}")
    print(f"  {'-'*72}")
    for i, row in df.head(n_show).iterrows():
        label_str = row["label"] if row["label"] else ""
        marker    = " <-- ANOMALY" if label_str == "ANOMALY" else ""
        print(f"  {i+1:<5} {row['filename']:<45} {row['score']:>10.2f}  "
              f"{label_str}{marker}")

    # ---- Save results ----
    out_path = os.path.join(args.chunks, "inference_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AR-BEATs inference on a folder of audio chunks"
    )
    parser.add_argument(
        "--chunks", type=str, required=True,
        help="Path to folder containing .wav chunks"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to trained checkpoint (default: checkpoints/best.pt)"
    )

    thresh_group = parser.add_mutually_exclusive_group()
    thresh_group.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed anomaly score threshold (e.g. -411.47 from training). "
             "Clips scoring above this are flagged as anomalous."
    )
    thresh_group.add_argument(
        "--percentile", type=float, default=None,
        help="Flag the top N%% of clips by score (e.g. 90 flags the top 10%%). "
             "Only appropriate when anomalies are rare in the folder."
    )

    args = parser.parse_args()
    main(args)
