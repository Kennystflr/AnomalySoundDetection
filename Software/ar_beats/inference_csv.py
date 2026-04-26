"""
inference_csv.py
----------------
Run AR-BEATs inference on any CSV with a 'filename' column.
Audio files are looked up in root_dir from config.yaml.

All original CSV columns are preserved in the output, with 'score'
and optionally 'prediction' appended.

Threshold is derived from the validation set at a given percentile
(default: 65th, matching the calibrated deployable threshold).

Usage:
    python inference_csv.py --csv data/comparison_test.csv
    python inference_csv.py --csv data/comparison_test.csv --percentile 65
    python inference_csv.py --csv data/comparison_test.csv --tau -411.47
    python inference_csv.py --csv data/comparison_test.csv --no-threshold
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchaudio.functional as TAF
import soundfile as sf

from data.dataset import make_recording_splits, BEATS_SAMPLE_RATE
from models.beats_encoder import BEATsEncoder
from models.ar_cnn import ARCNN
from evaluation.evaluate import agg_mean_std
from utils.seed import set_seed


# ── Minimal dataset that doesn't require a 'label' column ────────────────────

class InferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root_dir: str, clip_duration: float):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.clip_samples = int(clip_duration * BEATS_SAMPLE_RATE)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.root_dir, row["filename"])

        data, sr = sf.read(filepath, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)  # (C, T)

        if sr != BEATS_SAMPLE_RATE:
            waveform = TAF.resample(waveform, orig_freq=sr, new_freq=BEATS_SAMPLE_RATE)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        n = waveform.shape[-1]
        if n < self.clip_samples:
            waveform = F.pad(waveform, (0, self.clip_samples - n))
        else:
            waveform = waveform[..., :self.clip_samples]

        return {
            "waveform": waveform.squeeze(0),
            "filename": row["filename"],
        }


# ── Score clips ───────────────────────────────────────────────────────────────

def score_clips(encoder, ar_model, loader, device):
    ar_model.eval()
    encoder.eval()
    scores, filenames = [], []

    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            E = encoder(waveform)
            token_scores = ar_model.nll(E).cpu().numpy()
            for b in range(token_scores.shape[0]):
                scores.append(agg_mean_std(token_scores[b]))
            filenames.extend(batch["filename"])

    return filenames, np.array(scores)


# ── Derive threshold from validation set ─────────────────────────────────────

def compute_val_threshold(encoder, ar_model, config, device, percentile):
    print(f"\nComputing val threshold at p{percentile}...")
    df = pd.read_csv(config["data"]["metadata_file"])
    _, val_df, _ = make_recording_splits(
        df,
        train_frac=config["data"]["train_split"],
        val_frac=config["data"]["val_split"],
        seed=config["project"]["seed"],
    )
    val_dataset = InferenceDataset(
        val_df, config["data"]["root_dir"], config["data"]["clip_duration"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    _, val_scores = score_clips(encoder, ar_model, val_loader, device)
    tau = float(np.percentile(val_scores, percentile))
    print(f"  Val scores — mean: {val_scores.mean():.2f}  std: {val_scores.std():.2f}")
    print(f"  τ (p{percentile}) = {tau:.4f}  "
          f"({int((val_scores >= tau).sum())}/{len(val_scores)} val clips flagged)")
    return tau


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path, checkpoint_path, csv_path, output_path,
         percentile, tau, no_threshold):

    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load models ----
    token_mean = np.load(
        os.path.join(config["training"]["checkpoint_dir"], "token_mean.npy")
    )
    encoder = BEATsEncoder(
        model_path=config["beats"]["model_path"],
        device=str(device),
        token_mean=token_mean,
    )
    ar_cfg = config["ar_cnn"]
    ar_model = ARCNN(
        embed_dim=config["beats"]["embed_dim"],
        hidden_dim=ar_cfg["hidden_dim"],
        n_layers=ar_cfg["n_layers"],
        dilation=ar_cfg["dilation"],
    ).to(device)
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ar_model.load_state_dict(ck["model_state_dict"])
    print(f"Loaded checkpoint — epoch {ck['epoch']}  val NLL {ck['val_nll']:.4f}")

    # ---- Load input CSV ----
    input_df = pd.read_csv(csv_path)
    if "filename" not in input_df.columns:
        raise ValueError(
            f"Input CSV must have a 'filename' column. Found: {list(input_df.columns)}"
        )
    print(f"Input CSV: {len(input_df)} clips  ({csv_path})")

    # ---- Score ----
    dataset = InferenceDataset(
        input_df, config["data"]["root_dir"], config["data"]["clip_duration"]
    )
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    filenames, scores = score_clips(encoder, ar_model, loader, device)

    # ---- Threshold ----
    if not no_threshold:
        if tau is None:
            tau = compute_val_threshold(encoder, ar_model, config, device, percentile)
        else:
            print(f"Using provided τ = {tau:.4f}")

    # ---- Build output ----
    score_map = dict(zip(filenames, scores))
    out_df = input_df.copy()
    out_df["score"] = out_df["filename"].map(score_map)

    if not no_threshold:
        out_df["prediction"] = (out_df["score"] >= tau).astype(int)
        n_flagged = int(out_df["prediction"].sum())
        print(f"\nFlagged {n_flagged}/{len(out_df)} clips "
              f"({100*n_flagged/len(out_df):.1f}%) as anomalous  (τ={tau:.4f})")

    print(f"\nScore distribution:")
    print(f"  min={scores.min():.2f}  p25={np.percentile(scores,25):.2f}  "
          f"median={np.percentile(scores,50):.2f}  "
          f"p75={np.percentile(scores,75):.2f}  max={scores.max():.2f}")

    # ---- Save ----
    out_df = out_df.sort_values("score", ascending=False).reset_index(drop=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nResults → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AR-BEATs inference on a CSV of filenames."
    )
    parser.add_argument("--csv",        required=True,
                        help="Input CSV with a 'filename' column")
    parser.add_argument("--output",     default=None,
                        help="Output CSV path (default: <input>_scores.csv)")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")

    thresh = parser.add_mutually_exclusive_group()
    thresh.add_argument("--percentile", type=float, default=65,
                        help="Val-set percentile for threshold (default: 65)")
    thresh.add_argument("--tau",        type=float, default=None,
                        help="Fixed threshold value")
    thresh.add_argument("--no-threshold", action="store_true",
                        help="Output scores only, no prediction column")

    args = parser.parse_args()

    if args.output is None:
        stem = os.path.splitext(args.csv)[0]
        args.output = f"{stem}_scores.csv"

    main(
        config_path     = args.config,
        checkpoint_path = args.checkpoint,
        csv_path        = args.csv,
        output_path     = args.output,
        percentile      = args.percentile,
        tau             = args.tau,
        no_threshold    = args.no_threshold,
    )
