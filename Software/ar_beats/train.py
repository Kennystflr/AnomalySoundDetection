"""
train.py
--------
Entry point for training AR-BEATs.

Usage:
    python train.py --config configs/config.yaml
"""

import argparse
import yaml
import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import (UnderwaterAudioDataset, make_recording_splits,
                           compute_token_mean)
from models.beats_encoder import BEATsEncoder
from models.ar_cnn import ARCNN, compute_receptive_field
from training.trainer import Trainer
from utils.seed import set_seed


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU found — training on CPU will be slow.")

    ar_cfg = config["ar_cnn"]
    rf = compute_receptive_field(ar_cfg["n_layers"], ar_cfg["dilation"])
    print(f"\nAR CNN receptive field : {rf['total_tokens']} tokens = {rf['time_s']:.1f}s")
    print(f"Token grid per clip    : 31 × 8 = 248 tokens (H_p × W_p)")
    print(f"Each token covers      : 160ms × 16 mel bins")

    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)

    # ---- Step 1: recording-stratified split ----
    print("\nStep 1: Building recording-stratified splits...")
    df = pd.read_csv(config["data"]["metadata_file"])
    train_df, val_df, test_df = make_recording_splits(
        df,
        train_frac=config["data"]["train_split"],
        val_frac=config["data"]["val_split"],
        seed=config["project"]["seed"],
    )
    # Save split assignments for reproducibility
    split_path = os.path.join(config["training"]["checkpoint_dir"], "splits.csv")
    df_split = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ], ignore_index=True)
    df_split.to_csv(split_path, index=False)
    print(f"  Split assignments saved to {split_path}")

    # ---- Step 2: load BEATs ----
    print("\nStep 2: Loading BEATs encoder (frozen)...")
    encoder = BEATsEncoder(
        model_path=config["beats"]["model_path"],
        device=str(device),
        token_mean=None,
    )

    # ---- Step 3: compute token mean on training set ----
    token_mean_path = os.path.join(
        config["training"]["checkpoint_dir"], "token_mean.npy"
    )
    if os.path.exists(token_mean_path):
        print(f"\nStep 3: Loading existing token mean from {token_mean_path}")
        token_mean = np.load(token_mean_path)
    else:
        print("\nStep 3: Computing token normalization mean...")
        raw_train = UnderwaterAudioDataset(train_df, config["data"]["root_dir"], config)
        token_mean = compute_token_mean(raw_train, encoder, device, batch_size=16)
        np.save(token_mean_path, token_mean)
        print(f"  Saved token_mean.npy  shape: {token_mean.shape}")

    encoder.set_token_mean(token_mean)

    # ---- Step 4: build datasets and loaders ----
    print("\nStep 4: Building datasets...")
    train_dataset = UnderwaterAudioDataset(train_df, config["data"]["root_dir"], config)
    val_dataset   = UnderwaterAudioDataset(val_df,   config["data"]["root_dir"], config)

    tcfg = config["training"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg["num_workers"],
        pin_memory=tcfg["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["num_workers"],
        pin_memory=tcfg["pin_memory"],
    )

    # ---- Step 5: build AR model ----
    print("\nStep 5: Building AR CNN...")
    ar_model = ARCNN(
        embed_dim=config["beats"]["embed_dim"],
        hidden_dim=ar_cfg["hidden_dim"],
        n_layers=ar_cfg["n_layers"],
        dilation=ar_cfg["dilation"],
    ).to(device)

    total_params = sum(p.numel() for p in ar_model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ---- Step 6: train ----
    trainer = Trainer(encoder, ar_model, train_loader, val_loader, config, device)
    best_epoch = trainer.train()

    print(f"\nDone. Best checkpoint: {tcfg['checkpoint_dir']}/best.pt")
    print(f"[TODO — ASD pre-submission] Update max_epochs → {best_epoch} in config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)