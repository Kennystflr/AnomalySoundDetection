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
import torch
from torch.utils.data import DataLoader

from data.dataset import UnderwaterAudioDataset, compute_token_mean
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

    # ---- Receptive field info ----
    ar_cfg = config["ar_cnn"]
    rf = compute_receptive_field(ar_cfg["n_layers"], ar_cfg["dilation"])
    print(f"\nAR CNN receptive field : {rf['total_tokens']} tokens = {rf['time_s']:.1f}s")
    print(f"Token grid per clip    : 31 × 8 = 248 tokens (H_p × W_p)")
    print(f"Each token covers      : 160ms × 16 mel bins")

    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)

    # ---- Step 1: load BEATs (needed for token mean computation) ----
    print("\nStep 1: Loading BEATs encoder (frozen)...")
    encoder = BEATsEncoder(
        model_path=config["beats"]["model_path"],
        device=str(device),
        token_mean=None,   # no normalization yet
    )

    # ---- Step 2: compute token-level mean on raw training set ----
    token_mean_path = os.path.join(config["training"]["checkpoint_dir"], "token_mean.npy")

    if os.path.exists(token_mean_path):
        print(f"\nStep 2: Loading existing token mean from {token_mean_path}")
        token_mean = np.load(token_mean_path)
    else:
        print("\nStep 2: Computing token normalization mean...")
        raw_train = UnderwaterAudioDataset(
            metadata_file=config["data"]["metadata_file"],
            root_dir=config["data"]["root_dir"],
            config=config,
            split="train",
            token_mean=None,
        )
        token_mean = compute_token_mean(raw_train, encoder, device, batch_size=16)
        np.save(token_mean_path, token_mean)
        print(f"  Saved token_mean.npy  shape: {token_mean.shape}")

    # Apply normalization mean to encoder
    encoder.set_token_mean(token_mean)

    # ---- Step 3: build normalized datasets ----
    print("\nStep 3: Building datasets...")
    train_dataset = UnderwaterAudioDataset(
        config["data"]["metadata_file"], config["data"]["root_dir"],
        config, split="train", token_mean=token_mean,
    )
    val_dataset = UnderwaterAudioDataset(
        config["data"]["metadata_file"], config["data"]["root_dir"],
        config, split="val", token_mean=token_mean,
    )

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

    print(f"  Train clips : {len(train_dataset)}")
    print(f"  Val clips   : {len(val_dataset)}")

    # ---- Step 4: build AR model ----
    print("\nStep 4: Building AR CNN...")
    ar_model = ARCNN(
        embed_dim=config["beats"]["embed_dim"],
        hidden_dim=ar_cfg["hidden_dim"],
        n_layers=ar_cfg["n_layers"],
        dilation=ar_cfg["dilation"],
    ).to(device)

    total_params = sum(p.numel() for p in ar_model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # ---- Step 5: train ----
    trainer = Trainer(encoder, ar_model, train_loader, val_loader, config, device)
    best_epoch = trainer.train()

    print(f"\nDone. Best checkpoint: {tcfg['checkpoint_dir']}/best.pt")
    print(f"[TODO — ASD pre-submission] Update max_epochs → {best_epoch} in config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)