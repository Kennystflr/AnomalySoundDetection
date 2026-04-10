"""
evaluate.py
-----------
Entry point for evaluating a trained AR-BEATs checkpoint.

Usage:
    python evaluate.py --config configs/config.yaml \
                       --checkpoint checkpoints/best.pt
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset import UnderwaterAudioDataset, make_recording_splits
from models.beats_encoder import BEATsEncoder
from models.ar_cnn import ARCNN
from evaluation.evaluate import Evaluator
from utils.seed import set_seed


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load token mean ----
    token_mean = np.load(
        os.path.join(config["training"]["checkpoint_dir"], "token_mean.npy")
    )
    print(f"Loaded token_mean  shape: {token_mean.shape}")

    # ---- Reconstruct the same recording-stratified split ----
    df = pd.read_csv(config["data"]["metadata_file"])
    _, _, test_df = make_recording_splits(
        df,
        train_frac=config["data"]["train_split"],
        val_frac=config["data"]["val_split"],
        seed=config["project"]["seed"],
    )

    test_dataset = UnderwaterAudioDataset(
        test_df, config["data"]["root_dir"], config
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
    )
    print(f"Test clips: {len(test_dataset)}")

    # ---- Load models ----
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ar_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val NLL = {checkpoint['val_nll']:.4f})")

    # ---- Evaluate ----
    evaluator = Evaluator(encoder, ar_model, test_loader, config, device)
    results = evaluator.evaluate()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)