"""
training/trainer.py
-------------------
Training loop for AR-BEATs.

Features:
  - Trains AR CNN on normal clips only
  - Early stopping on val NLL (patience from config)
  - Saves best checkpoint + training curves
  - Logs epoch-level train/val NLL
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from models.ar_cnn import nll_loss
from utils.logging import TrainingLogger


class Trainer:
    """
    Args:
        encoder:    BEATsEncoder (frozen)
        ar_model:   ARCNN
        train_loader: DataLoader (normal clips only)
        val_loader:   DataLoader (normal clips only)
        config:     Full config dict
        device:     torch.device
    """

    def __init__(self, encoder, ar_model, train_loader, val_loader, config, device):
        self.encoder = encoder
        self.ar_model = ar_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        tcfg = config["training"]

        self.optimizer = AdamW(
            ar_model.parameters(),
            lr=tcfg["lr"],
            betas=tuple(tcfg["betas"]),
            weight_decay=tcfg["weight_decay"],
        )

        self.max_epochs = tcfg["max_epochs"]
        self.patience = tcfg["early_stopping_patience"]
        self.checkpoint_dir = tcfg["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.logger = TrainingLogger(log_dir=tcfg["log_dir"])

        # Early stopping state
        self.best_val_nll = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    # ----------------------------------------------------------
    def train(self):
        print(f"\n{'='*60}")
        print(f"  AR-BEATs Training")
        print(f"  Max epochs : {self.max_epochs}")
        print(f"  Patience   : {self.patience}")
        print(f"  Device     : {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            train_nll = self._run_epoch(self.train_loader, train=True)
            val_nll = self._run_epoch(self.val_loader, train=False)

            elapsed = time.time() - t0
            self.logger.log(epoch, train_nll, val_nll)

            print(
                f"Epoch {epoch:4d}/{self.max_epochs} | "
                f"Train NLL: {train_nll:.4f} | "
                f"Val NLL:   {val_nll:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Early stopping
            if val_nll < self.best_val_nll:
                self.best_val_nll = val_nll
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_nll, is_best=True)
                print(f"  ✓ New best val NLL: {val_nll:.4f} — checkpoint saved.")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(
                        f"\n  Early stopping at epoch {epoch}. "
                        f"Best was epoch {self.best_epoch} "
                        f"(val NLL = {self.best_val_nll:.4f})."
                    )
                    break

        self.logger.save_curves()
        print(f"\nTraining complete. Best epoch: {self.best_epoch}")
        print(f"Best val NLL: {self.best_val_nll:.4f}")
        print(
            f"\n[TODO — ASD pre-submission] Update max_epochs in config.yaml "
            f"with actual stopping epoch: {self.best_epoch}"
        )
        return self.best_epoch

    # ----------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.ar_model.train(train)
        total_nll = 0.0
        n_batches = 0

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for batch in loader:
                spec = batch["waveform"].to(self.device)  # (B, F, T)

                # Extract BEATs tokens (no grad through encoder)
                with torch.no_grad():
                    E = self.encoder(spec)  # (B, H_p, W_p, D)

                mu, log_var = self.ar_model(E)
                loss = nll_loss(E, mu, log_var)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        self.ar_model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                total_nll += loss.item()
                n_batches += 1

        return total_nll / n_batches

    # ----------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_nll: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "val_nll": val_nll,
            "model_state_dict": self.ar_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        path = os.path.join(self.checkpoint_dir, "best.pt" if is_best else f"epoch_{epoch}.pt")
        torch.save(state, path)
