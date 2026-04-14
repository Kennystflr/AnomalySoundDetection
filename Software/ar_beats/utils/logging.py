"""
utils/logging.py
----------------
Training logger — saves loss curves to CSV and PNG.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TrainingLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.records = []

    def log(self, epoch: int, train_nll: float, val_nll: float):
        self.records.append({
            "epoch": epoch,
            "train_nll": train_nll,
            "val_nll": val_nll,
        })

    def save_curves(self):
        df = pd.DataFrame(self.records)
        csv_path = os.path.join(self.log_dir, "training_curves.csv")
        df.to_csv(csv_path, index=False)

        plt.figure(figsize=(8, 4))
        plt.plot(df["epoch"], df["train_nll"], label="Train NLL")
        plt.plot(df["epoch"], df["val_nll"], label="Val NLL")
        best_epoch = df.loc[df["val_nll"].idxmin(), "epoch"]
        plt.axvline(best_epoch, color="red", linestyle="--", lw=1,
                    label=f"Best epoch ({best_epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("NLL")
        plt.title("AR-BEATs Training Curves")
        plt.legend()
        plt.tight_layout()
        png_path = os.path.join(self.log_dir, "training_curves.png")
        plt.savefig(png_path, dpi=150)
        plt.close()

        print(f"\nTraining curves saved to {self.log_dir}")
        print(f"  Best epoch : {best_epoch}")
        print(f"  Best val NLL : {df['val_nll'].min():.4f}")
