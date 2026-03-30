"""
evaluation/evaluate.py
----------------------
Evaluates a trained AR-BEATs model on the test set.

Outputs:
  - AUROC and Average Precision (AP)
  - Per-clip scores CSV for further analysis
  - AUROC/PR curve plots saved to results/
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


class Evaluator:
    """
    Args:
        encoder:     BEATsEncoder (frozen)
        ar_model:    ARCNN (loaded from checkpoint)
        test_loader: DataLoader (normal + anomalous clips)
        config:      Full config dict
        device:      torch.device
    """

    def __init__(self, encoder, ar_model, test_loader, config, device):
        self.encoder = encoder
        self.ar_model = ar_model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.results_dir = config["evaluation"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

    # ----------------------------------------------------------
    def evaluate(self) -> dict:
        self.ar_model.eval()
        self.encoder.eval()

        all_scores = []
        all_labels = []
        all_filenames = []

        with torch.no_grad():
            for batch in self.test_loader:
                spec = batch["waveform"].to(self.device)
                labels = batch["label"].numpy()
                filenames = batch["filename"]

                E = self.encoder(spec)                         # (B, H_p, W_p, D)
                scores = - self.ar_model.clip_score(E)           # (B,)
                scores = scores.cpu().numpy()

                all_scores.extend(scores.tolist())
                all_labels.extend(labels.tolist())
                all_filenames.extend(filenames)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # Metrics
        auroc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)

        print(f"\n{'='*40}")
        print(f"  Evaluation Results")
        print(f"{'='*40}")
        print(f"  AUROC : {auroc:.4f}")
        print(f"  AP    : {ap:.4f}")
        print(f"  N clips total    : {len(all_labels)}")
        print(f"  N anomalous      : {all_labels.sum()}")
        print(f"  Positive rate    : {all_labels.mean():.1%}")
        print(f"{'='*40}\n")

        # Save per-clip results
        self._save_scores(all_filenames, all_scores, all_labels)

        # Save curves
        self._plot_roc(all_labels, all_scores, auroc)
        self._plot_pr(all_labels, all_scores, ap)

        return {"auroc": auroc, "ap": ap}

    # ----------------------------------------------------------
    def _save_scores(self, filenames, scores, labels):
        df = pd.DataFrame({
            "filename": filenames,
            "score": scores,
            "label": labels,
        })
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        path = os.path.join(self.results_dir, "clip_scores.csv")
        df.to_csv(path, index=False)
        print(f"Per-clip scores saved to {path}")

    def _plot_roc(self, labels, scores, auroc):
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AR-BEATs (AUROC = {auroc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Underwater Bioacoustic Anomaly Detection")
        plt.legend(loc="lower right")
        plt.tight_layout()
        path = os.path.join(self.results_dir, "roc_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"ROC curve saved to {path}")

    def _plot_pr(self, labels, scores, ap):
        precision, recall, _ = precision_recall_curve(labels, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, lw=2, label=f"AR-BEATs (AP = {ap:.3f})")
        baseline = labels.mean()
        plt.axhline(baseline, color="k", linestyle="--", lw=1,
                    label=f"Baseline (random) = {baseline:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve — Underwater Bioacoustic Anomaly Detection")
        plt.legend(loc="upper right")
        plt.tight_layout()
        path = os.path.join(self.results_dir, "pr_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"PR curve saved to {path}")
