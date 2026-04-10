"""
evaluation/evaluate.py
----------------------
Evaluates a trained AR-BEATs model on the test set.
Clip-level score: mean + 2*std over the per-token NLL grid.

Produces figures and tables matching the paper's reporting style:
  - Table I  : Classification report (precision, recall, F1, support per class)
  - Table II : Confusion matrix
  - Fig. 1   : Precision-Recall curve with AUC annotation
  - Fig. 2   : ROC curve

Threshold strategies:
  - Oracle (τ_oracle): maximises F1 on test set — upper bound
  - Deployable (τ_p90): 90th percentile of normal scores — no labels needed
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              roc_curve, precision_recall_curve,
                              f1_score, precision_score, recall_score,
                              confusion_matrix)


# ---------------------------------------------------------------
# Aggregation function — exported so evaluate_loro.py can import it
# ---------------------------------------------------------------

def agg_mean_std(token_scores: np.ndarray, k: float = 2.0) -> float:
    """
    Clip-level anomaly score: mean + k * std over all token NLL scores.

    Captures two complementary signals:
      - mean: average predictive surprise across the clip
      - std:  spatial unevenness (elevated when a localized vocalization
              creates a high-NLL region surrounded by normal background)

    k=2 corresponds to the classical two-sigma threshold (< 2.5% of tokens
    from a Gaussian distribution exceed it by chance).

    Args:
        token_scores: np.ndarray of shape (H_p, W_p) — per-token NLL values.
        k:            multiplier for the standard deviation term. Default: 2.0.

    Returns:
        Scalar clip-level anomaly score.
    """
    return float(token_scores.mean() + k * token_scores.std())


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _oracle_threshold(scores, labels):
    """Return (tau, f1) that maximises F1 across all thresholds."""
    p, r, th = precision_recall_curve(labels, scores)
    f1s = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-8)
    bi  = np.argmax(f1s)
    return float(th[bi]), float(f1s[bi]), float(p[bi]), float(r[bi])


def _p90_threshold(scores, labels):
    """Return 90th percentile of normal scores — deployable, no labels needed."""
    return float(np.percentile(scores[labels == 0], 90))


def _classification_report(labels, preds, threshold, threshold_name):
    """
    Return a dict matching the Perch 2.0 / PyTorch reporting style.
    Classes: 0 = Void (normal), 1 = Anomaly.
    """
    classes = [0, 1]
    names   = ["Void (normal)", "Anomaly"]
    report  = {}
    for cls, name in zip(classes, names):
        mask = (labels == cls)
        cls_preds = preds[mask]
        tp = int((cls_preds == cls).sum())
        support = int(mask.sum())
        report[name] = {
            "precision": float(precision_score(labels, preds, pos_label=cls,
                                               zero_division=0)),
            "recall":    float(recall_score(labels, preds, pos_label=cls,
                                            zero_division=0)),
            "f1":        float(f1_score(labels, preds, pos_label=cls,
                                        zero_division=0)),
            "support":   support,
        }
    report["_meta"] = {
        "threshold":      threshold,
        "threshold_name": threshold_name,
        "weighted_f1":    float(f1_score(labels, preds, average="weighted",
                                         zero_division=0)),
        "weighted_prec":  float(precision_score(labels, preds, average="weighted",
                                                zero_division=0)),
        "weighted_rec":   float(recall_score(labels, preds, average="weighted",
                                             zero_division=0)),
        "accuracy":       float((labels == preds).mean()),
    }
    return report


def _print_report(report, title):
    m = report["_meta"]
    print(f"\n  {title}")
    print(f"  Threshold : {m['threshold']:.4f}  ({m['threshold_name']})")
    print(f"  {'Class':<18} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*52}")
    for cls, vals in report.items():
        if cls == "_meta":
            continue
        print(f"  {cls:<18} {vals['precision']:>10.4f} {vals['recall']:>8.4f} "
              f"{vals['f1']:>8.4f} {vals['support']:>9d}")
    print(f"  {'-'*52}")
    print(f"  {'Weighted avg':<18} {m['weighted_prec']:>10.4f} "
          f"{m['weighted_rec']:>8.4f} {m['weighted_f1']:>8.4f}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")


# ---------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------

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
        self.encoder     = encoder
        self.ar_model    = ar_model
        self.test_loader = test_loader
        self.config      = config
        self.device      = device
        self.results_dir = config["evaluation"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

    # ----------------------------------------------------------
    def evaluate(self) -> dict:
        self.ar_model.eval()
        self.encoder.eval()

        all_token_scores, all_labels, all_filenames = [], [], []

        with torch.no_grad():
            for batch in self.test_loader:
                waveform  = batch["waveform"].to(self.device)
                labels    = batch["label"].numpy()
                filenames = batch["filename"]

                E = self.encoder(waveform)
                token_scores = self.ar_model.nll(E).cpu().numpy()

                for b in range(token_scores.shape[0]):
                    all_token_scores.append(token_scores[b])
                all_labels.extend(labels.tolist())
                all_filenames.extend(filenames)

        all_labels  = np.array(all_labels)
        clip_scores = np.array([agg_mean_std(t) for t in all_token_scores])

        # ── Ranking metrics ────────────────────────────────────────────────────
        auroc = roc_auc_score(all_labels, clip_scores)
        ap    = average_precision_score(all_labels, clip_scores)

        # ── Thresholds ─────────────────────────────────────────────────────────
        tau_oracle, f1_oracle, prec_oracle, rec_oracle = _oracle_threshold(
            clip_scores, all_labels
        )
        tau_p90  = _p90_threshold(clip_scores, all_labels)
        preds_oracle = (clip_scores >= tau_oracle).astype(int)
        preds_p90    = (clip_scores >= tau_p90).astype(int)

        # ── Classification reports ─────────────────────────────────────────────
        report_oracle = _classification_report(
            all_labels, preds_oracle, tau_oracle, "oracle"
        )
        report_p90 = _classification_report(
            all_labels, preds_p90, tau_p90, "deployable p90"
        )

        # ── Print ──────────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  AR-BEATs Evaluation Results")
        print(f"{'='*60}")
        print(f"  N clips     : {len(all_labels)}")
        print(f"  N anomalous : {int(all_labels.sum())}  "
              f"({all_labels.mean():.1%} positive rate)")
        print(f"  AUROC       : {auroc:.4f}")
        print(f"  AP (PR-AUC) : {ap:.4f}   "
              f"(baseline random = {all_labels.mean():.4f})")
        _print_report(report_oracle, "Classification report — oracle threshold")
        _print_report(report_p90,   "Classification report — deployable p90 threshold")
        print(f"\n  NOTE: oracle threshold selected on test set (upper bound).")
        print(f"  Report AUROC and AP as primary metrics for fair comparison.")
        print(f"{'='*60}")

        # ── Save ───────────────────────────────────────────────────────────────
        self._save_scores(all_filenames, all_labels, clip_scores,
                          preds_oracle, preds_p90)
        self._save_report_csv(report_oracle, "classification_report_oracle.csv")
        self._save_report_csv(report_p90,    "classification_report_p90.csv")

        # ── Figures ────────────────────────────────────────────────────────────
        self._plot_pr(all_labels, clip_scores, ap, tau_oracle, f1_oracle)
        self._plot_roc(all_labels, clip_scores, auroc)
        self._plot_confusion_matrix(all_labels, preds_oracle,
                                    f"oracle (τ={tau_oracle:.2f})",
                                    "confusion_matrix_oracle.png")
        self._plot_confusion_matrix(all_labels, preds_p90,
                                    f"p90 deploy. (τ={tau_p90:.2f})",
                                    "confusion_matrix_p90.png")

        return {
            "auroc":       auroc,
            "ap":          ap,
            "f1_oracle":   f1_oracle,
            "tau_oracle":  tau_oracle,
            "prec_oracle": prec_oracle,
            "rec_oracle":  rec_oracle,
            "f1_p90":      report_p90["_meta"]["weighted_f1"],
            "tau_p90":     tau_p90,
        }

    # ----------------------------------------------------------
    def _save_scores(self, filenames, labels, scores, preds_oracle, preds_p90):
        df = pd.DataFrame({
            "filename":     filenames,
            "score":        scores,
            "label":        labels,
            "pred_oracle":  preds_oracle,
            "pred_p90":     preds_p90,
        })
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        path = os.path.join(self.results_dir, "clip_scores.csv")
        df.to_csv(path, index=False)
        print(f"\nPer-clip scores → {path}")

    def _save_report_csv(self, report, filename):
        rows = []
        for cls, vals in report.items():
            if cls == "_meta":
                continue
            rows.append({"class": cls, **vals})
        m = report["_meta"]
        rows.append({
            "class": "weighted avg",
            "precision": m["weighted_prec"],
            "recall": m["weighted_rec"],
            "f1": m["weighted_f1"],
            "support": sum(v["support"] for k, v in report.items() if k != "_meta"),
        })
        pd.DataFrame(rows).to_csv(
            os.path.join(self.results_dir, filename), index=False
        )

    # ----------------------------------------------------------
    def _plot_pr(self, labels, scores, ap, tau_oracle, f1_oracle):
        """
        Precision-Recall curve matching Perch 2.0's Fig. 1 style.
        Annotates the oracle operating point and the p90 deployable point.
        """
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        baseline = labels.mean()

        # Oracle point
        f1arr   = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-8)
        bi      = np.argmax(f1arr)

        # p90 point
        tau_p90 = _p90_threshold(scores, labels)
        preds_p90 = (scores >= tau_p90).astype(int)
        prec_p90  = precision_score(labels, preds_p90, zero_division=0)
        rec_p90   = recall_score(labels, preds_p90, zero_division=0)
        f1_p90    = f1_score(labels, preds_p90, zero_division=0)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, lw=2, color="#1a6eb5",
                label=f"AR-BEATs (AP = {ap:.4f})")
        ax.axhline(baseline, color="k", linestyle="--", lw=1,
                   label=f"Random baseline = {baseline:.4f}")

        # Oracle point
        ax.scatter(recall[bi], precision[bi], color="#d63a2f",
                   s=80, zorder=5,
                   label=f"Oracle  F1={f1_oracle:.4f}  τ={tau_oracle:.2f}")

        # p90 deployable point
        ax.scatter(rec_p90, prec_p90, marker="D", color="#1d9e75",
                   s=70, zorder=5,
                   label=f"Deployable p90  F1={f1_p90:.4f}  τ={tau_p90:.2f}")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_title("Precision-Recall Curve — AR-BEATs\n"
                     "Underwater Bioacoustic Anomaly Detection")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        path = os.path.join(self.results_dir, "pr_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"PR curve → {path}")

    def _plot_roc(self, labels, scores, auroc):
        """ROC curve."""
        fpr, tpr, _ = roc_curve(labels, scores)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, color="#1a6eb5",
                label=f"AR-BEATs (AUROC = {auroc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — AR-BEATs\n"
                     "Underwater Bioacoustic Anomaly Detection")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        path = os.path.join(self.results_dir, "roc_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"ROC curve → {path}")

    def _plot_confusion_matrix(self, labels, preds, threshold_label, filename):
        """
        Visual confusion matrix matching PyTorch Table III style.
        Classes: 0 = Void (normal), 1 = Anomaly.
        """
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()

        total  = len(labels)
        acc    = (tp + tn) / total
        f1     = f1_score(labels, preds, zero_division=0)
        prec   = precision_score(labels, preds, zero_division=0)
        rec    = recall_score(labels, preds, zero_division=0)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                                  gridspec_kw={"width_ratios": [1.4, 1]})

        # ── Left: confusion matrix heatmap ─────────────────────────────────────
        ax = axes[0]
        mat = np.array([[tn, fp], [fn, tp]])
        cax = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted:\nVoid", "Predicted:\nAnomaly"], fontsize=10)
        ax.set_yticklabels(["Actual:\nVoid", "Actual:\nAnomaly"], fontsize=10)
        ax.set_title(f"Confusion Matrix — {threshold_label}", fontsize=11, pad=10)

        # Cell annotations
        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                color = "white" if val > mat.max() * 0.6 else "black"
                label_map = {(0, 0): "TN", (0, 1): "FP",
                             (1, 0): "FN", (1, 1): "TP"}
                ax.text(j, i, f"{label_map[(i,j)]}\n{val}",
                        ha="center", va="center", fontsize=12,
                        fontweight="bold", color=color)

        plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

        # ── Right: metrics table ───────────────────────────────────────────────
        ax2 = axes[1]
        ax2.axis("off")

        col_labels = ["Class", "Prec.", "Recall", "F1", "Support"]
        rows = [
            ["Void (normal)", f"{tn/(tn+fp):.4f}" if (tn+fp) else "—",
             f"{tn/(tn+fn):.4f}" if (tn+fn) else "—",
             f"{2*tn/(2*tn+fp+fn):.4f}" if (2*tn+fp+fn) else "—",
             str(tn + fn)],
            ["Anomaly",       f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
             str(tp + fn)],
            ["Weighted avg",  f"{(prec*(tp+fn)+tn/(tn+fp)*(tn+fn))/total:.4f}",
             f"{acc:.4f}", f"{f1_score(labels,preds,average='weighted',zero_division=0):.4f}",
             str(total)],
        ]

        tbl = ax2.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 1.8)

        # Header style
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor("#1a6eb5")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        # Weighted avg row
        for j in range(len(col_labels)):
            tbl[(3, j)].set_facecolor("#e8f0fb")

        ax2.set_title(f"Metrics — {threshold_label}", fontsize=11, pad=10)

        # Overall accuracy annotation
        ax2.text(0.5, 0.08,
                 f"Accuracy: {acc:.4f}   |   AP (PR-AUC): reported separately",
                 ha="center", va="center", transform=ax2.transAxes,
                 fontsize=8, color="gray")

        plt.tight_layout()
        path = os.path.join(self.results_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix → {path}")