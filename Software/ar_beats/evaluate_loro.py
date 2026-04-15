"""
evaluate_loro.py
----------------
Leave-One-Recording-Out (LORO) cross-validation for AR-BEATs.

Fully dataset-agnostic: the number of folds, recording IDs, and
clip counts are all discovered from the metadata file at runtime.
Works identically whether you have 8 normal recordings or 80.

Protocol (one fold = one held-out normal recording):
  - Discover all normal-only recordings in the metadata.
  - For each fold k:
      1. Train on normal clips from all other normal recordings.
      2. Calibrate an unsupervised threshold from the validation scores.
      3. Evaluate on the held-out normal recording + all anomalous clips.
  - Average metrics across folds and report mean ± std.

Threshold strategies:
  VAL-CALIBRATED (deployable, no labels):
      τ = mean(val_scores) + k × std(val_scores)
      Clips scoring above τ are predicted anomalous.
      Tested at k=2 and k=3.

  ORACLE (upper bound, requires test labels):
      τ selected to maximise F1 on the test set.
      Not deployable — reported as a theoretical upper bound.

Scoring: mean + 2 × std over the 248-token NLL grid (mean+2std).

Usage:
    python evaluate_loro.py --config configs/config.yaml
    python evaluate_loro.py --config configs/config.yaml --k_vals 2 3 4
    python evaluate_loro.py --config configs/config.yaml --force_retrain
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, f1_score)

from data.dataset import (UnderwaterAudioDataset, _extract_recording_id,
                           compute_token_mean)
from models.beats_encoder import BEATsEncoder
from models.ar_cnn import ARCNN
from training.trainer import Trainer
from evaluation.evaluate import agg_mean_std
from utils.seed import set_seed


# ---------------------------------------------------------------
# Dataset discovery — fully agnostic
# ---------------------------------------------------------------

def discover_recordings(df: pd.DataFrame, min_clips: int = 36):
    """
    Discover recording structure from metadata.

    Args:
        df:         Full metadata DataFrame with filename and label columns.
        min_clips:  Minimum number of clips a normal recording must have
                    to be included as a LORO fold. Corresponds to a minimum
                    recording duration of min_clips × 5s. Recordings below
                    this threshold are excluded because their test sets are
                    dominated by anomalous clips, making AP and AUROC
                    nearly identical to the random baseline regardless of
                    model quality. Default: 36 (= 3 minutes at 5s/clip).

    Returns:
        normal_only_recs: list of qualifying recording IDs (label=0 only,
                          clip count >= min_clips).
        excluded_recs:    list of recordings excluded due to small size.
        anomalous_df:     DataFrame of all anomalous clips (always in test).
    """
    df = df.copy()
    df["recording_id"] = df["filename"].apply(_extract_recording_id)
    rec_max   = df.groupby("recording_id")["label"].max()
    rec_count = df.groupby("recording_id")["filename"].count()

    all_normal = rec_max[rec_max == 0].index
    qualifying  = [r for r in all_normal if rec_count[r] >= min_clips]
    excluded    = [r for r in all_normal if rec_count[r] <  min_clips]
    qualifying  = sorted(qualifying)

    anomalous_df = df[df["label"] == 1].copy()

    if excluded:
        print(f"\n  Excluded from LORO (< {min_clips} clips):")
        for r in excluded:
            n = rec_count[r]
            baseline_ap = len(anomalous_df) / (len(anomalous_df) + n)
            print(f"    {r}  clips={n}  baseline_AP={baseline_ap:.4f}")

    return qualifying, excluded, anomalous_df, df


# ---------------------------------------------------------------
# Forward pass — returns token-level NLL grid
# ---------------------------------------------------------------

def run_forward(encoder, ar_model, loader, device):
    """
    Single forward pass over a DataLoader.

    Returns:
        clip_scores: np.ndarray (N,)  — mean+2std aggregated scores
        labels:      np.ndarray (N,)  — 0/1
        filenames:   list[str]
    """
    ar_model.eval()
    encoder.eval()
    all_scores, all_labels, all_files = [], [], []

    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            E = encoder(waveform)                          # (B, H_p, W_p, D)
            token_scores = ar_model.nll(E).cpu().numpy()  # (B, H_p, W_p)
            clip_scores  = np.array([agg_mean_std(t) for t in token_scores])

            all_scores.extend(clip_scores.tolist())
            all_labels.extend(batch["label"].numpy().tolist())
            all_files.extend(batch["filename"])

    return np.array(all_scores), np.array(all_labels), all_files


# ---------------------------------------------------------------
# Threshold strategies
# ---------------------------------------------------------------

def val_calibrated_threshold(val_scores: np.ndarray, k: float) -> float:
    """
    Unsupervised threshold: mean + k*std of normal validation scores.
    No anomalous labels required — fully deployable.
    """
    return float(val_scores.mean() + k * val_scores.std())


def oracle_threshold(test_scores: np.ndarray, test_labels: np.ndarray) -> float:
    """
    Select threshold maximising F1 on test set.
    Requires test labels — upper bound, not for deployment.
    """
    _, _, thresholds = precision_recall_curve(test_labels, test_scores)
    best_f1, best_thresh = 0.0, thresholds[0]
    for tau in thresholds:
        preds = (test_scores >= tau).astype(int)
        if preds.sum() == 0:
            continue
        f = f1_score(test_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, tau
    return float(best_thresh)


def evaluate_at_threshold(scores, labels, tau):
    preds = (scores >= tau).astype(int)
    if preds.sum() == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    from sklearn.metrics import precision_score, recall_score
    return {
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
    }


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------

def compute_ranking_metrics(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return {"auroc": float("nan"), "ap": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap":    float(average_precision_score(labels, scores)),
    }


# ---------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------

def run_fold(fold_idx, held_out_rec, train_df, val_df, test_df,
             config, device, k_vals, force_retrain):

    tcfg   = config["training"]
    ar_cfg = config["ar_cnn"]
    ckpt_dir = tcfg["checkpoint_dir"]

    # ---- Load BEATs ----
    encoder = BEATsEncoder(
        model_path=config["beats"]["model_path"],
        device=str(device),
        token_mean=None,
    )

    # Cache keyed by recording ID, not fold index.
    # Keying by index is wrong: changing min_clips shifts which recordings
    # qualify as folds, so fold 0 would map to a different recording and
    # load incorrect cached weights.
    safe_rec  = held_out_rec.replace("/", "_").replace(" ", "_")
    mean_path = os.path.join(ckpt_dir, f"token_mean_rec_{safe_rec}.npy")
    fold_ckpt = os.path.join(ckpt_dir, f"best_rec_{safe_rec}.pt")
    if os.path.exists(mean_path) and not force_retrain:
        token_mean = np.load(mean_path)
    else:
        raw_ds = UnderwaterAudioDataset(train_df, config["data"]["root_dir"], config)
        token_mean = compute_token_mean(raw_ds, encoder, device, batch_size=16)
        np.save(mean_path, token_mean)
    encoder.set_token_mean(token_mean)

    # ---- DataLoaders ----
    def make_loader(df, shuffle=False, batch_size=32):
        ds = UnderwaterAudioDataset(df, config["data"]["root_dir"], config)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=tcfg["num_workers"], pin_memory=tcfg["pin_memory"],
            drop_last=(shuffle),
        )

    train_loader = make_loader(train_df, shuffle=True,
                               batch_size=tcfg["batch_size"])
    val_loader   = make_loader(val_df)
    test_loader  = make_loader(test_df)

    # ---- AR model ----
    ar_model = ARCNN(
        embed_dim=config["beats"]["embed_dim"],
        hidden_dim=ar_cfg["hidden_dim"],
        n_layers=ar_cfg["n_layers"],
        dilation=ar_cfg["dilation"],
    ).to(device)

    fold_ckpt = os.path.join(ckpt_dir, f"best_rec_{safe_rec}.pt")
    if os.path.exists(fold_ckpt) and not force_retrain:
        print(f"  Loading cached checkpoint: {fold_ckpt}")
        ckpt = torch.load(fold_ckpt, map_location=device)
        ar_model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = ckpt.get("epoch", "?")
    else:
        trainer = Trainer(encoder, ar_model, train_loader, val_loader,
                          config, device)
        trainer._fold_ckpt_name = f"best_rec_{safe_rec}.pt"
        best_epoch = trainer.train()

    # ---- Score val set (normal only — for threshold calibration) ----
    val_scores, _, _ = run_forward(encoder, ar_model, val_loader, device)

    # ---- Score test set ----
    test_scores, test_labels, test_files = run_forward(
        encoder, ar_model, test_loader, device
    )

    # ---- Ranking metrics (threshold-free) ----
    ranking = compute_ranking_metrics(test_scores, test_labels)

    # ---- Val-calibrated thresholds ----
    results = {
        "held_out_rec":   held_out_rec,
        "fold":           fold_idx,
        "n_train":        len(train_df),
        "n_val":          len(val_df),
        "n_test_normal":  int((test_labels == 0).sum()),
        "n_test_anom":    int((test_labels == 1).sum()),
        "best_epoch":     best_epoch,
        "auroc":          ranking["auroc"],
        "ap":             ranking["ap"],
        "val_mean_score": float(val_scores.mean()),
        "val_std_score":  float(val_scores.std()),
    }

    for k in k_vals:
        tau = val_calibrated_threshold(val_scores, k)
        m   = evaluate_at_threshold(test_scores, test_labels, tau)
        results[f"thresh_k{k}"]     = tau
        results[f"f1_k{k}"]         = m["f1"]
        results[f"precision_k{k}"]  = m["precision"]
        results[f"recall_k{k}"]     = m["recall"]

    # ---- Oracle threshold ----
    tau_oracle = oracle_threshold(test_scores, test_labels)
    m_oracle   = evaluate_at_threshold(test_scores, test_labels, tau_oracle)
    results["thresh_oracle"]     = tau_oracle
    results["f1_oracle"]         = m_oracle["f1"]
    results["precision_oracle"]  = m_oracle["precision"]
    results["recall_oracle"]     = m_oracle["recall"]

    # ---- Save per-clip predictions for this fold ----
    fold_preds = pd.DataFrame({
        "filename": test_files,
        "score":    test_scores,
        "label":    test_labels,
    })
    for k in k_vals:
        tau = results[f"thresh_k{k}"]
        fold_preds[f"pred_k{k}"] = (test_scores >= tau).astype(int)
    fold_preds["pred_oracle"] = (test_scores >= tau_oracle).astype(int)

    return results, fold_preds


# ---------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------

def print_summary(results_df, k_vals):
    n = len(results_df)
    sep = "=" * 66

    print(f"\n{sep}")
    print(f"  LORO Summary  ({n} folds)")
    print(sep)
    print(f"  {'Metric':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*58}")

    def row(label, col):
        s = results_df[col].dropna()
        print(f"  {label:<28} {s.mean():>8.4f} {s.std():>8.4f} "
              f"{s.min():>8.4f} {s.max():>8.4f}")

    row("AUROC (ranking)",  "auroc")
    row("AP (ranking)",     "ap")
    print(f"  {'-'*58}")
    for k in k_vals:
        row(f"F1  (τ = val+{k}σ)",    f"f1_k{k}")
        row(f"Prec (τ = val+{k}σ)",   f"precision_k{k}")
        row(f"Rec  (τ = val+{k}σ)",   f"recall_k{k}")
        print(f"  {'-'*58}")
    row("F1   (oracle τ)",  "f1_oracle")
    row("Prec (oracle τ)",  "precision_oracle")
    row("Rec  (oracle τ)",  "recall_oracle")
    print(sep)

    print(f"\n  Per-fold breakdown:")
    header = f"  {'Recording':<25} {'AUROC':>6} {'AP':>6}"
    for k in k_vals:
        header += f"  {'F1-k'+str(k):>6}"
    header += f"  {'F1-orc':>6}  {'τ-k'+str(k_vals[0]):>8}"
    print(header)
    print(f"  {'-'*78}")
    for _, r in results_df.iterrows():
        line = f"  {r['held_out_rec']:<25} {r['auroc']:>6.3f} {r['ap']:>6.3f}"
        for k in k_vals:
            line += f"  {r[f'f1_k{k}']:>6.3f}"
        line += f"  {r['f1_oracle']:>6.3f}  {r[f'thresh_k{k_vals[0]}']:>8.2f}"
        print(line)
    print()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def run_loro(config_path: str, k_vals: list, force_retrain: bool,
             min_clips: int = 30):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    results_dir = config["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # ---- Discover dataset structure ----
    df = pd.read_csv(config["data"]["metadata_file"])
    normal_recs, excluded_recs, anomalous_df, df = discover_recordings(
        df, min_clips=min_clips
    )

    print(f"\n{'='*60}")
    print(f"  Leave-One-Recording-Out Evaluation")
    print(f"  Dataset        : {config['data']['metadata_file']}")
    print(f"  Min clips/fold : {min_clips}")
    print(f"  Qualifying recs: {len(normal_recs)}  (= number of folds)")
    print(f"  Excluded recs  : {len(excluded_recs)}  (too few clips)")
    print(f"  Anomalous clips: {len(anomalous_df)}  (always in test)")
    print(f"  k_vals         : {k_vals}")
    print(f"  Scoring        : mean + 2×std over 248 tokens")
    print(f"{'='*60}\n")

    all_fold_results = []
    all_fold_preds   = []

    for fold_idx, held_out_rec in enumerate(normal_recs):
        print(f"\n--- Fold {fold_idx + 1}/{len(normal_recs)} "
              f"| held-out: {held_out_rec} ---")

        # All clips from other normal recordings → train+val
        other_normal = df[
            (df["label"] == 0) &
            (df["recording_id"] != held_out_rec)
        ].copy().reset_index(drop=True)

        # 10% of other normal → val (for early stopping + threshold calibration)
        n_val = max(2, int(len(other_normal) * 0.10))
        val_df   = other_normal.iloc[-n_val:].copy()
        train_df = other_normal.iloc[:-n_val].copy()

        # Held-out normal + all anomalous → test
        held_df = df[
            (df["label"] == 0) &
            (df["recording_id"] == held_out_rec)
        ].copy()
        test_df = pd.concat([held_df, anomalous_df], ignore_index=True)

        print(f"  Train : {len(train_df):4d} clips | "
              f"Val: {len(val_df):4d} clips | "
              f"Test: {len(test_df):4d} clips "
              f"({len(held_df)} normal + {len(anomalous_df)} anomalous)")

        fold_result, fold_preds = run_fold(
            fold_idx, held_out_rec, train_df, val_df, test_df,
            config, device, k_vals, force_retrain,
        )
        fold_preds["fold"] = fold_idx
        fold_preds["held_out_rec"] = held_out_rec

        all_fold_results.append(fold_result)
        all_fold_preds.append(fold_preds)

        # Print fold summary
        r = fold_result
        print(f"  AUROC={r['auroc']:.4f}  AP={r['ap']:.4f}  "
              + "  ".join(f"F1-k{k}={r[f'f1_k{k}']:.4f}" for k in k_vals)
              + f"  F1-oracle={r['f1_oracle']:.4f}")
        # print(f"  Val threshold (k=2): {r.get('thresh_k2', 'n/a'):.2f}  "
        #       f"Oracle threshold: {r['thresh_oracle']:.2f}")

    # ---- Aggregate ----
    results_df = pd.DataFrame(all_fold_results)
    preds_df   = pd.concat(all_fold_preds, ignore_index=True)

    # ---- Save ----
    results_path = os.path.join(results_dir, "loro_results.csv")
    preds_path   = os.path.join(results_dir, "loro_predictions.csv")
    results_df.to_csv(results_path, index=False)
    preds_df.to_csv(preds_path, index=False)

    print_summary(results_df, k_vals)

    print(f"  Per-fold metrics   → {results_path}")
    print(f"  Per-clip predictions → {preds_path}\n")

    return results_df


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leave-One-Recording-Out evaluation for AR-BEATs"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--k_vals", type=float, nargs="+", default=[2.0, 3.0],
                        help="Values of k for val-calibrated threshold "
                             "(τ = val_mean + k*val_std). Default: 2 3")
    parser.add_argument("--min_clips", type=int, default=36,
                        help="Minimum clips a normal recording must have to "
                             "be included as a LORO fold. Corresponds to a "
                             "minimum recording duration of min_clips * 5s. "
                             "Default: 36 (= 3 minutes at 5s per clip).")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Re-train all folds even if checkpoints exist")
    args = parser.parse_args()

    run_loro(args.config, args.k_vals, args.force_retrain, args.min_clips)