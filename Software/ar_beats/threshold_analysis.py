"""
utils/threshold_analysis.py
----------------------------
Post-processing score refinement — no retraining required.

Implements two strategies on top of existing clip_scores.csv:

  1. PER-RECORDING Z-SCORE NORMALISATION
     Subtract each recording's own mean and divide by its std before
     applying a threshold. Makes the threshold recording-agnostic:
     instead of "is this clip anomalous in absolute NLL terms", asks
     "is this clip unusual relative to other clips in the same recording."
     Directly addresses ambient noise level differences across recordings.

  2. PERCENTILE-BASED THRESHOLD
     Set τ = p-th percentile of normal clip scores instead of
     val_mean + k*std. More robust to skewed score distributions
     and adapts to the actual score range in the data.
     Tested at p=90, p=95, p=99.

Both strategies are evaluated against the oracle threshold for comparison.

Usage:
    python utils/threshold_analysis.py --scores results/clip_scores.csv
    python utils/threshold_analysis.py --scores results/clip_scores.csv --percentiles 90 95 99
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, f1_score,
                              precision_score, recall_score)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def extract_recording_id(filename: str) -> str:
    """Extract recording ID from clip filename (everything before _chunk)."""
    import re
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"(.+)_chunk\d+$", stem)
    return match.group(1) if match else stem


def metrics_at_threshold(scores, labels, tau):
    mask  = ~np.isnan(scores)
    scores, labels = scores[mask], labels[mask]
    preds = (scores >= tau).astype(int)
    if preds.sum() == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "n_flagged": 0, "tau": tau}
    return {
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "n_flagged": int(preds.sum()),
        "tau":       float(tau),
    }


def oracle_metrics(scores, labels):
    """Best F1 achievable across all thresholds (requires labels)."""
    mask = ~np.isnan(scores)
    scores, labels = scores[mask], labels[mask]
    p, r, th = precision_recall_curve(labels, scores)
    f1s = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-8)
    bi  = np.argmax(f1s)
    return {
        "f1":        float(f1s[bi]),
        "precision": float(p[bi]),
        "recall":    float(r[bi]),
        "n_flagged": int((scores >= th[bi]).sum()),
        "tau":       float(th[bi]),
    }



# ---------------------------------------------------------------
# Strategy 1 — Per-recording z-score normalisation
# ---------------------------------------------------------------

def normalise_per_recording(df: pd.DataFrame,
                             score_col: str = "score") -> pd.Series:
    """
    Z-score normalise scores within each recording.

    Edge cases:
      - Single-clip recording: std=0 → score set to 0.0 (neither high nor low)
      - Recording with identical scores: same as above
      - NaN in input scores: propagated as NaN (flagged and dropped downstream)
    """
    normed = df[score_col].copy().astype(float)
    for rec_id, group in df.groupby("recording_id"):
        mu  = group[score_col].mean()
        std = group[score_col].std()
        if pd.isna(std) or std < 1e-8:
            # Single clip or zero variance — assign neutral score
            normed.loc[group.index] = 0.0
        else:
            normed.loc[group.index] = (group[score_col] - mu) / std

    # Final safety check — report and fill any remaining NaNs
    n_nan = normed.isna().sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN scores after normalisation "
              f"— replacing with 0.0 (neutral)")
        normed = normed.fillna(0.0)

    return normed


def safe_ranking_metrics(scores, labels):
    """Ranking metrics with NaN guard."""
    mask = ~np.isnan(scores)
    if mask.sum() < len(scores):
        print(f"  WARNING: dropping {(~mask).sum()} NaN scores before metric computation")
    s, l = scores[mask], labels[mask]
    if l.sum() == 0 or l.sum() == len(l):
        return {"auroc": float("nan"), "ap": float("nan")}
    return {
        "auroc": float(roc_auc_score(l, s)),
        "ap":    float(average_precision_score(l, s)),
    }


# ---------------------------------------------------------------
# Strategy 2 — Percentile threshold
# ---------------------------------------------------------------

def percentile_threshold(scores: np.ndarray, labels: np.ndarray,
                          p: float) -> float:
    """
    Set threshold at the p-th percentile of NORMAL clip scores.
    NaN scores are excluded before computing the percentile.
    """
    normal_scores = scores[(labels == 0) & ~np.isnan(scores)]
    return float(np.percentile(normal_scores, p))


# ---------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------

def analyse(scores_path: str, percentiles: list, results_dir: str):
    df = pd.read_csv(scores_path)

    # Ensure we have a recording_id column
    if "recording_id" not in df.columns:
        df["recording_id"] = df["filename"].apply(extract_recording_id)

    raw_scores = df["score"].values
    labels     = df["label"].values

    print(f"\n{'='*64}")
    print(f"  Score refinement analysis")
    print(f"  Input : {scores_path}")
    print(f"  Clips : {len(df)}  |  Anomalous: {labels.sum()}  "
          f"| Positive rate: {labels.mean():.1%}")
    print(f"{'='*64}\n")

    results = {}

    # ── Baseline: raw scores ──────────────────────────────────────────────────
    rank_raw = safe_ranking_metrics(raw_scores, labels)
    ora_raw  = oracle_metrics(raw_scores, labels)
    results["raw_oracle"] = {**rank_raw, **ora_raw, "strategy": "raw (oracle)"}
    print(f"  Raw scores (baseline):")
    print(f"    AUROC={rank_raw['auroc']:.4f}  AP={rank_raw['ap']:.4f}  "
          f"F1-oracle={ora_raw['f1']:.4f}  "
          f"prec={ora_raw['precision']:.4f}  rec={ora_raw['recall']:.4f}")
    print()

    # ── Strategy 2: percentile thresholds on RAW scores ──────────────────────
    print(f"  Strategy 2 — Percentile threshold on raw scores:")
    for p in percentiles:
        tau = percentile_threshold(raw_scores, labels, p)
        m   = metrics_at_threshold(raw_scores, labels, tau)
        key = f"raw_p{int(p)}"
        results[key] = {**rank_raw, **m, "strategy": f"raw p{int(p)}"}
        print(f"    p{int(p):3d}  τ={tau:8.2f}  "
              f"F1={m['f1']:.4f}  prec={m['precision']:.4f}  "
              f"rec={m['recall']:.4f}  flagged={m['n_flagged']}")
    print()

    # ── Strategy 1: per-recording normalisation ───────────────────────────────
    df["score_normed"] = normalise_per_recording(df, "score")
    norm_scores = df["score_normed"].values
    rank_norm   = safe_ranking_metrics(norm_scores, labels)
    ora_norm    = oracle_metrics(norm_scores, labels)
    results["norm_oracle"] = {**rank_norm, **ora_norm,
                               "strategy": "per-rec norm (oracle)"}

    print(f"  Strategy 1 — Per-recording z-score normalisation:")
    print(f"    AUROC={rank_norm['auroc']:.4f}  AP={rank_norm['ap']:.4f}  "
          f"F1-oracle={ora_norm['f1']:.4f}  "
          f"prec={ora_norm['precision']:.4f}  rec={ora_norm['recall']:.4f}")

    # Percentile thresholds on normalised scores
    print(f"    Percentile thresholds on normalised scores:")
    for p in percentiles:
        tau = percentile_threshold(norm_scores, labels, p)
        m   = metrics_at_threshold(norm_scores, labels, tau)
        key = f"norm_p{int(p)}"
        results[key] = {**rank_norm, **m,
                        "strategy": f"per-rec norm p{int(p)}"}
        print(f"      p{int(p):3d}  τ={tau:6.3f}  "
              f"F1={m['f1']:.4f}  prec={m['precision']:.4f}  "
              f"rec={m['recall']:.4f}  flagged={m['n_flagged']}")
    print()

    # ── Best result summary ───────────────────────────────────────────────────
    best_key = max(
        (k for k in results if "oracle" not in k),
        key=lambda k: results[k]["f1"]
    )
    best = results[best_key]
    print(f"  Best non-oracle F1: {best['f1']:.4f}  "
          f"strategy={best['strategy']}  "
          f"prec={best['precision']:.4f}  rec={best['recall']:.4f}")
    print()

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)

    # Predictions CSV with all strategies
    out = df[["filename", "recording_id", "score", "score_normed", "label"]].copy()
    for p in percentiles:
        tau_raw  = percentile_threshold(raw_scores,  labels, p)
        tau_norm = percentile_threshold(norm_scores, labels, p)
        out[f"pred_raw_p{int(p)}"]  = (raw_scores  >= tau_raw).astype(int)
        out[f"pred_norm_p{int(p)}"] = (norm_scores >= tau_norm).astype(int)
    pred_path = os.path.join(results_dir, "clip_scores_refined.csv")
    out.to_csv(pred_path, index=False)
    print(f"  Refined predictions saved to {pred_path}")

    # Metrics summary CSV
    metrics_df = pd.DataFrame(results).T.reset_index(drop=True)
    metrics_path = os.path.join(results_dir, "threshold_comparison.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics summary saved to {metrics_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_comparison(raw_scores, norm_scores, labels, percentiles, results_dir)
    _plot_score_distributions(raw_scores, norm_scores, labels, results_dir)

    return results


# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

def _plot_comparison(raw, normed, labels, percentiles, results_dir):
    strategies = []
    f1s, precs, recs = [], [], []

    # Raw oracle
    m = oracle_metrics(raw, labels)
    strategies.append("raw\noracle")
    f1s.append(m["f1"]); precs.append(m["precision"]); recs.append(m["recall"])

    # Raw percentiles
    for p in percentiles:
        tau = percentile_threshold(raw, labels, p)
        m   = metrics_at_threshold(raw, labels, tau)
        strategies.append(f"raw\np{int(p)}")
        f1s.append(m["f1"]); precs.append(m["precision"]); recs.append(m["recall"])

    # Norm oracle
    m = oracle_metrics(normed, labels)
    strategies.append("norm\noracle")
    f1s.append(m["f1"]); precs.append(m["precision"]); recs.append(m["recall"])

    # Norm percentiles
    for p in percentiles:
        tau = percentile_threshold(normed, labels, p)
        m   = metrics_at_threshold(normed, labels, tau)
        strategies.append(f"norm\np{int(p)}")
        f1s.append(m["f1"]); precs.append(m["precision"]); recs.append(m["recall"])

    x = np.arange(len(strategies))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(strategies) * 1.2), 4))
    ax.bar(x - w, f1s,   w, label="F1",        color="#1D9E75", zorder=3)
    ax.bar(x,     precs, w, label="Precision",  color="#378ADD", zorder=3)
    ax.bar(x + w, recs,  w, label="Recall",     color="#D85A30", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Threshold strategy comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    # Divider between raw and normed
    n_raw = 1 + len(percentiles)
    ax.axvline(n_raw - 0.5, color="gray", linestyle=":", lw=1)
    ax.text(n_raw / 2 - 0.5, 0.96, "raw scores",
            ha="center", fontsize=8, color="gray")
    ax.text(n_raw + len(percentiles) / 2, 0.96, "per-rec normalised",
            ha="center", fontsize=8, color="gray")
    plt.tight_layout()
    path = os.path.join(results_dir, "threshold_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Comparison chart saved to {path}")


def _plot_score_distributions(raw, normed, labels, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, scores, title in zip(
        axes,
        [raw, normed],
        ["Raw scores (mean+2std)", "Per-recording normalised scores"],
    ):
        normal_s = scores[labels == 0]
        anom_s   = scores[labels == 1]
        bins = np.linspace(
            min(scores.min(), scores.min()),
            max(scores.max(), scores.max()),
            60,
        )
        ax.hist(normal_s, bins=bins, alpha=0.55, color="#378ADD",
                label=f"Normal (n={len(normal_s)})", density=True)
        ax.hist(anom_s,   bins=bins, alpha=0.65, color="#D85A30",
                label=f"Anomalous (n={len(anom_s)})", density=True)
        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        # Annotate separation
        sep = abs(anom_s.mean() - normal_s.mean())
        pooled_std = (normal_s.std() + anom_s.std()) / 2
        d_prime = sep / (pooled_std + 1e-8)
        ax.set_title(f"{title}\nd' = {d_prime:.2f}")

    plt.suptitle("Score distribution: normal vs anomalous", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(results_dir, "score_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Score distributions saved to {path}")


# ---------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-recording normalisation and percentile threshold analysis"
    )
    parser.add_argument("--scores", type=str,
                        default="results/clip_scores.csv",
                        help="Path to clip_scores.csv from evaluate.py")
    parser.add_argument("--percentiles", type=float, nargs="+",
                        default=[90.0, 95.0, 99.0],
                        help="Percentiles for threshold (default: 90 95 99)")
    parser.add_argument("--results_dir", type=str,
                        default="results",
                        help="Output directory for plots and CSVs")
    args = parser.parse_args()

    analyse(args.scores, args.percentiles, args.results_dir)