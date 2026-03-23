import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    auc
)


def find_optimal_threshold(csv_file, output_folder="images"):
    # 1. Loading and cleaning
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Created folder: {output_folder}")

    df = pd.read_csv(csv_file)
    mask = df['Validation_Humaine'].isin(['ANOMALIE', 'RAS'])
    df_clean = df[mask].copy()

    if df_clean.empty:
        print("❌ Not enough validated data to run evaluation.")
        return

    # y_true : Ground Truth (1 for Anomaly, 0 for RAS/Noise)
    y_true = df_clean['Validation_Humaine'].map({'ANOMALIE': 1, 'RAS': 0}).values
    # Distances as raw scores
    distances = df_clean['Distance'].values

    # 2. Calculate curves
    precisions, recalls, thresholds = precision_recall_curve(y_true, distances)
    # Calculate F1-Score for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find optimal index
    ix = np.argmax(f1_scores)
    best_threshold = thresholds[ix]

    # Final Metrics at optimal point
    y_pred_opti = (distances >= best_threshold).astype(int)
    f1_final = f1_score(y_true, y_pred_opti)
    pr_auc = auc(recalls, precisions)

    print("\n" + "=" * 40)
    print(f"🚀 PERFORMANCE SUMMARY - OPTIMAL THRESHOLD ({best_threshold:.4f})")
    print("=" * 40)
    print(classification_report(y_true, y_pred_opti, target_names=['Noise (RAS)', 'Anomaly']))
    print("-" * 30)
    print(f"✅ Precision : {precision_score(y_true, y_pred_opti):.2%}")
    print(f"✅ Recall    : {recall_score(y_true, y_pred_opti):.2%}")
    print(f"✅ F1-Score  : {f1_final:.2%}")
    print(f"✅ PR-AUC    : {pr_auc:.4f}")
    print("-" * 30)

    # --- PLOT 1: Precision & Recall vs Threshold ---
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
    plt.axvline(best_threshold, color='red', linestyle=':', label=f'Optimal Threshold ({best_threshold:.3f})')
    plt.title(f'Threshold Optimization (F1-Max: {f1_final:.2%})')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_folder}/threshold_optimization.png")
    plt.show()

    # --- PLOT 2: Precision-Recall Curve (Standard PR Curve) ---
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='purple', lw=2, label=f'PR-AUC = {pr_auc:.4f}')
    plt.fill_between(recalls, precisions, alpha=0.2, color='purple')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_folder}/precision_recall_curve.png")
    plt.show()

    # --- PLOT 3: F1-Score Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores[:-1], color='orange', lw=2, label='F1-Score')
    plt.axhline(f1_final, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('F1-Score Curve across Thresholds')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_folder}/f1_score_curve.png")
    plt.show()

    print(f"✅ All plots saved in the '{output_folder}' directory.")
    return best_threshold


if __name__ == "__main__":
    CSV_FILE = "rapport_anomalies_optimize.csv"
    best_seuil = find_optimal_threshold(CSV_FILE)