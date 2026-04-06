import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os

# --- CONFIGURATION ---
CSV_FILE = "rapport_anomalies_optimize.csv"
VECTOR_COL = "Embedding"  # Vérifie bien le nom de la colonne qui contient le vecteur [0.1, ...]
LABEL_COL = "Validation_Humaine"
OUTPUT_DIR = "images"


def visualize_embeddings(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ Fichier {csv_path} introuvable.")
        return

    # 1. Chargement des données
    df = pd.read_csv(csv_path)

    # On ne garde que les données annotées (RAS ou ANOMALIE)
    df = df[df[LABEL_COL].isin(['RAS', 'ANOMALIE'])].copy()

    if len(df) < 5:
        print("❌ Pas assez de données annotées pour visualiser.")
        return

    print(f"🔄 Traitement de {len(df)} échantillons...")

    # 2. Conversion de la colonne texte en matrice NumPy
    # On transforme la chaîne "[0.1, 0.2...]" en liste réelle
    try:
        X = np.array([ast.literal_eval(vec) if isinstance(vec, str) else vec for vec in df[VECTOR_COL]])
    except Exception as e:
        print(f"❌ Erreur lors de la lecture des vecteurs : {e}")
        print("Vérifiez que la colonne 'Embedding' contient bien les vecteurs Perch.")
        return

    # Normalisation (très important pour le PCA/t-SNE)
    X_scaled = StandardScaler().fit_transform(X)

    # 3. Réduction de dimension avec PCA (2D)
    print("📉 Calcul du PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 4. Réduction de dimension avec t-SNE (2D)
    print("🧬 Calcul du t-SNE (cela peut prendre quelques secondes)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # 5. Création du Graphique Comparatif
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = {'RAS': '#2ecc71', 'ANOMALIE': '#e74c3c'}
    c_list = [colors[label] for label in df[LABEL_COL]]

    # Plot PCA
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=c_list, alpha=0.6, edgecolors='w')
    ax1.set_title(f"PCA (Linear Projection)\nSeparation of {len(df)} samples")
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.grid(True, alpha=0.2)

    # Plot t-SNE
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=c_list, alpha=0.7, edgecolors='w')
    ax2.set_title(f"t-SNE (Non-linear Clusters)\nPerplexity=30")
    ax2.set_xlabel("t-SNE axis 1")
    ax2.set_ylabel("t-SNE axis 2")
    ax2.grid(True, alpha=0.2)

    # Légende personnalisée
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='RAS', markerfacecolor='#2ecc71', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='ANOMALIE', markerfacecolor='#e74c3c',
                              markersize=10)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2)

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    plt.savefig(f"{OUTPUT_DIR}/visualisation_embeddings_2D.png")
    print(f"✅ Graphique enregistré dans {OUTPUT_DIR}/visualisation_embeddings_2D.png")
    plt.show()


if __name__ == "__main__":
    visualize_embeddings(CSV_FILE)