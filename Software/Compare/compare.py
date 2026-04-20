import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def csv_to_anomaly_vector(csv_path):
    """
    Lit le CSV et génère un vecteur de 0 et 1.
    L'indice du vecteur correspond au numéro du segment ('Part').
    """
    df = pd.read_csv(csv_path)

    # On trouve le nombre total de segments pour définir la taille du vecteur
    # On prend le max de la colonne 'Part' + 1
    num_segments = int(df['Part'].max()) + 1

    # Initialisation du vecteur avec des zéros
    vector = np.zeros(num_segments, dtype=int)

    # On remplit avec 1 là où le statut est 'ANOMALIE'
    # On filtre pour être sûr de ne prendre que les lignes marquées ANOMALIE
    anomalies = df[df['Status'] == 'ANOMALIE']
    for _, row in anomalies.iterrows():
        index = int(row['Part'])
        vector[index] = 1

    return vector


def plot_anomaly_spectrogram(audio_path, anomaly_vector, sr=16000, duration_per_step=5):
    """
    Affiche le Mel-spectrogramme synchronisé avec le vecteur d'anomalies.
    """
    y, sr = librosa.load(audio_path, sr=sr)

    # Calcul du spectrogramme
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- Spectrogramme ---
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax1)
    ax1.set_title(f"Analyse Acoustique : {audio_path.split('/')[-1]}")

    # --- Vecteur d'Anomalies ---
    # Création de l'axe temporel pour le plot en escalier
    time_axis = np.arange(len(anomaly_vector) + 1) * duration_per_step
    # Doubler la dernière valeur pour le tracé 'post'
    display_vec = np.append(anomaly_vector, anomaly_vector[-1])

    ax2.step(time_axis, display_vec, where='post', color='red', lw=2)
    ax2.fill_between(time_axis, display_vec, step="post", color='red', alpha=0.3)

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'ANOMALIE'])
    ax2.set_xlabel("Temps (secondes)")
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# --- PIPELINE COMPLET ---
if __name__ == "__main__":
    CSV_FILE = "perch2.0_compare.csv"
    AUDIO_FILE = "ml17_280a_0083.wav"

    # 1. Extraction du vecteur depuis le CSV
    vecteur = csv_to_anomaly_vector(CSV_FILE)
    print(f"Vecteur généré ({len(vecteur)} segments) : {vecteur}")

    # 2. Affichage
    plot_anomaly_spectrogram(AUDIO_FILE, vecteur)