import csv
import re
import numpy as np
import os
import librosa
from scipy.spatial.distance import cosine
import time
import tensorflow as tf
import tensorflow_hub as hub

# --- CONFIGURATION YAMNET ---
print("Chargement de YAMNet...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')


def get_audio_embeddings(audio_path):
    """Extrait tous les embeddings par fenêtre (frames) pour un fichier."""
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    # YAMNet génère un embedding toutes les 0.48s (fenêtres de 0.96s)
    _, embeddings, _ = yamnet_model(audio)
    return embeddings.numpy()


def assess_neutral(folder_path):
    """Crée la signature de référence en moyennant les frames de TOUS les fichiers neutres."""
    all_frames = []
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.flac'))]

    if not audio_files:
        return None

    print(f"Extraction des caractéristiques sur {len(audio_files)} fichiers neutres...")
    for filename in audio_files:
        try:
            frames = get_audio_embeddings(os.path.join(folder_path, filename))
            all_frames.append(frames)
        except Exception as e:
            print(f"Erreur sur {filename}: {e}")

    # On regroupe toutes les fenêtres de tous les fichiers pour créer UN neutre universel
    matrix_normal = np.vstack(all_frames)
    reference_signature = np.mean(matrix_normal, axis=0)
    return reference_signature


def compile_reference(folder_neutre):
    ref_file = "reference_yamnet_v3.npy"
    if os.path.exists(ref_file):
        print(f"📂 Chargement de la signature existante...")
        return np.load(ref_file)
    else:
        ref_sig = assess_neutral(folder_neutre)
        if ref_sig is not None:
            np.save(ref_file, ref_sig)
            return ref_sig
    return None


if __name__ == "__main__":
    start_total = time.perf_counter()
    folder_neutre = "noise_sounds_5sec"
    folder_test = "Sound2test"
    output_csv = "rapport_anomalie_YAMNet_MAX.csv"

    ref_sig = compile_reference(folder_neutre)

    if ref_sig is not None:
        test_files = [f for f in os.listdir(folder_test) if f.lower().endswith(('.wav', '.mp3'))]
        all_results = []

        print(f"\n⌛ Analyse de {len(test_files)} fichiers (Méthode Max Distance)...")

        for filename in test_files:
            try:
                full_path = os.path.join(folder_test, filename)
                # 1. On récupère TOUTES les fenêtres du fichier de 5s
                frames = get_audio_embeddings(full_path)

                # 2. On calcule la distance de chaque fenêtre par rapport au neutre
                dists = [cosine(f, ref_sig) for f in frames]

                # 3. On prend la PIRE distance (le maximum)
                # Si un seul moment est anormal, la valeur sera haute.
                max_dist = max(dists)

                match = re.search(r'part(\d+)', filename)
                part_idx = int(match.group(1)) if match else 0

                all_results.append({
                    'Source Audio': filename,
                    'Part': part_idx,
                    'Distance_Max': float(max_dist)
                })
            except Exception as e:
                print(f"❌ Erreur sur {filename}: {e}")

        # --- CALIBRATION DYNAMIQUE ---
        # On repère les fichiers qui sortent vraiment du lot
        d_vals = [r['Distance_Max'] for r in all_results]
        threshold = np.mean(d_vals) + (2 * np.std(d_vals))  # Seuil à +2 écart-types

        for res in all_results:
            res['Status'] = "ANOMALIE" if res['Distance_Max'] > threshold else "RAS"
            t = res['Part'] * 5
            res['Timestamp'] = f"{t // 60}:{t % 60:02d}"

        # Sauvegarde
        results_sorted = sorted(all_results, key=lambda x: x['Distance_Max'], reverse=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Source Audio', 'Part', 'Timestamp', 'Distance_Max', 'Status'])
            writer.writeheader()
            writer.writerows(results_sorted)

        print(f"\n✅ Analyse terminée. Seuil utilisé : {threshold:.4f}")
        print(f"Anomalies trouvées : {sum(1 for r in all_results if r['Status'] == 'ANOMALIE')}")