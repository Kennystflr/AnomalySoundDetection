import csv
import re
import shutil

import onnxruntime as ort
import numpy as np
import os
import librosa
import pandas as pd
from scipy.spatial.distance import cosine
import time
from Treatment import frac_audio

session = ort.InferenceSession("perch_v2.onnx")

def get_audio_signature(audio_path):
    audio, _ = librosa.load(audio_path, sr=32000, duration=5.0)
    input_tensor = np.zeros((1, 160000), dtype=np.float32)
    input_tensor[0, :len(audio)] = audio
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    embedding = outputs[1]
    return embedding



def assess_neutral(folder_path):
    signatures = []
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    if not audio_files:
        print("❌ Aucun fichier trouvé dans le dossier neutre.")
        return None, None
    print(f"Analyse de {len(audio_files)} segments neutres...")
    for filename in audio_files:
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        try:
            sig = get_audio_signature(file_path)
            signatures.append(sig.flatten())
        except Exception as e:
            print(f"Erreur sur {filename}: {e}")
    matrix_normal = np.vstack(signatures)
    reference_signature = np.mean(matrix_normal, axis=0)
    std_dev = np.std(matrix_normal, axis=0)
    print(f"the standart deviation on each line is {std_dev}")
    overall_stability = np.mean(std_dev)
    print("-" * 30)
    print("✅ Signature de référence générée avec succès !")
    print(f"Stabilité du milieu (plus c'est bas, mieux c'est) : {overall_stability:.4f}")
    print("-" * 30)

    return reference_signature, std_dev

def check_anomaly(new_sample_path, reference, threshold=0.25):
    new_sig = get_audio_signature(new_sample_path).flatten()
    dist = cosine(new_sig, reference)
    file_name = os.path.basename(new_sample_path)
    if dist > threshold:
        print(f"⚠️ ANOMALIE : {file_name:30} | Distance : {dist:.4f}")
        return True
    else:
        print(f"✅ RAS      : {file_name:30} | Distance : {dist:.4f}")
        return False


def get_weighted_distance(current_sig, ref_sig, std_dev):
    """
    Calcule une distance cosinus pondérée par l'inverse de la variance.
    Plus une dimension est stable (std bas), plus elle compte.
    """
    # 1. Calculer les poids (inverse de la variance)
    # On ajoute une petite constante (1e-6) pour éviter la division par zéro
    weights = 1.0 / (std_dev ** 2 + 1e-6)

    # 2. Appliquer les poids aux signatures
    # On multiplie par la racine carrée des poids pour que le produit scalaire
    # dans le calcul du cosinus corresponde à une pondération par la variance
    w_current = current_sig * np.sqrt(weights)
    w_ref = ref_sig * np.sqrt(weights)

    # 3. Calculer la distance cosinus sur les vecteurs pondérés
    dist = cosine(w_current, w_ref)

    return dist

def compile_reference(folder_neutre):
    ref_file = "second_signature.npy"
    std_file = ".npy"
    if os.path.exists(ref_file):
        print("📂 Chargement de la signature de référence existante...")
        ref_sig = np.load(ref_file)
        return ref_sig
    else:
        print("⚙️ Calcul de la signature de référence...")
        ref_sig, std_dev = assess_neutral(folder_neutre)
        if ref_sig is not None:
            np.save(ref_file, ref_sig)
            np.save(std_file, std_dev)
            print("💾 Signature sauvegardée.")
            print( ref_sig,std_dev)


def compile_anomaly_reference(csv_file, folder_test):
    """
    Crée une signature moyenne basée sur les fichiers validés manuellement comme ANOMALIE.
    """
    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)
    if 'Validation_Humaine' not in df.columns:
        return None

    # On filtre pour ne garder que tes validations réelles
    anomaly_files = df[df['Validation_Humaine'] == 'ANOMALIE']['Source Audio'].tolist()

    signatures = []
    print(f"🧬 Compilation de la signature de référence des ANOMALIES ({len(anomaly_files)} fichiers)...")

    for filename in anomaly_files:
        path = os.path.join(folder_test, filename)  # On cherche dans Sound2test
        if os.path.exists(path):
            try:
                sig = get_audio_signature(path).flatten()
                signatures.append(sig)
            except Exception:
                continue

    if not signatures:
        return None

    return np.mean(signatures, axis=0)

def anomaly_detector_gaussian():
    start_total = time.perf_counter()  # ⏱ début chrono
    folder_neutre = "noise_sounds_5sec"
    folder_test = "SAMPLES_EXTRACTED"
    output_csv = "SAMPLES_EXTRACTED/rapport_anomalies_cosinus_2.csv"

    # --- SEUILS ---
    threshold = 0.316  # Seuil pour une ANOMALIE
    auto_label_limit = 0.0001# Seuil pour enrichir le dossier neutre

    # Assurer que le dossier neutre existe
    if not os.path.exists(folder_neutre):
        os.makedirs(folder_neutre)

    print("🔍 Compilation de la référence...")
    ref_sig = compile_reference(folder_neutre)

    if ref_sig is not None:
        test_files = [f for f in os.listdir(folder_test) if f.lower().endswith(('.wav', '.mp3'))]
        all_results = []
        segment_duration = 5  # secondes

        print(f"⌛ Analyse de {len(test_files)} fichiers en cours...")

        for filename in test_files:
            full_path = os.path.join(folder_test, filename)
            try:
                # Calcul de la position temporelle
                match = re.search(r'part(\d+)', filename)
                part_index = int(match.group(1)) if match else 0
                real_start_sec = part_index * segment_duration

                # Calcul de la signature et de la distance
                current_sig = get_audio_signature(full_path).flatten()
                dist = cosine(current_sig, ref_sig)
                dist_val = round(float(dist), 4)

                status = "RAS"

                # --- LOGIQUE DE DÉPLACEMENT (DISTANCE < 0.1) ---
                if dist_val < auto_label_limit:
                    dest_path = os.path.join(folder_neutre, filename)
                    # On vérifie si le fichier n'existe pas déjà pour éviter les erreurs
                    if not os.path.exists(dest_path):
                        shutil.move(full_path, dest_path)
                        status = "ENRICHI (Neutre)"
                        print(
                            f"✅ {filename} est très proche de la référence ({dist_val}). Déplacé vers {folder_neutre}")
                    else:
                        os.remove(full_path)  # Doublon, on le supprime simplement
                        status = "DOUBLON (Supprimé)"

                elif dist_val > threshold:
                    status = "ANOMALIE"

                # Formatage du temps
                minutes, seconds = divmod(real_start_sec, 60)
                timestamp_min = f"{int(minutes)}:{int(seconds):02d}"

                all_results.append({
                    'Source Audio': filename,
                    'Part': part_index,
                    'Début (sec)': real_start_sec,
                    'Début (min:sec)': timestamp_min,
                    'Distance': dist_val,
                    'Status': status
                })

            except Exception as e:
                print(f"❌ Erreur sur le fichier {filename}: {e}")

        # Tri par distance (plus grosses anomalies en premier)
        results_sorted = sorted(all_results, key=lambda x: x['Distance'], reverse=True)

        # Écriture du CSV
        keys = ['Source Audio', 'Part', 'Début (sec)', 'Début (min:sec)', 'Distance', 'Status']
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results_sorted)

            print(f"\n--- ANALYSE TERMINÉE ---")
            print(f"📂 Fichier CSV créé : {output_csv}")
            print(f"📊 Segments traités : {len(results_sorted)}")

            print("\nTop 5 des anomalies détectées :")
            for res in results_sorted[:5]:
                if res['Status'] == "ANOMALIE":
                    print(f"⚠️ {res['Source Audio']} (Part {res['Part']}) - Distance: {res['Distance']}")

        except Exception as e:
            print(f"❌ Erreur lors de l'écriture du CSV : {e}")

        end_total = time.perf_counter()
        print("\n" + "=" * 40)
        print(f"⏱ Temps total de traitement : {end_total - start_total:.2f} secondes")
        print("=" * 40)

if __name__ == "__main__":
    #compile_reference("noise_sounds_5sec")
    #assess_neutral("noise_sounds_5sec")
    anomaly_detector_gaussian()
    #compile_reference("noise_sounds_5sec")
    #print("salut")
    #get_audio_signature("noise_sounds_5sec/ml19_292a_0015_part0.wav")

