import csv
import re
import onnxruntime as ort
import numpy as np
import os
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine
import time

session = ort.InferenceSession("perch_v2.onnx")

def frac_audio(in_folder, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print(f"Dossier créé : {out_folder}")
    extensions = ('.wav', '.mp3', '.flac', '.ogg')
    for filename in os.listdir(in_folder):
        if filename.lower().endswith(extensions):
            file_path = os.path.join(in_folder, filename)

            # load at 32kHz
            audio, sr = librosa.load(file_path, sr=32000)
            buffer_5s = 5 * sr
            total_samples = len(audio)
            num_segments = total_samples // buffer_5s
            if num_segments == 0:
                print(f"Skipping {filename} : trop court (< 5s)")
                continue
            for i in range(num_segments):
                start = i * buffer_5s
                end = start + buffer_5s
                segment = audio[start:end]
                name_without_ext = os.path.splitext(filename)[0]
                out_filename = f"{name_without_ext}_part{i}.wav"
                out_path = os.path.join(out_folder, out_filename)
                sf.write(out_path, segment, sr)
            print(f"✅ {filename} découpé en {num_segments} segments.")



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
        try:
            sig = get_audio_signature(file_path)
            signatures.append(sig.flatten())
        except Exception as e:
            print(f"Erreur sur {filename}: {e}")
    matrix_normal = np.vstack(signatures)
    reference_signature = np.mean(matrix_normal, axis=0)
    std_dev = np.std(matrix_normal, axis=0)
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

def compile_reference():
    ref_file = "reference_signature.npy"
    std_file = "reference_std.npy"
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
            return ref_sig


if __name__ == "__main__":
    start_total = time.perf_counter()  # ⏱ début chrono

    folder_neutre = "neutre_sounds_5sec"
    folder_test = "Sound2test"

    threshold = 0.25

    ref_sig = compile_reference()

    if ref_sig is not None:
        test_files = [f for f in os.listdir(folder_test) if f.lower().endswith(('.wav', '.mp3'))]
        all_results = []
        output_csv = "rapport_anomalies.csv"
        segment_duration = 5  # secondes

        print(f"\n⌛ Analyse segmentée de {len(test_files)} fichiers...")

        for filename in test_files:
            full_path = os.path.join(folder_test, filename)
            try:
                match = re.search(r'part(\d+)', filename)
                if match:
                    part_index = int(match.group(1))
                else:
                    part_index = 0
                real_start_sec = part_index * segment_duration
                current_sig = get_audio_signature(full_path ).flatten()
                dist = cosine(current_sig, ref_sig)
                minutes = int(real_start_sec // 60)
                seconds = int(real_start_sec % 60)
                timestamp_min = f"{minutes}:{seconds:02d}"
                all_results.append({
                    'Source Audio': filename,
                    'Part': part_index,
                    'Début (sec)': real_start_sec,
                    'Début (min:sec)': timestamp_min,
                    'Distance': round(float(dist), 4),
                    'Status': "ANOMALIE" if dist > threshold else "RAS"
                })

            except Exception as e:
                print(f"❌ Erreur sur le fichier {filename}: {e}")
        results_sorted = sorted(all_results, key=lambda x: x['Distance'], reverse=True)
        keys = ['Source Audio', 'Part', 'Début (sec)', 'Début (min:sec)', 'Distance', 'Status']
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results_sorted)
            print(f"\n--- ANALYSE TERMINÉE ---")
            print(f"📂 Fichier CSV créé : {output_csv}")
            print(f"📊 Nombre de segments analysés : {len(results_sorted)}")

            print("\nTop 5 des anomalies détectées :")
            for res in results_sorted[:5]:
                if res['Distance'] > threshold:
                    print(f"⚠️ {res['Source Audio']} (Part {res['Part']}) - Distance: {res['Distance']}")
        except Exception as e:
            print(f"❌ Erreur lors de l'écriture du CSV : {e}")
        end_total = time.perf_counter()
        print("\n" + "=" * 40)
        print(f"⏱ Temps total de traitement : {end_total - start_total:.2f} secondes")
        print("=" * 40)