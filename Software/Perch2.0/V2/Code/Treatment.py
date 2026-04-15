import os
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
import soundfile as sf

# ─── CONFIG (must match annotation_tracker.py) ────────────────
CSV_FILE      = "rapport_Anomalies_ml17_280a.csv"
AUDIO_FOLDER  = "ml17_280a_5sec"
MEL_FOLDER    = "ml17_280a_5sec_mel"   # output folder for .npy files
CLIP_DURATION = 5.0
N_MELS        = 128
FMAX          = 8000
# ──────────────────────────────────────────────────────────────

def extract_for_validation(CSV_FILE = "CSV/new_threshold.csv", SOURCE_FOLDER = "ml17_280a_5sec", DEST_FOLDER = "SAMPLES_EXTRACTED_2", QUANTITY = 50):
    """
    Randomly samples a set of anomalies and normal sounds for human validation.
    It copies the selected audio files to a destination folder and generates a
    control CSV with an empty 'Human_Validation' column for the user to fill.
    """
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found.")
        return

    # 1. Load data
    df = pd.read_csv(CSV_FILE, sep=";")

    # 2. Filtering (Anomalies vs Normal)
    anomalies_df = df[df['Status'].str.contains('⚠️ ANOMALIE', na=False)]
    voids_df = df[df['Status'].str.contains('✅ NORMAL', na=False)]

    print(f"📊 Available: {len(anomalies_df)} anomalies | {len(voids_df)} normal files.")

    # 3. Sampling
    anom_sample = anomalies_df.sample(n=min(QUANTITY, len(anomalies_df)))
    void_sample = voids_df.sample(n=min(QUANTITY, len(voids_df)))

    # Combine both samples
    selected_df = pd.concat([anom_sample, void_sample]).copy()

    # 4. Prepare folder and new CSV
    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)

    # Add an empty column for future human input
    selected_df["Validation_Humaine"] = ""

    print(f"📁 Copying original files to: {DEST_FOLDER}...")

    success_count = 0
    for _, row in selected_df.iterrows():
        filename = row['Source Audio']
        src_path = os.path.join(SOURCE_FOLDER, filename)
        dst_path = os.path.join(DEST_FOLDER, filename)  # Strict original filename

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            success_count += 1
        else:
            print(f"❓ Missing: {filename}")

    # 5. Save the validation CSV inside the folder
    output_csv_path = os.path.join(DEST_FOLDER, "list_to_validate.csv")
    selected_df.to_csv(output_csv_path, index=False, sep=";")

    print("-" * 40)
    print(f"✅ Finished!")
    print(f"🔊 Audio files copied: {success_count}")
    print(f"📄 Validation CSV created: {output_csv_path}")
    print("📝 User task: fill the 'Validation_Humaine' column.")

def compute_and_save(audio_path: str, out_path: str) -> bool:
    """
    Computes the Mel-spectrogram of a specific audio file, converts it to dB scale,
    and saves the result (including sample rate and fmax) as a binary .npy file.
    Returns True on success.
    """
    try:
        y, sr = librosa.load(audio_path, duration=CLIP_DURATION, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
        S_dB = librosa.power_to_db(S, ref=np.max).astype(np.float32)

        data = {"S_dB": S_dB, "sr": sr, "fmax": FMAX}
        np.save(out_path, data, allow_pickle=True)
        return True
    except Exception as exc:
        print(f"  [ERROR] {audio_path}: {exc}")
        return False


def main():
    """
    Main batch processing function for spectrogram generation.
    Scans the audio folder or reads the CSV, checks for already processed files
    to avoid duplicates, and triggers compute_and_save for each file.
    """
    os.makedirs(MEL_FOLDER, exist_ok=True)

    # Read file list from CSV
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] CSV not found: {CSV_FILE}")
        print("Falling back to scanning audio folder directly...")
        audio_files = [f for f in os.listdir(AUDIO_FOLDER)
                       if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]
    else:
        df = pd.read_csv(CSV_FILE)
        audio_files = df["Source Audio"].dropna().unique().tolist()

    total = len(audio_files)
    success = 0
    skipped = 0

    print(f"\nProcessing {total} files → saving to '{MEL_FOLDER}/'")
    print("─" * 60)

    for i, filename in enumerate(audio_files, 1):
        audio_path = os.path.join(AUDIO_FOLDER, filename)
        stem = Path(filename).stem
        out_path = os.path.join(MEL_FOLDER, stem + ".npy")

        # Skip if already computed
        if os.path.exists(out_path):
            print(f"  [{i:>4}/{total}]  SKIP (already exists)  {filename}")
            skipped += 1
            continue

        if not os.path.exists(audio_path):
            print(f"  [{i:>4}/{total}]  MISSING audio          {filename}")
            continue

        ok = compute_and_save(audio_path, out_path)
        if ok:
            success += 1
            print(f"  [{i:>4}/{total}]  OK                     {filename}")

    print("─" * 60)
    print(f"\nDone.  {success} computed,  {skipped} skipped,  "
          f"{total - success - skipped} missing/failed\n")
    print(f"Set MEL_FOLDER = \"{MEL_FOLDER}\" in annotation_tracker.py to use these files.")


def amplify_directory(input_dir, output_dir, gain=2.0):
    """
    Iterates through a directory and multiplies the amplitude of all .wav files by a 'gain' factor.
    Applies clipping to prevent digital distortion (overflow) before saving.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            sample_rate, data = wavfile.read(input_path)
            data_float = data.astype(np.float32)
            amplified = data_float * gain
            # Get data type limits to avoid wrapping
            max_val = np.iinfo(data.dtype).max
            min_val = np.iinfo(data.dtype).min
            amplified = np.clip(amplified, min_val, max_val)
            amplified = amplified.astype(data.dtype)
            wavfile.write(output_path, sample_rate, amplified)
            print(f"Processed: {filename}")
    print("Done.")


def frac_audio(in_folder, out_folder):
    """
    Splits all audio files in a directory into 5-second segments.
    Maintains the sub-directory structure in the output folder and
    saves each segment as a .wav file.
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print(f"Folder created: {out_folder}")

    extensions = ('.wav', '.mp3', '.flac', '.ogg')

    for root, dirs, files in os.walk(in_folder):
        for filename in files:
            if filename.lower().endswith(extensions):
                file_path = os.path.join(root, filename)

                # Load audio at a fixed sample rate (32kHz)
                audio, sr = librosa.load(file_path, sr=32000)

                buffer_5s = 5 * sr
                total_samples = len(audio)
                num_segments = total_samples // buffer_5s

                if num_segments == 0:
                    print(f"Skipping {filename}: too short (< 5s)")
                    continue

                for i in range(num_segments):
                    start = i * buffer_5s
                    end = start + buffer_5s
                    segment = audio[start:end]

                    name_without_ext = os.path.splitext(filename)[0]

                    # Replicate directory structure
                    relative_path = os.path.relpath(root, in_folder)
                    out_dir = os.path.join(out_folder, relative_path)

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    out_filename = f"{name_without_ext}_part{i}.wav"
                    out_path = os.path.join(out_dir, out_filename)

                    sf.write(out_path, segment, sr)

                print(f"✅ {filename} split into {num_segments} segments.")