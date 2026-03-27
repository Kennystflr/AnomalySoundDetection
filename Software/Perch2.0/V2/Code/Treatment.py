import os

import librosa
import numpy as np
from scipy.io import wavfile
import soundfile as sf

def amplify_directory(input_dir, output_dir, gain=2.0):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            sample_rate, data = wavfile.read(input_path)
            data_float = data.astype(np.float32)
            amplified = data_float * gain
            max_val = np.iinfo(data.dtype).max
            min_val = np.iinfo(data.dtype).min
            amplified = np.clip(amplified, min_val, max_val)
            amplified = amplified.astype(data.dtype)
            wavfile.write(output_path, sample_rate, amplified)
            print(f"Processed: {filename}")
    print("Done.")


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

#if __name__ == "__main__":
