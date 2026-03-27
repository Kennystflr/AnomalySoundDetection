import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

# ===================== PARAMETERS =====================
INPUT_DIR = "Song"
OUTPUT_DIR = "Song_amplified"
GAIN = 2.0        # facteur d'amplification (2.0 ≈ +6 dB)
HP_CUTOFF = 20    # passe-haut >20 Hz

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== HELPER FUNCTIONS =====================
def highpass_filter(data, sr, cutoff=20, order=4):
    """Applique un filtre passe-haut Butterworth sur les données audio."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# ===================== PROCESS FILES =====================
for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith(".wav"):
        continue

    input_path = os.path.join(INPUT_DIR, file)
    output_path = os.path.join(OUTPUT_DIR, file)

    print(f"Processing {file}...")

    # ---------- LOAD AUDIO ----------
    sr, data = wavfile.read(input_path)

    # Convert to float32 [-1,1]
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data /= max_val

    # ---------- HIGH-PASS FILTER ----------
    filtered = highpass_filter(data, sr, cutoff=HP_CUTOFF)

    # ---------- AMPLIFY ----------
    amplified = filtered * GAIN

    # Prevent clipping
    amplified = np.clip(amplified, -1.0, 1.0)

    # Convert back to int16
    amplified_int16 = (amplified * 32767).astype(np.int16)

    # ---------- SAVE ----------
    wavfile.write(output_path, sr, amplified_int16)

print("✅ All files filtered and amplified saved in Song_amplified/")