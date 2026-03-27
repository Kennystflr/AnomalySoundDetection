import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, welch, butter, filtfilt

# ===================== PARAMETERS =====================
N_MELS = 126          # pas utilisé pour le PSD (à garder pour d'autres usages)
N_FFT = 96000        # 3 second windows ( like anatole )
HOP_LENGTH = 16384   # 50 % de recouvrement

AUDIO_DIR = "Song"
OUT_DIR = "mel_images"
os.makedirs(OUT_DIR, exist_ok=True)

# ===================== MEL CONVERSIONS =====================
def hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)

def mel_to_hz(m):
    return 700 * (10**(m / 2595) - 1)

def mel_filterbank(sr, n_fft, n_mels):
    f_min, f_max = 0, sr / 2
    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(1, n_mels + 1):
        fb[i - 1, bins[i - 1]:bins[i]] = (
            np.arange(bins[i - 1], bins[i]) - bins[i - 1]
        ) / (bins[i] - bins[i - 1])

        fb[i - 1, bins[i]:bins[i + 1]] = (
            bins[i + 1] - np.arange(bins[i], bins[i + 1])
        ) / (bins[i + 1] - bins[i])

    return fb

def mel_band_frequencies(sr, n_fft, n_mels):
    f_min, f_max = 0, sr / 2
    mels = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz = mel_to_hz(mels)
    return 0.5 * (hz[1:-1] + hz[2:])

# ===================== PROCESS FILES =====================
for file in os.listdir(AUDIO_DIR):
    if not file.lower().endswith(".wav"):
        continue

    path = os.path.join(AUDIO_DIR, file)
    print(f"Processing {file}...")

    # ---------- LOAD AUDIO ----------
    sr, data = wavfile.read(path)

    if data.dtype != np.float32:
        data = data.astype(np.float32)
        data /= np.max(np.abs(data))

    duration = len(data) / sr

    # ---------- STFT ----------
    f, t, Zxx = stft(
        data,
        fs=sr,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP_LENGTH
    )
    power_spec = np.abs(Zxx) ** 2

    # ---------- MEL SPECTROGRAM ----------
    mel_fb = mel_filterbank(sr, N_FFT, N_MELS)
    mel_spec = mel_fb @ power_spec
    mel_db = 10 * np.log10(mel_spec + 1e-10)
    mel_freqs = mel_band_frequencies(sr, N_FFT, N_MELS)

    # ---------- PSD ----------
    freqs_psd, psd = welch(data, fs=sr, nperseg=N_FFT)
    psd_db = 10 * np.log10(psd + 1e-12)

    # ===================== PLOT =====================
    fig, axs = plt.subplots(
        1, 2,
        figsize=(14, 4),
        gridspec_kw={"width_ratios": [3, 1]}
    )

    # Mel Spectrogram
    im = axs[0].imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        extent=[t.min(), t.max(), mel_freqs[0], mel_freqs[-1]]
    )
    axs[0].set_title(f"Mel Spectrogram – {file} ({duration:.2f}s)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_ylim(20, 5000)
    plt.colorbar(im, ax=axs[0], label="Power (dB)")

    # PSD
    axs[1].plot(psd_db, freqs_psd)
    axs[1].set_title("PSD (Welch)")
    axs[1].set_xlabel("Power (dB)")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_ylim(0, 200)
    axs[1].grid(True)

    plt.tight_layout()

    # ---------- SAVE ----------
    out_path = os.path.join(OUT_DIR, file.replace(".wav", ".png"))
    plt.savefig(out_path, dpi=200)
    plt.close()

print("✅ All files processed (HP filtered <20Hz) and saved in mel_images/")