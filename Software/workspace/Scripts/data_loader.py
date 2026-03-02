import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import librosa

def extract_mel(
    wav_path,
    sr=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
):
    y, sr = librosa.load(wav_path, sr=sr)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # shape: (n_mels, n_frames)

class SoundDataset(Dataset):
    def __init__(self, csv_file):
        self.meta = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        x = np.load(row["path"])         # (F, T)
        y = int(row["label"])

        x = torch.tensor(x, dtype=torch.float32)
        return x, y
    
class TaskDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        window_size,
        hop_size,
        task,              # "ae" or "ar"
        format="conv"      # "conv" or "transformer"
    ):
        self.base = base_dataset
        self.window = window_size
        self.hop = hop_size
        self.task = task
        self.format = format

        self.index = []
        for i in range(len(self.base)):
            x, _ = self.base[i]
            T = x.shape[1]
            extra = 1 if task == "ar" else 0
            for start in range(0, T - window_size - extra, hop_size):
                self.index.append((i, start))
    
    def _format(self, x):
        if self.format == "transformer":
            return x.transpose(0, 1)  
        return x                     
    
    def __getitem__(self, idx):
        file_idx, start = self.index[idx]
        x, label = self.base[file_idx]

        if self.task == "ae":
            x_win = x[:, start:start+self.window]
            x_win = self._format(x_win)
            return x_win, x_win, label

        if self.task == "ar":
            x_in = x[:, start:start+self.window]
            x_tg = x[:, start+1:start+self.window+1]

            x_in = self._format(x_in)
            x_tg = self._format(x_tg)

            return x_in, x_tg, label

class NormalOnly(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = [
            i for i in range(len(dataset))
            if dataset[i][2] == 0
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

