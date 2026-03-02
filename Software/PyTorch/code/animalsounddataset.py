import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

class AnimalSoundDataset(Dataset):
    def __init__(self, 
                 annotations_file, 
                 audio_dir, 
                 transformation, 
                 target_sample_rate, 
                 num_samples,
                 device):
        self.annotations = self._load_annotations(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        # Define yo label mapping here
        self.label_map = {
            'HW': 1,
            'ABW': 2,
            'Spermwhale': 3,
            'SW': 4,
            'FW' : 5,
            'SRW' : 6,
            'Delphinid clicks' : 7,
            # Add all known string labels here
        }
        self.default_label = 0  # Use 0 for unknown labels
        
    
    def _load_annotations(self, annotations_file):
        ext = os.path.splitext(annotations_file)[1].lower()  # get file extension
        if ext == ".csv":
            return pd.read_csv(annotations_file)
        elif ext in [".xls", ".xlsx"]:
            return pd.read_excel(annotations_file)
        else:
            raise ValueError(f"Unsupported annotations file format: {ext}")

    def __len__(self):
        return len(self.annotations)

    #
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path) #signal and sample rate
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal) #mix it down to mono
        signal = self._cut_if_necessary(signal) #too many samples
        signal = self._right_pad_if_necessary(signal) #zero-pad it if not enough samples
        signal = self.transformation(signal) #pass the signal into the mel spectrogram func
        #signal is a tensor object -> (num_channels, samples)

        #mapping strings to ints for labels
        if pd.isna(label):
            label = self.default_label
        elif isinstance(label, str):
            label = self.label_map.get(label, self.default_label)
        else:
            try:
                label = int(label)
            except (ValueError, TypeError):
                label = self.default_label
        label = torch.tensor(label, dtype=torch.long)

        return signal, label
    
    #take out the last few samples if too many
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples] #take everything up until the num_samples
        return signal
    
    #add zero-ed samples if not enough
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_needed = self.num_samples - length_signal
            last_dim_padding = (0, num_needed) #(num_prepend, num_append)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    #change the sample rate (how many samples per second) to make them all the same
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate: #if needed
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    #converting multi-channel audio (left right speakers) into mono channel
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: # ex shape is (2, 1000)
            signal = torch.mean(signal, dim=0, keepdim=True) #want to have 1 channel
        return signal

    #get the path of the actual audio
    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 4])
        return path
    

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 5]

if __name__ == "__main__":
    ANNOTATIONS_FILE = "/Users/saranorouzinia/Documents/Anomaly Sound Detection/code/annotations_file.xlsx"
    AUDIO_DIR = "/Users/saranorouzinia/Documents/Anomaly Sound Detection/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE, #max frequency we can analyze
        n_fft=1024, #number of samples per FFT window
        hop_length=512, #how much the window moves forward each time
        n_mels=64 
    )

    usd = AnimalSoundDataset(ANNOTATIONS_FILE, 
                            AUDIO_DIR, 
                            mel_spectrogram, 
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    print(f"There are {len(usd)} samples in the dataset.")
    signal, label = usd[0]
    a = 1