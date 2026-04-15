import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os
import re
import torch.nn as nn

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

        self.transformation = nn.Sequential(
            transformation,
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        ).to(self.device)

        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        # Define yo label mapping here
        self.label_map = {
            'Void': 0,
            'Anomaly': 1
        }
        self.default_label = 0  # Use 0 for unknown labels
        
    
    def _load_annotations(self, annotations_file):
        ext = os.path.splitext(annotations_file)[1].lower()  # get file extension
        if ext == ".csv":
            df = pd.read_csv(annotations_file)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(annotations_file)
        else:
            raise ValueError(f"Unsupported annotations file format: {ext}")
        
        df = df[df.iloc[:, 6].isin(["Void", "Anomaly"])].reset_index(drop=True) #only keeps rows that dont have doute (is this what we want tho?)
        return df

    def __len__(self):
        return len(self.annotations)

    #
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)

        start_time = int(self.annotations.iloc[index, 2]) #start time in seconds
        frame_offset = int(start_time * self.target_sample_rate) #five seconds
        num_frames = self.num_samples

        signal, sr = torchaudio.load(audio_sample_path, frame_offset=frame_offset, num_frames=num_frames) #signal and sample rate
        signal = signal.to(self.device)
        #standardizing
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal) #mix it down to mono
        signal = self._cut_if_necessary(signal) #too many samples
        signal = self._right_pad_if_necessary(signal) #zero-pad it if not enough samples
        signal = self.transformation(signal) #pass the signal into the mel spectrogram func
        # mel spectrogram: (1, 64, T)

        # 3. Normalize the tensor to [0, 1] range
        # Decibel values are usually negative (e.g., -80 to 0)
        dist_min, dist_max = signal.min(), signal.max()
        if dist_max - dist_min > 0:
            signal = (signal - dist_min) / (dist_max - dist_min)

        # Resize for ConvNeXt (224x224)
        signal = torch.nn.functional.interpolate(
            signal.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(0)  # (1, 224, 224)

        signal = signal.repeat(3, 1, 1)  # add this — repeats the 1 channel 3 times because ConvNext expects 3 channels not 1
        #signal is a tensor object -> (num_channels, samples)
        #now its # (3, 224, 224)

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
        label = torch.tensor(label, dtype=torch.float) #BCEWithLogitsLoss requres floats

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
        filename = self.annotations.iloc[index, 0]
        base_file = re.sub(r'_part\d+\.wav$', '.wav', filename)
        path = os.path.join(self.audio_dir, base_file)
        return path
    

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/GTL/snorouzi/Documents/Anomaly Sound Detection/AnomalySoundDetection/Software/Perch2.0/V2/CSV/cosine_final_synced.csv"
    AUDIO_DIR = "/home/GTL/snorouzi/Documents/Anomaly Sound Detection/audio"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050 * 5 # 5 seconds if NUM_SAMPLES = 5 * sample_rate
    
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
                            device="cpu")

    print(f"There are {len(usd)} samples in the dataset.")
    print("Class counts:", usd.annotations.iloc[:, 6].value_counts()[["Void", "Anomaly"]].fillna(0).to_dict())
    signal, label = usd[0]
    a = 1