import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animalsounddataset import AnimalSoundDataset
from cnn import CNNNetwork


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001
ANNOTATIONS_FILE = "/Users/saranorouzinia/Documents/Anomaly Sound Detection/code/annotations_file.xlsx"
AUDIO_DIR = "/Users/saranorouzinia/Documents/Anomaly Sound Detection/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def train_one_epoch(model, data_loader, loss_fn, optimiser, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device) #assign these to a device

        #1 - calculate loss (forward pass)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets) #loss function

        #2 - backpropagate loss and update weights (backwards pass)
        optimiser.zero_grad() #opimiser calculates gradient, this resets it to zero
        loss.backward() #actually performs back propegation
        optimiser.step() #updates weights
    
    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimiser, device)
        print("--------------")

    print("Training is done.")

if __name__ == "__main__":
    #get device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    #instantiate dataset object
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

    #2 - create data loader
    train_data_loader = DataLoader(usd, batch_size=BATCH_SIZE) #loads in batches of 128 (its an iterable)
    #how does this work with our data?


    print(f"Using {device} device")
    cnn = CNNNetwork().to(device)
    print(cnn)

    #4 - train model
    loss_fn = nn.CrossEntropyLoss() #loss function
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "feedforwardnet.pth") #storing the model
    print("Model trained and stored at feedforwardnet.pth")
    