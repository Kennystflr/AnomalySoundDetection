import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animalsounddataset import AnimalSoundDataset
from cnn import CNNNetwork


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001 #adjust?

ANNOTATIONS_FILE = "/Users/saranorouzinia/Documents/Anomaly Sound Detection/AnomalySoundDetection/Software/Perch2.0/rapport_anomalies.csv"
AUDIO_DIR = "/Users/saranorouzinia/Documents/Anomaly Sound Detection/AnomalySoundDetection/Software/PyTorch/audio"
SAMPLE_RATE = 32000 #1 sec
NUM_SAMPLES = 160000 #5 seconds


def train_one_epoch(model, data_loader, loss_fn, optimiser, device, epoch_losses):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device) #assign these to a device
        #print(targets)
        #1 - calculate loss (forward pass)
        predictions = model(inputs)
        loss = loss_fn(predictions, targets) #loss function

        #2 - backpropagate loss and update weights (backwards pass)
        optimiser.zero_grad() #opimiser calculates gradient, this resets it to zero
        loss.backward() #actually performs back propegation
        optimiser.step() #updates weights

        epoch_losses.append(loss.item())
    
    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    all_epoch_losses = []

    for i in range(epochs):
        print(f"Epoch {i+1}")
        epoch_losses = []
        train_one_epoch(model, data_loader, loss_fn, optimiser, device, epoch_losses)

        all_epoch_losses.append(epoch_losses)
        print("--------------")

    print("Training is done.")
    return all_epoch_losses

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

    # Split dataset
    train_size = int(0.9 * len(usd)) #90% training data
    test_size = len(usd) - train_size #10% test data

    train_dataset, test_dataset = random_split(usd, [train_size, test_size]) 

    # Create loaders
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    print(f"Training using {device} device")
    cnn = CNNNetwork().to(device)
    #print(cnn)

    #4 - train model

    labels = usd.annotations.iloc[:,5].map({"RAS":0,"ANOMALIE":1}).fillna(0)
    class_counts = np.bincount(labels.astype(int))

    weights = 1.0 / class_counts
    weights = torch.tensor(weights).float().to(device)

    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE) #updates the model's weights to minimize loss
    all_epoch_losses = train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnnnet.pth") #storing the model
    print("Model trained and stored at cnnnet.pth")
    all_losses = [loss for epoch in all_epoch_losses for loss in epoch]
    
    plt.hist(all_losses, bins=30)
    plt.xlabel("Loss value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Training Loss")
    plt.savefig("loss_histogram.png")
    plt.close()
    
    plt.plot(all_losses)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Raw Training Loss")
    plt.savefig("loss_raw_curve.png")
    plt.close()
        