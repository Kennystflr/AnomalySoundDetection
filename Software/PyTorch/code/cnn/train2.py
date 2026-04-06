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
from animalsounddataset2 import AnimalSoundDataset
from cnn2 import ConvNeXtBinary

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001 #adjust?

ANNOTATIONS_FILE = "/Users/saranorouzinia/Documents/Anomaly Sound Detection/AnomalySoundDetection/Software/Perch2.0/V2/rapport_anomalies_optimize.csv"
AUDIO_DIR = "/Users/saranorouzinia/Documents/Anomaly Sound Detection/audio"
SAMPLE_RATE = 32000 #1 sec
NUM_SAMPLES = 32000 * 5 #5 seconds


def train_one_epoch(model, data_loader, loss_fn, optimiser, device, epoch_losses):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device) #assign these to a device

        targets = targets.unsqueeze(1)  # add this line to match BCEWithLogitsLoss shape: [128] → [128, 1]

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
    train_size = int(0.95 * len(usd)) #95% training data
    test_size = len(usd) - train_size #5% test data

    train_dataset, test_dataset = random_split(usd, [train_size, test_size]) 

    # Create loaders
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    print(f"Training using {device} device")
    cnn = ConvNeXtBinary().to(device)
    #print(cnn)

    #4 - train model

    labels = usd.annotations.iloc[:,6].map({"RAS":0,"ANOMALIE":1}).fillna(0)
    class_counts = np.bincount(labels.astype(int))

    # BCEWithLogitsLoss with pos_weight for Cross Entry Loss function and Sigmoid function (Logistic Regression)
    # pos_weight = number of normal samples / number of anomaly samples -> DON'T HARDCODE
    
    pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #regularizer applied : tweak value tho
    #Only passing classifier head to optimizer, not the frozen layer because it's unnecessary
    optimiser = torch.optim.Adam(
        cnn.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    ) #updates the model's weights to minimize loss
    all_epoch_losses = train(cnn, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnnnet2.pth") #storing the model
    print("Model trained and stored at cnnnet2.pth")
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
        