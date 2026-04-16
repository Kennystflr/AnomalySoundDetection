import re
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
from sklearn.metrics import f1_score
from splitting import get_file_based_splits

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 5e-5 #adjust?

ANNOTATIONS_FILE = "/home/GTL/snorouzi/Documents/Anomaly Sound Detection/AnomalySoundDetection/Software/Perch2.0/V2/CSV/cosine_final_synced.csv"
AUDIO_DIR = "/home/GTL/snorouzi/Documents/Anomaly Sound Detection/audio"
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


def validate(model, data_loader, loss_fn, device):
    model.eval()
    all_probs = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets_unsqueezed = targets.unsqueeze(1)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets_unsqueezed)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).squeeze(1)

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Convert to numpy
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    thresholds = np.linspace(0.1, 0.9, 17)
    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_targets, preds, average="weighted", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    final_preds = (all_probs >= best_thresh).astype(int)
    print("Unique preds:", np.unique(final_preds))
    print("Unique targets:", np.unique(all_targets))

    avg_loss = total_loss / len(data_loader)
    weighted_f1 = f1_score(all_targets, final_preds, average="weighted", zero_division=0)

    return avg_loss, weighted_f1, best_thresh


def train(model, data_loader, val_loader, loss_fn, optimiser, device, epochs, scheduler):
    best_f1 = 0.0
    patience = 5 #early stopping patience
    epochs_no_improve = 0
    all_epoch_losses = []
    #val_losses = []
    #val_f1s = []

    for i in range(epochs):
        print(f"Epoch {i+1}")
        model.train()
        epoch_losses = []
        train_one_epoch(model, data_loader, loss_fn, optimiser, device, epoch_losses)

        val_loss, val_f1, thresh = validate(model, val_loader, loss_fn, device)
        scheduler.step(val_f1)

        #when val loss starts going up you are overfitting
        #val_losses.append(val_loss)
        #val_f1s.append(val_f1)
        print(f"Epoch {i+1} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        #early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0

            # save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "threshold": thresh
            }, "cnnnet2.pth")
        else:
            epochs_no_improve += 1

        all_epoch_losses.append(epoch_losses)

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break


    print(f"Training done.")
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
                            device="cpu")


    #generator = torch.Generator().manual_seed(42)
    #train_dataset, val_dataset, test_dataset = random_split(
    #    usd, [train_size, val_size, test_size], generator=generator
    #)
 # Split dataset (train/test only)
    train_val_dataset, test_dataset = get_file_based_splits(usd, train_size=0.7, test_size=0.3, random_state=42)
    train_dataset, val_dataset = get_file_based_splits(train_val_dataset, train_size=0.8, test_size=0.2, random_state=42)
    
    test_labels = [int(usd[i][1].item()) for i in test_dataset.indices]
    print("Test class counts:", np.bincount(test_labels))

    # Create loaders
    train_labels = [int(usd[i][1].item()) for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    
    class_weights = [1.0 / class_counts[label] for label in train_labels]

    #just on train dataset
    #sampler = torch.utils.data.WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)
    if class_counts[1] < class_counts[0]:
        pos_weight = torch.tensor([class_counts[0] / class_counts[1]]).to(device)
    else:
        pos_weight = torch.tensor([1.0]).to(device)
        print("Warning: Anomaly is majority — pos_weight is ignored")
    
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) #sampler is only for training
    test_data_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


    print(f"Training using {device} device")
    cnn = ConvNeXtBinary().to(device)
    #print(cnn)


    # BCEWithLogitsLoss with pos_weight for Cross Entry Loss function and Sigmoid function (Logistic Regression)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Class counts: Void={class_counts[0]}, Anomaly={class_counts[1]}")

    #regularizer applied : tweak value tho
    #Only passing classifier head to optimizer, not the frozen layer because it's unnecessary
    optimiser = torch.optim.Adam(
        cnn.parameters(), #all parameters - unfrozen backbone
        lr=5e-5,
        weight_decay=1e-4
    ) #updates the model's weights to minimize loss

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode='max', patience=2, factor=0.6
    )

    all_epoch_losses = train(cnn, train_data_loader, val_data_loader, loss_fn, optimiser, device, EPOCHS, scheduler)
    
    checkpoint = torch.load("cnnnet2.pth", map_location=device, weights_only=False)
    cnn.load_state_dict(checkpoint["model_state_dict"])
    best_thresh = checkpoint["threshold"]

    print("Model trained and stored at cnnnet2.pth")

    test_loss, test_f1, test_thresh = validate(cnn, test_data_loader, loss_fn, device) #testing on the "best model" we just loaded
    
    print(f"Test loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | Best threshold: {test_thresh:.2f}")
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
        