import numpy as np
import torch
import torchaudio
from cnn2 import ConvNeXtBinary
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animalsounddataset2 import AnimalSoundDataset, ExpertResultDataset
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
from splitting import get_file_based_splits
from train2 import AUDIO_DIR, NUM_SAMPLES, SAMPLE_RATE, ANNOTATIONS_FILE


EXPERT_CSV = "/home/GTL/snorouzi/Documents/Anomaly Sound Detection/AnomalySoundDetection/Software/Expert_Result/Expert_result.csv"
EXPERT_AUDIO_DIR = "/home/GTL/snorouzi/Documents/Anomaly/ml17_280a_5sec"

def evaluate(model, data_loader, device, threshold):
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.sigmoid(outputs) #converts logits to probabilities
            probs = probs.squeeze(1)             # [batch, 1] → [batch]
            preds = (probs >= threshold).long()        # threshold calculated using PR curve

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_targets, all_probs

def predict(model, input, target, class_mapping):
    model.eval() #pytorch method that switches the model from training mode to evaluation mode
    device = next(model.parameters()).device #get the device of the model parameters (cpu or gpu)
    input = input.to(device)
    target = target.to(device)

    with torch.no_grad(): # makes it s    class_mapping = {0: "RAS", 1: "ANOMALIE"}o that the model doesn't calculate the grad because it is
                    # not needed when we are not training
        predictions = model(input)
        # Tensor (1, 10) -> given 1 class of input, tries to predict 10 values

        prob = torch.sigmoid(predictions).item()          # single probability
        predicted_index = 1 if prob >= BEST_THRESHOLD else 0   # threshold
        # We are interested with the index with the highest value - highest chance

        #map predicted index to a class
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target.item()]
    
    return predicted, expected


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    #load back the modeltorch.device
    cnn = ConvNeXtBinary().to(device)
    checkpoint = torch.load("cnnnet2.pth", map_location=device, weights_only=False) #loading the model we stored
    cnn.load_state_dict(checkpoint["model_state_dict"]) #loading the dict into the model
    BEST_THRESHOLD = checkpoint["threshold"]

    #load our dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate = SAMPLE_RATE, #max frequency we can analyze
        n_fft=1024, #number of samples per FFT window
        hop_length=512, #how much the window moves forward each time
        n_mels=64 
    )

    #test_usd = ExpertResultDataset(
        #EXPERT_CSV,
        #AUDIO_DIR,
        #mel_spectrogram,
        #SAMPLE_RATE,
        #NUM_SAMPLES,
        #device="cpu"
    #)

    usd = AnimalSoundDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mel_spectrogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        "cpu"
    )

    class_mapping = {0: "Void", 1: "Anomaly"}

    #get a sample from the 
    #ion dataset for inference
    input, target = usd[0][0], usd[0][1] #initial sample both input and target
    input.unsqueeze_(0)
    #[batch size, num_channels, fr, time]
    target = target.long()  # class_mapping expects an int key
    #make an inference - will build a new function
    predicted, expected = predict(cnn, input, target, class_mapping) #NN's know nothing about the classes, 
                                                                                #they just use integers. class_mapping will map the integers to the classses

    for annotator in ['Exploration', 'Human_validation']:
        test_usd = ExpertResultDataset(
            EXPERT_CSV, EXPERT_AUDIO_DIR, mel_spectrogram,
            SAMPLE_RATE, NUM_SAMPLES, device="cpu",
            annotator=annotator
        )

        test_loader = DataLoader(test_usd, batch_size=32, shuffle=False) #the size of this batch size determines how fast it trains (larger == faster)

        predictions, targets, probabilities = evaluate(cnn, test_loader, device, threshold=BEST_THRESHOLD) #rerun using saved BEST_THRESHOLD

        # Compute and print F1 score
        f1 = f1_score(targets, predictions, average="weighted")
        print(f"\n=== Annotator: {annotator} ===")
        print(f"Weighted F1 Score: {f1:.4f}")


    
        report = classification_report(targets, predictions, labels=[0, 1],
                                    target_names=["Void", "Anomaly"],
                                    output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(f"classification_report_{annotator}.csv")

        cm = confusion_matrix(targets, predictions)
        cm_df = pd.DataFrame(cm, index=["Void Actual", "Anomaly Actual"], columns=["Void Predicted", "Anomaly Predicted"])
        with open("classification_report2.csv", "a") as f:
            f.write("\nConfusion Matrix\n")
            cm_df.to_csv(f)

        # Display confusion matrix as a grid table
        print("\nConfusion Matrix:")
        print(cm_df)

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.close()

        precision, recall, thresholds = precision_recall_curve(targets, probabilities)
        ap = average_precision_score(targets, probabilities)

        plt.figure()
        plt.plot(recall, precision, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.savefig("precision_recall_curve.png")
        plt.close()
