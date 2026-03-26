import torch
import torchaudio
from cnn import CNNNetwork
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animalsounddataset import AnimalSoundDataset
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score, f1_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
from train import AUDIO_DIR, NUM_SAMPLES, SAMPLE_RATE, ANNOTATIONS_FILE

def evaluate(model, data_loader, device):
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            anomaly_probs = probs[:, 1]   # probability of class 1
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(anomaly_probs.cpu().numpy())

    return all_preds, all_targets, all_probs

def predict(model, input, target, class_mapping):
    model.eval() #pytorch method that switches the model from training mode to evaluation mode
    with torch.no_grad(): # makes it s    class_mapping = {0: "RAS", 1: "ANOMALIE"}o that the model doesn't calculate the grad because it is
                    # not needed when we are not training
        predictions = model(input)
        # Tensor (1, 10) -> given 1 class of input, tries to predict 10 values
        
        # We are interested with the index with the highest value - highest chance
        predicted_index = predictions[0].argmax(0) #index zero because we only have 1 sample
        #map predicted index to a class
        predicted = class_mapping[predicted_index.item()]
        expected = class_mapping[target.item()]
    
    return predicted, expected


if __name__ == "__main__":
    #load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnnnet.pth", map_location=torch.device('cpu')) #loading the model we stored
    cnn.load_state_dict(state_dict) #loading the dict into the model

    #load our dataset
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
                            "cpu")

    class_mapping = {0: "RAS", 1: "ANOMALIE"}

    #get a sample from the validation dataset for inference
    input, target = usd[0][0], usd[0][1] #initial sample both input and target
    input.unsqueeze_(0)
    #[batch size, num_channels, fr, time]
    #make an inference - will build a new function
    predicted, expected = predict(cnn, input, target, class_mapping) #NN's know nothing about the classes, 
                                                                                #they just use integers. class_mapping will map the integers to the classses
    
    #add metrics here
    train_size = int(0.9 * len(usd))
    test_size = len(usd) - train_size

    _, test_dataset = random_split(usd, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) #the size of this batch size determines how fast it trains (larger == faster)

    predictions, targets, probabilities = evaluate(cnn, test_loader, "cpu")

    # Compute and print F1 score
    f1 = f1_score(targets, predictions, average="weighted")
    print(f"Weighted F1 Score: {f1:.4f}")


    report = classification_report(targets, predictions, labels=[0, 1], target_names=["RAS", "ANOMALIE"],output_dict=True,
    zero_division=0)
    df = pd.DataFrame(report).transpose()

    df.to_csv("classification_report.csv")

    cm = confusion_matrix(targets, predictions)
    cm_df = pd.DataFrame(cm, index=["RAS Actual", "ANOMALIE Actual"], columns=["RAS Predicted", "ANOMALIE Predicted"])
    with open("classification_report.csv", "a") as f:
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

    
    #print(f"Predicted: '{predicted}', expected: '{expected}'")