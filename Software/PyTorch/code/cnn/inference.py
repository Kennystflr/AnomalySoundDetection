import torch
import torchaudio
from cnn import CNNNetwork
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animalsounddataset import AnimalSoundDataset
from train import AUDIO_DIR, NUM_SAMPLES, SAMPLE_RATE, ANNOTATIONS_FILE

#make these the sounds we are identifying
class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

def predict(model, input, target, class_mapping):
    model.eval() #pytorch method that switches the model from training mode to evaluation mode
    with torch.no_grad(): # makes it so that the model doesn't calculate the grad because it is
                    # not needed when we are not training
        predictions = model(input)
        # Tensor (1, 10) -> given 1 class of input, tries to predict 10 values
        
        # We are interested with the index with the highest value - highest chance
        predicted_index = predictions[0].argmax(0) #index zero because we only have 1 sample
        #map predicted index to a class
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    
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

    #get a sample from the validation dataset for inference
    input, target = usd[0][0], usd[0][1] #initial sample both input and target
    input.unsqueeze_(0)
    #[batch size, num_channels, fr, time]
    #make an inference - will build a new function
    predicted, expected = predict(cnn, input, target, class_mapping) #NN's know nothing about the classes, 
                                                                                #they just use integers. class_mapping will map the integers to the classses
    print(f"Predicted: '{predicted}', expected: '{expected}'")