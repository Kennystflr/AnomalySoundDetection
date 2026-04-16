# ⚡ PyTorch Experiments & Custom Loaders

This directory contains PyTorch-specific code for manipulating, loading, and modeling audio data.

## Contents

*   **`code/`**: The main directory containing Python implementations.
    *   `animalsounddataset.py` & `animalsounddataset2.py`: Custom PyTorch `Dataset` wrappers designed to ingest raw audio and labels from the accompanying annotation spreadsheets seamlessly.
    *   `annotations_file.xlsx`: Ground truth annotations linking audio splits to their specific classes and tags.
    *   `cnn/`: Experimental Convolutional Neural Network architectures for direct audio classification or feature projection.
*   **`audio/`**: Directory meant to hold smaller sub-samples of audio data specifically used to validate the PyTorch datasets.

## Usage Overview

These scripts are intended to be imported into main training loops. They efficiently handle multi-threading, dynamic batching, and on-the-fly transformations needed for neural network processing in the broader scope of the anomaly sound detection project.
