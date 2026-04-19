# Anomaly Sound Detection

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

Welcome to the **Anomaly Sound Detection** repository! This is a research-oriented project dedicated to detecting anomalous sound events, with a strong focus on environmental and underwater bioacoustics. 

The repository encompasses several deep learning and machine learning approaches to identify unusual audio patterns without relying completely on labeled anomalous data (unsupervised/semi-supervised anomaly detection).

## Project Overview

Anomalous sound detection has critical applications in machinery monitoring, environmental conservation, and medical diagnostics. This project explores state-of-the-art anomaly detection techniques such as:
- **Spatial Autoregressive Modeling** on audio embeddings.
- **Autoencoders and CNNs** for direct spectrogram anomaly scoring.
- Utilizing robust pretrained models like **Perch 2.0** and **BEATs** to extract rich audio features.

## Repository Structure

- **[`Software/`](Software/)**: The core implementation directory containing various models and data pipelines.
  - **[`ar_beats/`](Software/ar_beats/)**: Autoregressive Anomaly Detection (AR-BEATs) tailored for underwater bioacoustics.
  - **[`anomaly-data-loader/`](Software/anomaly-data-loader/)**: A complete pipeline for extracting features via Perch 2.0 embeddings and log-mel spectrograms.
  - **[`PyTorch/`](Software/PyTorch/)**: Custom PyTorch datasets, CNN wrappers, and experimental code.
  - **[`Perch2.0/`](Software/Perch2.0/)**: Scripts and notebook comparisons utilizing the Perch 2.0 sound event detection model.
  - **[`Anatole_Result/`](Software/Anatole_Result/)**: Stored datasets, validation splits, and parsed anomaly comparisons (`.csv` / `.wav`).
- **[`papers/`](papers/)**: A curated collection of relevant research literature driving this project.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. Specific sub-projects may have their own `requirements.txt`. For the primary neural network architectures, you will likely need PyTorch and standard audio processing libraries.

```bash
# Clone the repository
git clone https://github.com/Kennystflr/AnomalySoundDetection.git
cd AnomalySoundDetection
```

### Downloading Pretrained Models

To utilize the Perch capabilities in this repository, you may need to download the Perch 2.0 model in `.onnx` format:
- 🔗 [**Download Perch 2.0 ONNX Model**](https://huggingface.co/justinchuby/Perch-onnx/resolve/main/perch_v2.onnx?download=true)

Place the model in the appropriate directory (e.g., inside `Software/Perch2.0/` or where referenced by your pipeline).

##  Literature & Research

We base our methodology on extensive research from the field of audio anomaly detection. Feel free to explore the `papers/` directory, which covers topics spanning from deep denoising autoencoders to WaveNet-based anomaly detection mechanisms.

##  Contributing

Contributions to this research project are welcome. Feel free to open an issue to discuss proposed features or bug fixes, and submit a Pull Request.

---

*This repository is continuously maintained as part of an ongoing research project on Anomalous Sound Detection.*
