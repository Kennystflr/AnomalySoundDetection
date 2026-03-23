# Anomaly Sound Detection Data Loader

Complete data preparation pipeline for anomalous sound detection using **Perch 2.0** embeddings and mel-spectrograms.

## Features

- **Automatic Audio Chunking**: Splits long audio files into 5-second segments
- **Perch 2.0 Embeddings**: State-of-the-art bird sound embeddings using ONNX model
- **Mel-Spectrograms**: Extracts log-mel spectrograms for each chunk (16kHz, 64 mel-bins)
- **Automatic Labeling**: Detects anomalies using cosine distance from reference
- **Metadata Generation**: Generates comprehensive CSV with all chunk information
- **Reference Building**: Builds reference embeddings from normal audio samples

## Project Structure

```
anomaly-data-loader/
├── raw_data/                    # Input long audio files (.wav)
├── data/
│   ├── audio_chunks/            # 5s chunk WAV files
│   ├── mels/                    # Mel-spectrogram .npy files
│   ├── embeddings/              # Perch 2.0 embedding .npy files
│   ├── metadata.csv             # Main metadata file
│   ├── reference_embedding.npy  # Reference embedding (optional)
│   └── reference_std.npy        # Reference std deviation (optional)
├── src/
│   ├── __init__.py
│   ├── extractor.py             # Perch 2.0 extractor class
│   ├── prepare_features.py      # Main feature extraction pipeline
│   ├── ref_builder.py           # Reference embedding builder
│   ├── data_loader.py           # PyTorch Dataset classes
│   ├── saver.py                 # Utility save functions
│   └── utils.py
├── scripts/
│   └── run_pipeline.py          # Main orchestration script
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

```bash
# Clone and navigate
cd anomaly-data-loader

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- `librosa` - Audio processing
- `numpy`, `pandas` - Data handling
- `soundfile` - WAV I/O
- `torch` - PyTorch tensors
- `onnxruntime` - Perch 2.0 inference
- `scipy` - Distance metrics
- `tqdm` - Progress bars

## Usage

### 1. Prepare Input Data

Place your long audio files (`.wav` format) in the `raw_data/` directory:

```bash
raw_data/
├── recording1.wav
├── recording2.wav
└── ...
```

### 2. Run the Pipeline

```bash
cd anomaly-data-loader
python scripts/run_pipeline.py
```

This will:
1. Chunk all audio files into 5-second segments
2. Extract Perch 2.0 embeddings for each chunk
3. Extract mel-spectrograms for each chunk
4. Generate `metadata.csv` with all information

**Output:**
```
data/
├── audio_chunks/000001.wav      # 5s WAV chunk at 16kHz
├── mels/000001.npy              # Mel-spectrogram (64, T)
├── embeddings/000001.npy        # Perch embedding (D,)
└── metadata.csv
```

### 3. Optional: Build Reference Embeddings

If you have a folder of **normal/clean audio** chunks, build a reference for anomaly detection:

```bash
python -m src.ref_builder data/audio_chunks/normal data
```

Or from Python:

```python
from src.ref_builder import build_reference
build_reference("data/audio_chunks/normal", "data")
```

This generates:
- `reference_embedding.npy` - Mean embedding of normal audio
- `reference_std.npy` - Std deviation for stability analysis

**With reference embeddings**, the pipeline will:
- Compute cosine distance to reference for each chunk
- Auto-label chunks as anomaly (1) or normal (0) based on threshold (default: 0.25)
- Include `label` and `distance_to_ref` columns in metadata.csv

### 4. Load Data with PyTorch

Use the provided Dataset classes:

```python
from src.data_loader import AudioChunkDataset, MetaModalDataset
from torch.utils.data import DataLoader

# Simple loader (flexible)
dataset = AudioChunkDataset(
    "data/metadata.csv",
    load_mel=True,
    load_embedding=True,
    load_audio=True
)
loader = DataLoader(dataset, batch_size=32)

# Multi-modal loader (all three modalities)
mm_dataset = MetaModalDataset("data/metadata.csv")
loader = DataLoader(mm_dataset, batch_size=32)

for batch in loader:
    audio = batch["audio"]          # (B, 16000*5)
    mel = batch["mel"]              # (B, 64, T)
    embedding = batch["embedding"] # (B, D)
    label = batch["label"]          # (B,)
    distance = batch["distance"]    # (B,)
```

## Metadata CSV Format

`metadata.csv` contains:

| Column | Type | Description |
|--------|------|-------------|
| `chunk_id` | str | Unique chunk identifier |
| `audio_path` | str | Path to 5s WAV file |
| `mel_path` | str | Path to mel-spectrogram .npy |
| `embedding_path` | str | Path to Perch embedding .npy |
| `label` | int | 0=normal, 1=anomaly, -1=unknown |
| `distance_to_ref` | float | Cosine distance to reference (-1 if no ref) |

Example rows:
```
chunk_id,audio_path,mel_path,embedding_path,label,distance_to_ref
recording1_chunk0000,data/audio_chunks/recording1_chunk0000.wav,data/mels/recording1_chunk0000.npy,data/embeddings/recording1_chunk0000.npy,0,0.1823
recording1_chunk0001,data/audio_chunks/recording1_chunk0001.wav,data/mels/recording1_chunk0001.npy,data/embeddings/recording1_chunk0001.npy,1,0.2847
```

## Advanced Usage

### Custom Anomaly Threshold

Edit in `prepare_features.py`:

```python
ANOMALY_THRESHOLD = 0.25  # Change this (lower = more sensitive)
```

### Build Reference from Subfolder

```python
from pathlib import Path
from src.ref_builder import build_reference

build_reference("data/audio_chunks/normal", "data")
```

### Load Only Specific Modalities

```python
# Load only embeddings and labels (faster)
dataset = AudioChunkDataset(
    "data/metadata.csv",
    load_mel=False,
    load_embedding=True,
    load_audio=False
)
```

## Perch 2.0 Details

- **Input**: 32kHz mono audio, 5 seconds = 160,000 samples
- **Model**: ONNX-optimized version (from `Software/Perch2.0/perch_v2.onnx`)
- **Output**: Fixed-size embedding (typically 8192 dims for environmental/bird sound)
- **Distance Metric**: Cosine distance (0=identical, 2=orthogonal)
- **Threshold**: Typically 0.2-0.3 for anomaly detection

## References

- **Perch 2.0**: Advanced sound event detection model
- Bird sound embeddings provide excellent transfer learning for various anomaly detection tasks

## License

See LICENSE file.
