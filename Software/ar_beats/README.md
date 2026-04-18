# AR-BEATs: Autoregressive Anomalous Sound Detection

Unsupervised anomaly detection for underwater audio. A frozen **BEATs** encoder extracts 2D token grids from raw waveforms; a lightweight **AR CNN** is trained to predict the distribution of normal audio. At inference, anomaly score = negative log-likelihood of the observed tokens under the learned normal model.

Inspired by: Erdil et al., "Spatial Autoregressive Modeling of DINOv3 Embeddings for Unsupervised Anomaly Detection", arXiv:2603.02974, 2026.

---

## Architecture

```
Raw waveform (16 kHz)
        │
        ▼
 BEATs Encoder (frozen)          ← pretrained on AudioSet
        │  128-bin mel → patch embedding (patch_size=16)
        ▼
 Token grid  (B, 31, 8, 768)     ← 31 time × 8 freq × 768-dim
        │
        ▼
 AR CNN (trainable)              ← 6 masked dilated conv layers
        │  predicts μ, σ² per token via raster-scan causal masking
        ▼
 NLL score per token → max over grid → clip-level anomaly score
```

- Each token covers **160 ms × 16 mel bins**
- Default receptive field: **97 tokens ≈ 9.7 s**
- ~11M trainable parameters

| Choice | Value | Rationale |
|---|---|---|
| Encoder | BEATs (frozen) | Token-level 2D grid, analogous to DINOv3 |
| Covariance | Diagonal Gaussian | Non-uniform variance across BEATs dims |
| AR layers | 6 masked conv | Single forward pass, no memory bank |
| Dilation | d=8, k=3 | RF=97 tokens ≈ 9.7s, covers cetacean calls |
| Aggregation | Max-pooling | Sparse event detection in 5s clips |
| Metrics | AUROC + AP | AP robust to class imbalance |

---

## Prerequisites

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. BEATs source code

The BEATs encoder requires Microsoft's source to be on `PYTHONPATH`. The `models/unilm/` directory is already included in this repo. Set the path before running any script:

```bash
# from Software/ar_beats/
export PYTHONPATH=$PYTHONPATH:$(pwd)/models/unilm/beats
```

If `models/unilm/` is missing, clone it:

```bash
git clone https://github.com/microsoft/unilm models/unilm
```

### 3. BEATs pretrained checkpoint — required before any run

Download the pretrained weights (~340 MB) into the `checkpoints/` directory:

```bash
mkdir -p checkpoints
wget -L "https://aka.ms/beats/BEATs_iter3_plus_AS2M.pt" -O checkpoints/BEATs_iter3_plus_AS2M.pt
```

Verify the download succeeded (should be ~340 MB, **not** an HTML error page):

```bash
python -c "import torch; ck = torch.load('checkpoints/BEATs_iter3_plus_AS2M.pt', weights_only=False); print('OK, keys:', list(ck.keys()))"
```

---

## Data Format

Provide a CSV with two columns:

```
filename,label
ml19_292b_0027_chunk0052.wav,0
ml19_292b_0065_chunk0025.wav,1
```

- `label=0` → normal, `label=1` → anomalous
- Filenames must follow the `<recording_id>_chunk<N>.wav` pattern — all chunks from the same recording are kept in the same split to prevent data leakage
- Audio can be any sample rate / channel count (resampled to 16 kHz mono automatically)

Set `root_dir` and `metadata_file` in `configs/config.yaml`.

---

## Configuration

All scripts share `configs/config.yaml`:

| Section | Key | Default | Description |
|---|---|---|---|
| `data` | `root_dir` | — | Directory containing `.wav` files |
| `data` | `metadata_file` | — | CSV path (filename, label) |
| `data` | `clip_duration` | 5.0 | Clip length in seconds |
| `beats` | `model_path` | `checkpoints/BEATs_iter3_plus_AS2M.pt` | Checkpoint path |
| `ar_cnn` | `n_layers` | 6 | Number of masked conv layers |
| `ar_cnn` | `dilation` | 8 | Dilation factor |
| `ar_cnn` | `hidden_dim` | 512 | Feature dimension |
| `training` | `batch_size` | 64 | Clips per batch |
| `training` | `max_epochs` | 150 | Upper epoch limit |
| `training` | `lr` | 1e-3 | Learning rate (AdamW) |
| `training` | `early_stopping_patience` | 6 | Epochs without val improvement before stopping |
| `training` | `checkpoint_dir` | `checkpoints` | Where to save model artifacts |
| `evaluation` | `aggregation` | `max` | Token→clip aggregation (`max` or `mean`) |

---

## Usage

All commands should be run from `Software/ar_beats/` with `PYTHONPATH` set as above.

### Train

```bash
python train.py --config configs/config.yaml
```

Produces:

| File | Description |
|---|---|
| `checkpoints/best.pt` | Best model weights (lowest val NLL) |
| `checkpoints/token_mean.npy` | Per-dimension token mean for normalization |
| `checkpoints/splits.csv` | Recording→split assignments for reproducibility |
| `results/logs/training_curves.png` | Train / val NLL curves |

### Evaluate (held-out test set)

```bash
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
```

Reports **AUROC** and **Average Precision** on the test split.


## Project Structure

```
ar_beats/
├── configs/
│   └── config.yaml          # All hyperparameters in one place
├── data/
│   └── dataset.py           # Recording-stratified splits, audio loading
├── models/
│   ├── beats_encoder.py     # Frozen BEATs wrapper → token grid
│   ├── ar_cnn.py            # Masked dilated AR CNN (diagonal Gaussian)
│   └── unilm/beats/         # BEATs source (microsoft/unilm)
├── training/
│   └── trainer.py           # Training loop, early stopping, checkpointing
├── utils/
│   ├── logging.py
│   └── seed.py
├── checkpoints/             # Created at runtime (git-ignored)
├── results/                 # Created at runtime (git-ignored)
├── train.py                 # Entry point — training
└── evaluate.py              # Entry point — evaluation
```
