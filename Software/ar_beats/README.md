# AR-BEATs — Autoregressive Anomaly Detection for Underwater Bioacoustics

Unsupervised anomaly detection in underwater acoustic recordings using
BEATs patch embeddings and a spatial autoregressive CNN.

Inspired by: Erdil et al., "Spatial Autoregressive Modeling of DINOv3
Embeddings for Unsupervised Anomaly Detection", arXiv:2603.02974, 2026.

---

## Project Structure

```
ar_beats/
├── configs/
│   └── config.yaml          # All hyperparameters in one place
├── data/
│   ├── dataset.py           # Dataset class + mel-spectrogram pipeline
│   └── transforms.py        # Per-frequency normalization
├── models/
│   ├── beats_encoder.py     # Frozen BEATs wrapper → token grid
│   └── ar_cnn.py            # Masked + dilated AR CNN (diagonal Gaussian)
├── training/
│   └── trainer.py           # Training loop, early stopping, checkpointing
├── evaluation/
│   └── evaluate.py          # AUROC, AP, per-clip scoring
├── utils/
│   ├── logging.py           # Logging + loss curve saving
│   └── seed.py              # Reproducibility
├── notebooks/
│   └── explore.ipynb        # EDA, spectrogram visualization
├── checkpoints/             # Saved model weights (git-ignored)
├── results/                 # AUROC/AP outputs, curves (git-ignored)
├── train.py                 # Entry point — training
├── evaluate.py              # Entry point — evaluation
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Train
python train.py --config configs/config.yaml

# Evaluate
python evaluate.py --config configs/config.yaml \
                   --checkpoint checkpoints/best.pt
```

---

## Key Design Decisions

| Choice | Value | Rationale |
|---|---|---|
| Encoder | BEATs (frozen) | Token-level 2D grid, analogous to DINOv3 |
| Token grid | 49 × 12 × 768 | 5s clip, 128 mel bins, stride 10 |
| Covariance | Diagonal Gaussian | Non-uniform variance across BEATs dims |
| AR layers | 6 masked conv | Single forward pass, no memory bank |
| Dilation | d=8, k=3×3 | RF=97 tokens ≈ 9.7s, covers cetacean calls |
| Aggregation | Max-pooling | Sparse event detection in 5s clips |
| Metrics | AUROC + AP | AP robust to 15% class imbalance |

---

## TODO

- [ ] **[ASD — pre-submission]** Replace `max_epochs: 100` in
      `configs/config.yaml` with actual stopping epoch from first
      training run (early stopping patience = 10).
      *Calendar reminder set: 2026-03-28.*
