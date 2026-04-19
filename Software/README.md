# 💻 Software Implementations

This directory houses all the programmatic assets, models, and data pipelines for the Anomalous Sound Detection project.

## Directory Breakdown

*   **[`anomaly-data-loader/`](anomaly-data-loader/)**: A fully-featured data preparation pipeline. Processes raw `.wav` chunks into Mel-spectrograms and exacts embeddings via Perch 2.0. Includes utilities for building references and filtering anomalies via cosine distance.
*   **[`ar_beats/`](ar_beats/)**: An innovative unsupervised anomaly detection framework for underwater bioacoustics, heavily relying on BEATs patch embeddings combined with a spatial autoregressive CNN pipeline.
*   **[`PyTorch/`](PyTorch/)**: Contains raw PyTorch implementations ranging from Custom Dataset classes (`animalsounddataset.py`) to CNN experimental structures.
*   **[`Perch2.0/`](Perch2.0/)**: Codebase dedicated directly to leveraging Google's Perch v2 model for environmental acoustics.
*   **[`Expert_Result/`](Expert_Result/)**: Validation files, chunks, and `.csv` comparison matrices tracking model performance against reference standards.

Each major module contains its own descriptive `README.md` for specific execution environments and details.
