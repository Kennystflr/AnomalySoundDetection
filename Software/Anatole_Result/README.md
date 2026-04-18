# 📊 Anatole Results & Data Chunks

This directory serves as a storage and validation layer containing evaluation data, split experimental chunks, and parsed performance metrics.

## 📂 Contents

### 🎵 Audio Data
* **`.wav` Part Files**: Sliced and pre-processed audio files (e.g., `ml17_280a_xxxx_partyyy.wav`). 
    * These segments are used as **holdout validation chunks** or as direct evaluation slices for Anomaly Sound Detection (ASD) algorithms.

### 📈 Metrics & Results
* **`Anatole_result.csv`**: The primary output matrix for Anatole-specific evaluations. **This dataset was professionally annotated by Dr. Gros-Martial.**
    * **`exploration`**: Indicates there is something "interesting" in the sound, but it is **not necessarily an animal**. It marks a technical anomaly or a point of interest for further analysis.
    * **`validation_human`**: Confirms a **verified animal detection** identified by a human observer (Dr. Gros-Martial).
* **`Perch_compare.csv`**: A comparative spreadsheet analyzing the correlation and performance gaps between current results (using various thresholding techniques) and the established **Perch embedding** baselines.

---

## 🎯 Purpose
These files act as the **benchmark references** required to reproduce experimental runs or to compare current model iterations against established baseline performance. 

> **Note:** When filtering data, use `validation_human` for confirmed biological presence and `exploration` to analyze broader acoustic anomalies.