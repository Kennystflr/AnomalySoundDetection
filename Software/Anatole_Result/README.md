# 📊 Anatole Results & Data Chunks

This directory serves as a storage and validation layer containing evaluation data, split experimental chunks, and parsed performance metrics.

## Contents

*   **`.wav` Part Files**: Sliced, pre-processed audio files (`ml17_280a_xxxx_partyyy.wav`). These are used either as holdout validation chunks or direct evaluation slices for the Anomaly Sound Detection algorithms.
*   **`Anatole_result.csv`**: Compiled model output or annotation result matrices specific to 'Anatole' evaluations.
*   **`Perch_compare.csv`**: A spreadsheet analyzing the correlation/difference between results using different thresholding techniques natively compared against Perch embedding baselines.

These files act as the benchmark references to reproduce or compare current model iterations against established baseline runs.
