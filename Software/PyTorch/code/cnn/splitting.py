import numpy as np
import re
from torch.utils.data import Subset

def get_file_based_splits(dataset, train_size=0.8, test_size=0.2, random_state=42):
    # 1. Group by base file
    file_to_indices = {}
    for idx in range(len(dataset)):
        filename = dataset.annotations.iloc[idx, 0]
        base_file = re.sub(r'_part\d+\.wav$', '.wav', filename)
        if base_file not in file_to_indices:
            file_to_indices[base_file] = []
        file_to_indices[base_file].append(idx)

    # 2. Shuffle files
    file_names = list(file_to_indices.keys())
    np.random.seed(random_state)
    np.random.shuffle(file_names)

    # 3. Split files into train and test
    n = len(file_names)
    n_train = int(n * train_size)
    train_files = set(file_names[:n_train])
    test_files  = set(file_names[n_train:])

    # 4. Expand file splits to segment indices
    train_indices = [idx for f in train_files for idx in file_to_indices[f]]
    test_indices  = [idx for f in test_files for idx in file_to_indices[f]]

    return Subset(dataset, train_indices), Subset(dataset, test_indices)