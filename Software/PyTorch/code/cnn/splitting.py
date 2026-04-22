import numpy as np
import re
from torch.utils.data import Subset
from torch.utils.data import Subset as TorchSubset

def _get_annotations_and_indices(dataset):
    """Unwrap a Subset to get (base_dataset, list_of_indices_into_base)."""
    if isinstance(dataset, TorchSubset):
        return dataset.dataset.annotations, dataset.indices
    else:
        return dataset.annotations, list(range(len(dataset)))

def get_file_based_splits(dataset, train_size=0.8, val_size=0.2, random_state=42):
    annotations, available_indices = _get_annotations_and_indices(dataset)

    # 1. Group by base file
    file_to_indices = {}
    for idx in available_indices:
        filename = annotations.iloc[idx, 0]
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
    train_files = file_names[:n_train]
    test_files  = file_names[n_train:]

    # 4. Expand file splits to segment indices
    train_indices = [idx for f in train_files for idx in file_to_indices[f]]
    test_indices  = [idx for f in test_files  for idx in file_to_indices[f]]

    root = dataset.dataset if isinstance(dataset, TorchSubset) else dataset
    return Subset(root, train_indices), Subset(root, test_indices)