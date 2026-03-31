import numpy as np
import os
from typing import Literal

from cs336_systems.util import constants


def load_dataset(dataset_key: str, split: Literal["train", "valid"] = "train"):
    input_path = constants.get_encoded_dataset_path(dataset_key=dataset_key, split=split)
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        exit(1)

    return np.load(input_path, mmap_mode='r')