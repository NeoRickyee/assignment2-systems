import os
from typing import Literal

DATASETS = {
    "tinystory": "data/TinyStoriesV2-GPT4-train.txt",
    "openwebtext": "data/owt_train.txt",
}

DATASETS_VALID = {
    "tinystory": "data/TinyStoriesV2-GPT4-valid.txt",
    "openwebtext": "data/owt_valid.txt",
}

BPE_SAVE_DIR = {
    "tinystory": "data/tinystory",
    "openwebtext": "data/openwebtext",
}

CHECKPOINT_SAVE_DIR = {
    "tinystory": "checkpoint/tinystory",
    "openwebtext": "checkpoint/openwebtext"
}

FUNDAMENTAL_MODEL_SAVE_DIR = {
    "tinystory": "fundamental_model/tinystory",
    "openwebtext": "fundamental_model/openwebtext"
}

VOCAB_SIZE = {
    "tinystory": 10000,
    "openwebtext": 32000,
}

def init_directories(dataset_key: str):
    os.makedirs(CHECKPOINT_SAVE_DIR.get(dataset_key, "unknown"), exist_ok=True)
    os.makedirs(FUNDAMENTAL_MODEL_SAVE_DIR.get(dataset_key, "unknown"), exist_ok=True)

def get_vocab_path(dataset_key: str):
    return os.path.join(BPE_SAVE_DIR.get(dataset_key, "unknown"), "vocab.json")

def get_encoded_dataset_path(
    dataset_key: str, split: Literal["train", "valid"] = "train"
):
    file_name = f"{split}_encoded.npy"
    return os.path.join(BPE_SAVE_DIR.get(dataset_key, "unknown"), file_name)

def get_checkpoint_output_path(
    dataset_key: str, steps: int
):
    file_name = f"checkpoint_{steps}.pt"
    return os.path.join(CHECKPOINT_SAVE_DIR.get(dataset_key, "unknown"), file_name)

def get_fundamental_model_save_path(
    dataset_key: str
):
    file_name = "model.pt"
    return os.path.join(FUNDAMENTAL_MODEL_SAVE_DIR.get(dataset_key, "unknown"), file_name)
    