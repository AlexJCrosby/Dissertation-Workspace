import os
from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf

# Import your preprocessing function (make sure audio_preprocessing.py is in the same folder)
from audio_preprocessing import preprocess_file_to_logmel, DATASET_PATH


# ----------------------------
# Config
# ----------------------------
BATCH_SIZE = 32
SHUFFLE_BUFFER = 4000  # adjust if you like
AUTOTUNE = tf.data.AUTOTUNE

# Speech Commands usually provides these official split files in the dataset root
VALIDATION_LIST = "validation_list.txt"
TESTING_LIST = "testing_list.txt"


# ----------------------------
# File discovery + official splits
# ----------------------------
def list_all_wavs(dataset_path: str) -> List[str]:
    """
    Recursively list all wav files under dataset_path, excluding underscore folders (e.g. _background_noise_).
    Assumes structure: <root>/<label>/<file>.wav
    """
    root = Path(dataset_path)
    wavs: List[str] = []

    for label_dir in root.iterdir():
        if not label_dir.is_dir():
            continue
        if label_dir.name.startswith("_"):
            continue

        wavs.extend([str(p) for p in label_dir.glob("*.wav")])

    wavs.sort()
    if not wavs:
        raise FileNotFoundError("No .wav files found. Check DATASET_PATH.")
    return wavs


def read_split_file(dataset_path: str, filename: str) -> List[str]:
    """
    Reads validation_list.txt / testing_list.txt which store relative paths like:
    'bed/01234567_nohash_0.wav'
    Returns absolute paths.
    """
    split_path = Path(dataset_path) / filename
    if not split_path.exists():
        return []

    lines = split_path.read_text(encoding="utf-8").splitlines()
    # Make absolute paths
    abs_paths = [str(Path(dataset_path) / line.strip()) for line in lines if line.strip()]
    return abs_paths


def build_official_splits(dataset_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Uses official Speech Commands split lists if present.
    Train = all_wavs - val - test
    """
    all_wavs = set(list_all_wavs(dataset_path))

    val_wavs = set(read_split_file(dataset_path, VALIDATION_LIST))
    test_wavs = set(read_split_file(dataset_path, TESTING_LIST))

    # Keep only those that actually exist in the folder (robustness)
    val_wavs = val_wavs.intersection(all_wavs)
    test_wavs = test_wavs.intersection(all_wavs)

    train_wavs = list(all_wavs.difference(val_wavs).difference(test_wavs))
    val_wavs = list(val_wavs)
    test_wavs = list(test_wavs)

    train_wavs.sort()
    val_wavs.sort()
    test_wavs.sort()

    # If split files not found, fall back to a simple random split
    if len(val_wavs) == 0 and len(test_wavs) == 0:
        all_list = list(all_wavs)
        rng = tf.random.Generator.from_seed(42)
        idx = rng.shuffle(tf.range(len(all_list))).numpy()
        all_list = [all_list[i] for i in idx]

        n = len(all_list)
        n_test = int(0.1 * n)
        n_val = int(0.1 * n)

        test_wavs = sorted(all_list[:n_test])
        val_wavs = sorted(all_list[n_test : n_test + n_val])
        train_wavs = sorted(all_list[n_test + n_val :])

    return train_wavs, val_wavs, test_wavs


# ----------------------------
# Label vocabulary + encoding
# ----------------------------
def get_label_from_path(path: tf.Tensor) -> tf.Tensor:
    parts = tf.strings.split(path, os.sep)
    return parts[-2]  # folder name


def build_label_vocab(file_paths: List[str]) -> List[str]:
    """
    Builds sorted list of label names from provided file paths.
    """
    labels = sorted({Path(p).parent.name for p in file_paths})
    if not labels:
        raise ValueError("No labels found when building vocabulary.")
    return labels


def make_label_lookup(labels: List[str]) -> tf.lookup.StaticHashTable:
    """
    Creates a lookup table: label_string -> label_id
    """
    keys = tf.constant(labels)
    values = tf.constant(list(range(len(labels))), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=-1,
    )
    return table


# ----------------------------
# Dataset building
# ----------------------------
def preprocess_and_encode_label(
    file_path: tf.Tensor,
    label_table: tf.lookup.StaticHashTable
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    file_path -> (logmel_features, label_id)
    """
    features, label_str = preprocess_file_to_logmel(file_path)
    label_id = label_table.lookup(label_str)
    return features, label_id


def make_dataset(
    file_paths: List[str],
    label_table: tf.lookup.StaticHashTable,
    training: bool
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(file_paths)

    if training:
        ds = ds.shuffle(buffer_size=min(SHUFFLE_BUFFER, len(file_paths)), reshuffle_each_iteration=True)

    ds = ds.map(lambda p: preprocess_and_encode_label(p, label_table), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)

    return ds


def build_datasets(dataset_path: str = DATASET_PATH):
    train_files, val_files, test_files = build_official_splits(dataset_path)

    # Build label vocab from TRAIN (common practice). If you prefer all labels, use train+val+test.
    label_vocab = build_label_vocab(train_files)
    label_table = make_label_lookup(label_vocab)

    train_ds = make_dataset(train_files, label_table, training=True)
    val_ds = make_dataset(val_files, label_table, training=False)
    test_ds = make_dataset(test_files, label_table, training=False)

    info = {
        "num_train": len(train_files),
        "num_val": len(val_files),
        "num_test": len(test_files),
        "label_vocab": label_vocab,
        "num_classes": len(label_vocab),
    }
    return train_ds, val_ds, test_ds, info


# ----------------------------
# Quick sanity check
# ----------------------------
if __name__ == "__main__":
    train_ds, val_ds, test_ds, info = build_datasets(DATASET_PATH)

    print("\n--- Dataset Pipeline Summary ---")
    print(f"Train files: {info['num_train']}")
    print(f"Val files:   {info['num_val']}")
    print(f"Test files:  {info['num_test']}")
    print(f"Num classes: {info['num_classes']}")
    print("First 10 labels:", info["label_vocab"][:10])

    # Take one batch and print shapes
    for features, label_ids in train_ds.take(1):
        print("\n--- One Batch ---")
        print("Features shape:", features.shape)  # (batch, time, mel, channels)
        print("Label IDs shape:", label_ids.shape)
        print("Label IDs (first 10):", label_ids.numpy()[:10])