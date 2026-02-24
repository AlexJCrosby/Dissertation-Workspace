import os
import tensorflow as tf
import numpy as np

# ----------------------------
# 1. Dataset Path
# ----------------------------
DATASET_PATH = r"C:\Users\Alex\OneDrive\Desktop\Year 3\0 - Dissertation\Datasets\Google Speech Commands"  

# ----------------------------
# 2. Count Samples Per Class
# ----------------------------
print("\n--- Dataset Overview ---\n")

class_counts = {}

for item in os.listdir(DATASET_PATH):
    item_path = os.path.join(DATASET_PATH, item)

    if os.path.isdir(item_path) and not item.startswith("_"):
        files = [f for f in os.listdir(item_path) if f.endswith(".wav")]
        class_counts[item] = len(files)

total_files = sum(class_counts.values())

for label, count in sorted(class_counts.items()):
    print(f"{label}: {count} samples")

print(f"\nTotal samples (excluding background noise): {total_files}")

# ----------------------------
# 3. Load One Example File
# ----------------------------
print("\n--- Inspecting One Example Audio File ---\n")

# Get first class
first_class = list(class_counts.keys())[0]
first_class_path = os.path.join(DATASET_PATH, first_class)

# Get first wav file
example_file = [f for f in os.listdir(first_class_path) if f.endswith(".wav")][0]
example_path = os.path.join(first_class_path, example_file)

# Load audio
audio_binary = tf.io.read_file(example_path)
waveform, sample_rate = tf.audio.decode_wav(audio_binary)

waveform = tf.squeeze(waveform)

duration = waveform.shape[0] / sample_rate.numpy()

print(f"Example file: {example_file}")
print(f"Class label: {first_class}")
print(f"Sample rate: {sample_rate.numpy()} Hz")
print(f"Number of samples: {waveform.shape[0]}")
print(f"Duration: {duration:.2f} seconds")