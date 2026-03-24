import os
from pathlib import Path

import tensorflow as tf
import numpy as np

# Optional: for plotting
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
DATASET_PATH = Path("/data")

TARGET_SAMPLE_RATE = 16000
CLIP_DURATION_S = 1
DESIRED_SAMPLES = TARGET_SAMPLE_RATE * CLIP_DURATION_S

# STFT params
FRAME_LENGTH = 255
FRAME_STEP = 128
FFT_LENGTH = 256

# Mel params
NUM_MEL_BINS = 40
LOWER_EDGE_HZ = 20.0
UPPER_EDGE_HZ = 8000.0  # Nyquist for 16kHz audio = 8kHz


# ----------------------------
# Core helpers
# ----------------------------
def load_wav_16k_mono(file_path: tf.Tensor) -> tf.Tensor:
    """
    Loads a WAV file into a float32 1D tensor waveform.
    Assumes mono audio or squeezes to mono.
    Validates sample rate (Speech Commands is typically 16kHz).
    """
    audio_binary = tf.io.read_file(file_path)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)  # shape: [num_samples]

    sample_rate = tf.cast(sample_rate, tf.int32)

    tf.debugging.assert_equal(
        sample_rate,
        TARGET_SAMPLE_RATE,
        message="Unexpected sample rate. Expected 16000Hz.",
    )

    return waveform


def pad_or_trim(waveform: tf.Tensor, desired_samples: int = DESIRED_SAMPLES) -> tf.Tensor:
    """
    Ensures waveform is exactly desired_samples long.
    - If too short: pad zeros at end
    - If too long: trim
    """
    waveform_len = tf.shape(waveform)[0]

    def pad():
        padding = desired_samples - waveform_len
        return tf.pad(waveform, paddings=[[0, padding]], mode="CONSTANT")

    def trim():
        return waveform[:desired_samples]

    waveform = tf.cond(waveform_len < desired_samples, pad, trim)
    waveform.set_shape([desired_samples])
    return waveform


def waveform_to_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    """
    Converts waveform -> magnitude spectrogram using STFT.
    Returns shape: [time_frames, freq_bins]
    """
    stft = tf.signal.stft(
        waveform,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
    )
    spectrogram = tf.abs(stft)
    return spectrogram


def spectrogram_to_log_mel(spectrogram: tf.Tensor) -> tf.Tensor:
    """
    Converts a linear-frequency magnitude spectrogram to log-mel spectrogram.
    Returns shape: [time_frames, num_mel_bins]
    """
    num_spectrogram_bins = spectrogram.shape[-1]  # freq_bins
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=TARGET_SAMPLE_RATE,
        lower_edge_hertz=LOWER_EDGE_HZ,
        upper_edge_hertz=UPPER_EDGE_HZ,
    )

    mel_spectrogram = tf.matmul(spectrogram, mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram


def file_path_to_label(file_path: tf.Tensor) -> tf.Tensor:
    """
    Extracts label from folder name: .../<label>/<file>.wav
    Returns a string tensor label.
    """
    parts = tf.strings.split(file_path, os.sep)
    # Label folder is parent directory name
    label = parts[-2]
    return label


def preprocess_file_to_logmel(file_path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Full preprocessing pipeline for one file:
    file_path -> waveform -> fixed length -> spectrogram -> log-mel
    Returns: (features, label_string)
    """
    waveform = load_wav_16k_mono(file_path)
    waveform = pad_or_trim(waveform)

    spec = waveform_to_spectrogram(waveform)
    logmel = spectrogram_to_log_mel(spec)

    # Add channel dim so it looks like an "image" for CNN: [T, M, 1]
    logmel = tf.expand_dims(logmel, axis=-1)

    label = file_path_to_label(file_path)
    return logmel, label


# ----------------------------
# Quick sanity-check runner
# ----------------------------
def pick_one_example_wav(dataset_path: str) -> str:
    """
    Picks the first wav file from the first non-underscore directory.
    Mirrors the approach in dataset_inspection.py but in a reusable way.
    """
    dataset_path = Path(dataset_path)
    class_dirs = [p for p in dataset_path.iterdir() if p.is_dir() and not p.name.startswith("_")]
    class_dirs.sort(key=lambda p: p.name.lower())

    if not class_dirs:
        raise FileNotFoundError("No class folders found. Check DATASET_PATH.")

    wavs = sorted(class_dirs[0].glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found in {class_dirs[0]}")

    return str(wavs[0])


def plot_waveform_and_logmel(waveform: np.ndarray, logmel: np.ndarray, title: str = ""):
    """
    waveform: [16000]
    logmel: [time_frames, mel_bins]
    """
    plt.figure()
    plt.title(f"Waveform {title}")
    plt.plot(waveform)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title(f"Log-Mel Spectrogram {title}")
    plt.imshow(logmel.T, aspect="auto", origin="lower")
    plt.xlabel("Time frame")
    plt.ylabel("Mel bin")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_path = pick_one_example_wav(DATASET_PATH)
    print("Example file:", example_path)

    # Run preprocessing
    file_tensor = tf.constant(example_path)
    features, label = preprocess_file_to_logmel(file_tensor)

    print("Label:", label.numpy().decode("utf-8"))
    print("Features shape [time, mel, channels]:", features.shape)

    # Also show waveform length check
    wf = load_wav_16k_mono(file_tensor)
    wf = pad_or_trim(wf)
    print("Waveform shape:", wf.shape)

    # Plot
    plot_waveform_and_logmel(
        waveform=wf.numpy(),
        logmel=tf.squeeze(features, axis=-1).numpy(),
        title=f"({label.numpy().decode('utf-8')})",
    )