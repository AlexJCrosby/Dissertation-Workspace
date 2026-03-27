from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from audio_preprocessing import (
    DESIRED_SAMPLES,
    TARGET_SAMPLE_RATE,
    load_wav_16k_mono,
    pad_or_trim,
    waveform_to_spectrogram,
    spectrogram_to_log_mel,
)
from transfer_model import TransferConfig, build_transfer_model


DEFAULT_LABELS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow",
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine",
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero",
]


# The dissertation model was trained on a 30-class subset.
# We keep the 30-class fallback here so microphone inference can run locally
# even when the Docker-only /data mount is unavailable.
DEFAULT_30_CLASS_LABELS = sorted([
    "bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy",
    "house", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero",
])


def in_docker() -> bool:
    return Path("/.dockerenv").exists()


def infer_input_shape() -> tuple[int, int, int]:
    waveform = tf.zeros([DESIRED_SAMPLES], dtype=tf.float32)
    spec = waveform_to_spectrogram(waveform)
    logmel = spectrogram_to_log_mel(spec)
    return tuple(logmel.shape) + (1,)


def discover_labels_from_dataset(dataset_path: Path) -> list[str]:
    labels = sorted([
        p.name for p in dataset_path.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    ])
    if not labels:
        raise ValueError(f"No label folders found in dataset path: {dataset_path}")
    return labels


def get_class_names(dataset_path: str | None, num_classes: int) -> list[str]:
    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        labels = discover_labels_from_dataset(path)
        if len(labels) != num_classes:
            raise ValueError(
                f"Dataset contains {len(labels)} labels but model expects {num_classes}. "
                f"Point --dataset-path at the same dataset/version used for training."
            )
        return labels

    if num_classes == len(DEFAULT_30_CLASS_LABELS):
        return DEFAULT_30_CLASS_LABELS

    if num_classes == len(DEFAULT_LABELS):
        return sorted(DEFAULT_LABELS)

    raise ValueError(
        "Could not determine class names automatically. Provide --dataset-path to the dataset used for training."
    )


def record_wav(path: Path, duration: float = 1.0, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """
    Record mono audio from the default microphone and save it as a WAV file.

    Simplest path:
    - run this script locally on Windows/macOS/Linux, not inside Docker
    - install: pip install sounddevice scipy
    """
    if in_docker():
        raise RuntimeError(
            "Microphone recording is not available in this Docker container.\n"
            "Simplest fix: run inference_demo.py locally from your project .venv instead of inside Docker.\n"
            "Example:\n"
            "  .venv\\Scripts\\activate\n"
            "  pip install sounddevice scipy\n"
            "  python inference_demo.py --model-path outputs/transfer_runs/run_001/transfer_mobilenetv2_best.keras --mic --dataset-path \"C:\\Users\\Alex\\OneDrive\\Desktop\\Year 3\\0 - Dissertation\\Datasets\\Google Speech Commands\""
        )

    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except ImportError as exc:
        raise ImportError(
            "Missing microphone packages. Install with: pip install sounddevice scipy"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "Audio backend not available. On Windows, the simplest fix is to run this script in your local .venv, not Docker."
        ) from exc

    num_samples = int(duration * sample_rate)
    print(f"\nRecording for {duration:.1f} second(s)... Speak now.")
    audio = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")

    audio = np.squeeze(audio, axis=-1)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    write(str(path), sample_rate, audio_int16)



def load_inference_model(model_path: Path, input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    if "transfer_mobilenetv2" in model_path.name:
        cfg = TransferConfig(
            input_shape=input_shape,
            num_classes=num_classes,
            image_size=224,
            dropout_rate=0.30,
            dense_units=128,
            learning_rate=1e-3,
            backbone_trainable=False,
        )
        model = build_transfer_model(cfg)
        model.load_weights(str(model_path))
        return model

    return tf.keras.models.load_model(model_path)



def preprocess_wav_for_inference(wav_path: Path) -> tf.Tensor:
    file_tensor = tf.constant(str(wav_path))
    waveform = load_wav_16k_mono(file_tensor)
    waveform = pad_or_trim(waveform)
    spectrogram = waveform_to_spectrogram(waveform)
    logmel = spectrogram_to_log_mel(spectrogram)
    logmel = tf.expand_dims(logmel, axis=-1)
    return logmel



def predict_file(model: tf.keras.Model, wav_path: Path, class_names: list[str]) -> tuple[str, np.ndarray]:
    features = preprocess_wav_for_inference(wav_path)
    features = tf.expand_dims(features, axis=0)
    probs = model.predict(features, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    return pred_label, probs



def print_top_predictions(probs: np.ndarray, class_names: list[str], top_k: int = 5) -> None:
    top_indices = np.argsort(probs)[::-1][:top_k]
    print("\nTop predictions:")
    for i, idx in enumerate(top_indices, start=1):
        print(f"{i}. {class_names[idx]:<15} {probs[idx]:.4f}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-file or live-microphone speech-command inference using the dissertation model."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved .keras model file")
    parser.add_argument("--file", type=str, default=None, help="Path to an existing WAV file")
    parser.add_argument("--mic", action="store_true", help="Record from microphone instead of using --file")
    parser.add_argument("--duration", type=float, default=1.0, help="Recording duration in seconds")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional dataset root used to recover class names locally, e.g. your Google Speech Commands folder",
    )
    args = parser.parse_args()

    if bool(args.file) == bool(args.mic):
        raise ValueError("Choose exactly one input mode: either --file PATH or --mic")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    input_shape = infer_input_shape()

    # Your trained model currently uses 30 classes.
    # If you later swap models, update this or point --dataset-path to a matching dataset.
    num_classes = 30
    class_names = get_class_names(args.dataset_path, num_classes=num_classes)

    print("\n--- Inference Setup ---")
    print("Model path:", model_path)
    print("Input shape:", input_shape)
    print("Num classes:", num_classes)
    print("Input mode:", "microphone" if args.mic else "wav file")

    model = load_inference_model(
        model_path=model_path,
        input_shape=input_shape,
        num_classes=num_classes,
    )

    if args.file:
        wav_path = Path(args.file)
        if not wav_path.exists():
            raise FileNotFoundError(f"Input WAV file not found: {wav_path}")
    else:
        temp_dir = Path(tempfile.gettempdir())
        wav_path = temp_dir / "inference_demo_input.wav"
        record_wav(wav_path, duration=args.duration, sample_rate=TARGET_SAMPLE_RATE)

    print("Input WAV:", wav_path)
    pred_label, probs = predict_file(model, wav_path, class_names)

    print("\nPredicted class:", pred_label)
    print_top_predictions(probs, class_names, top_k=5)


if __name__ == "__main__":
    main()
