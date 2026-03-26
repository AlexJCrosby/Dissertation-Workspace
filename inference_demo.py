from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from audio_preprocessing import (
    load_wav_16k_mono,
    pad_or_trim,
    waveform_to_spectrogram,
    spectrogram_to_log_mel,
)
from dataset_pipeline import build_datasets, DATASET_PATH
from transfer_model import TransferConfig, build_transfer_model


TARGET_SAMPLE_RATE = 16000


def record_wav(path: Path, duration: float = 1.0, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """
    Record mono audio from the default microphone and save it as a WAV file.

    Requires:
        pip install sounddevice scipy
    """
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except ImportError as exc:
        raise ImportError(
            "Missing packages for microphone recording. Install with: pip install sounddevice scipy"
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
    """
    Load a saved model.

    For transfer MobileNetV2 checkpoints, rebuild the architecture and load weights,
    matching the pattern already used in evaluate_model.py.
    """
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
    """
    Preprocess a WAV file for inference using the same steps as training.

    Returns shape: [time, mel, channels]
    """
    file_tensor = tf.constant(str(wav_path))
    waveform = load_wav_16k_mono(file_tensor)
    waveform = pad_or_trim(waveform)
    spectrogram = waveform_to_spectrogram(waveform)
    logmel = spectrogram_to_log_mel(spectrogram)
    logmel = tf.expand_dims(logmel, axis=-1)
    return logmel



def predict_file(model: tf.keras.Model, wav_path: Path, class_names: list[str]) -> tuple[str, np.ndarray]:
    """
    Preprocess a WAV file using the existing training-compatible pipeline and return:
    - predicted class label
    - probability vector
    """
    features = preprocess_wav_for_inference(wav_path)
    features = tf.expand_dims(features, axis=0)  # [1, time, mel, channels]

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
        description="Run single-file speech-command inference using a trained dissertation model."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved .keras model file",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional path to an existing WAV file. If omitted, audio is recorded from microphone.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Recording duration in seconds when using microphone input",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    train_ds, _, _, info = build_datasets(str(DATASET_PATH))
    class_names = info["label_vocab"]

    example_batch = next(iter(train_ds))
    x_batch, _ = example_batch
    input_shape = tuple(x_batch.shape[1:])

    print("\n--- Inference Setup ---")
    print("Model path:", model_path)
    print("Input shape:", input_shape)
    print("Num classes:", info["num_classes"])

    model = load_inference_model(
        model_path=model_path,
        input_shape=input_shape,
        num_classes=info["num_classes"],
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
