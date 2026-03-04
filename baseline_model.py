from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class BaselineConfig:
    input_shape: tuple[int, int, int]   # (time, mel_bins, channels) e.g. (124, 40, 1)
    num_classes: int                    # e.g. 30
    dropout_rate: float = 0.30
    learning_rate: float = 1e-3


def build_baseline_cnn(cfg: BaselineConfig) -> keras.Model:
    """
    Small CNN baseline for log-mel spectrogram inputs.
    Includes Dropout for generalisation (per supervisor action points).
    """
    inputs = keras.Input(shape=cfg.input_shape, name="logmel_input")

    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout to help generalisation
    x = layers.Dropout(cfg.dropout_rate, name="dropout_1")(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_2")(x)

    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="class_probs")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn_logmel")

    optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],  # compile with accuracy
    )
    return model