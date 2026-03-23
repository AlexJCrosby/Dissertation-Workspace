from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class TransferConfig:
    input_shape: tuple[int, int, int]   # e.g. (124, 40, 1)
    num_classes: int
    image_size: int = 224
    dropout_rate: float = 0.30
    dense_units: int = 128
    learning_rate: float = 1e-3
    backbone_trainable: bool = False    # False for stage 1, can enable later for fine-tuning


def minmax_normalize_per_sample(x: tf.Tensor) -> tf.Tensor:
    """
    Min-max normalise each spectrogram sample independently to [0, 1].
    This helps adapt log-mel inputs to an image-style pretrained backbone.
    """
    x_min = tf.reduce_min(x, axis=[1, 2, 3], keepdims=True)
    x_max = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-6)


def build_transfer_model(cfg: TransferConfig) -> keras.Model:
    """
    Transfer-learning classifier for log-mel spectrogram inputs.

    Pipeline:
    log-mel input
      -> per-sample min-max normalisation
      -> resize to ImageNet backbone size
      -> convert 1 channel to 3 channels
      -> MobileNetV2 backbone
      -> custom classification head
    """
    inputs = keras.Input(shape=cfg.input_shape, name="logmel_input")

    # 1) Normalise each sample to [0, 1]
    x = layers.Lambda(minmax_normalize_per_sample, name="minmax_norm")(inputs)

    # 2) Resize to backbone input size
    x = layers.Resizing(cfg.image_size, cfg.image_size, name="resize_to_backbone")(x)

    # 3) Convert grayscale spectrogram to RGB-like 3 channels
    x = layers.Lambda(
        lambda t: tf.image.grayscale_to_rgb(t),
        name="grayscale_to_rgb"
    )(x)

    # 4) Scale to the range expected by MobileNetV2 preprocess_input
    x = layers.Rescaling(255.0, name="to_255_range")(x)
    x = layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocess"
    )(x)

    # 5) Pretrained backbone
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(cfg.image_size, cfg.image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = cfg.backbone_trainable

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # 6) Classification head
    x = layers.Dropout(cfg.dropout_rate, name="dropout_1")(x)
    x = layers.Dense(cfg.dense_units, activation="relu", name="dense_1")(x)
    x = layers.Dropout(cfg.dropout_rate, name="dropout_2")(x)

    outputs = layers.Dense(
        cfg.num_classes,
        activation="softmax",
        name="class_probs"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="transfer_mobilenetv2_logmel"
    )

    optimizer = keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model