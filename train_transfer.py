from __future__ import annotations

from pathlib import Path
from typing import cast

import tensorflow as tf

from dataset_pipeline import build_datasets, DATASET_PATH
from transfer_model import TransferConfig, build_transfer_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    # ----------------------------
    # 1) Build datasets
    # ----------------------------
    train_ds, val_ds, test_ds, info = build_datasets(DATASET_PATH)

    # Infer input shape from one batch
    example_batch = next(iter(train_ds))
    x_batch, y_batch = example_batch
    input_shape = tuple(x_batch.shape[1:])  # (time, mel, channels)

    print("\n--- Transfer Training Setup ---")
    print("Input shape:", input_shape)
    print("Num classes:", info["num_classes"])
    print("Train/Val/Test:", info["num_train"], info["num_val"], info["num_test"])

    # ----------------------------
    # 2) Build model
    # ----------------------------
    cfg = TransferConfig(
        input_shape=input_shape,
        num_classes=info["num_classes"],
        image_size=224,
        dropout_rate=0.30,
        dense_units=128,
        learning_rate=1e-3,
        backbone_trainable=False,   # stage 1: frozen backbone
    )

    model = build_transfer_model(cfg)
    model.summary()

    # ----------------------------
    # 3) Output directories
    # ----------------------------
    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    out_dir.mkdir(exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "transfer_mobilenetv2_best.keras"
    final_model_path = model_dir / "transfer_mobilenetv2_final.keras"
    log_path = out_dir / "transfer_training_log.csv"

    # ----------------------------
    # 4) Callbacks
    # ----------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(log_path),
            append=False,
        ),
    ]

    # ----------------------------
    # 5) Train
    # ----------------------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        callbacks=callbacks,
    )

    # ----------------------------
    # 6) Evaluate on test set
    # ----------------------------
    print("\n--- Final Evaluation (Test Set) ---")
    results = cast(dict[str, float], model.evaluate(test_ds, verbose=2, return_dict=True))

    test_loss = results["loss"]
    test_acc = results["accuracy"]

    print("\n--- Final Evaluation (Test Set) ---")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # ----------------------------
    # 7) Save final model
    # ----------------------------
    model.save(str(final_model_path))
    print("Saved best model to:", best_model_path)
    print("Saved final model to:", final_model_path)
    print("Saved training log to:", log_path)


if __name__ == "__main__":
    main()