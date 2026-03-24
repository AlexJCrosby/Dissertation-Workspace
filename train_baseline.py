from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf

from dataset_pipeline import build_datasets, DATASET_PATH
from baseline_model import BaselineConfig, build_baseline_cnn

def main():
    # Build datasets
    train_ds, val_ds, test_ds, info = build_datasets(DATASET_PATH)

    # Infer input shape from a single batch (robust)
    example_batch = next(iter(train_ds))
    x_batch, y_batch = example_batch
    input_shape = tuple(x_batch.shape[1:])  # (time, mel, channels)

    print("\n--- Training Setup ---")
    print("Input shape:", input_shape)
    print("Num classes:", info["num_classes"])
    print("Train/Val/Test:", info["num_train"], info["num_val"], info["num_test"])

    # Build + compile model
    cfg = BaselineConfig(
        input_shape=input_shape,
        num_classes=info["num_classes"],
        dropout_rate=0.30,
        learning_rate=1e-3,
    )
    model = build_baseline_cnn(cfg)
    model.summary()

    # Output dirs
    out_dir = Path("outputs")
    model_dir = out_dir / "models"
    out_dir.mkdir(exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    ckpt_path = model_dir / "baseline_cnn_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
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
            filename=str(out_dir / "baseline_training_log.csv"),
            append=False,
        ),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks,
    )

    # Evaluate
    print("\n--- Final Evaluation (Test Set) ---")
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save final model after training finishes
    final_path = model_dir / "baseline_cnn_final.keras"
    model.save(str(final_path))
    print("Saved best model to:", ckpt_path)
    print("Saved final model to:", final_path)


if __name__ == "__main__":
    main()