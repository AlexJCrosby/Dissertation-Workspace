from __future__ import annotations

from pathlib import Path
from typing import cast
from datetime import datetime

import re

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

from dataset_pipeline import build_datasets, DATASET_PATH
from transfer_model import TransferConfig, build_transfer_model

def get_next_run_dir(base_dir: Path) -> Path:
    """
    Create the next numbered run directory, e.g.:
    run_001, run_002, run_003, ...
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    existing = []
    for p in base_dir.iterdir():
        if p.is_dir():
            match = re.match(r"run_(\d+)", p.name)
            if match:
                existing.append(int(match.group(1)))

    next_num = max(existing, default=0) + 1
    run_dir = base_dir / f"run_{next_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir



def main():
    # ----------------------------
    # 1) Build datasets
    # ----------------------------
    train_ds, val_ds, test_ds, info = build_datasets(str(DATASET_PATH))

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
        learning_rate=1e-5,
        backbone_trainable=True,   # stage 1: frozen backbone
    )

    model = build_transfer_model(cfg)
    model.summary()

    # ----------------------------
    # 3) Output directories
    # ----------------------------
    out_dir = Path("outputs")
    runs_dir = out_dir / "transfer_runs"
    run_dir = get_next_run_dir(runs_dir)

    best_model_path = run_dir / "transfer_mobilenetv2_best.keras"
    final_model_path = run_dir / "transfer_mobilenetv2_final.keras"
    log_path = run_dir / "transfer_training_log.csv"

    print("Run directory:", run_dir)

    # SAVE RUN CONFIG
    (run_dir / "run_notes.txt").write_text(
        f"run_time={datetime.now()}\n"
        f"image_size={cfg.image_size}\n"
        f"dropout_rate={cfg.dropout_rate}\n"
        f"dense_units={cfg.dense_units}\n"
        f"learning_rate={cfg.learning_rate}\n"
        f"backbone_trainable={cfg.backbone_trainable}\n",
        encoding="utf-8",
    )   

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
        epochs=20,
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