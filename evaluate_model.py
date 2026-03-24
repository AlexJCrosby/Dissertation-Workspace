from __future__ import annotations
from transfer_model import minmax_normalize_per_sample
from transfer_model import TransferConfig, build_transfer_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from dataset_pipeline import build_datasets, DATASET_PATH


def collect_y_true_and_probs(model: tf.keras.Model, dataset: tf.data.Dataset):
    """
    Runs model prediction over a dataset and returns:
    - y_true: true integer labels
    - y_pred: predicted integer labels
    - y_probs: softmax probabilities
    """
    y_true_batches = []
    for _, y_batch in dataset:
        y_true_batches.append(y_batch.numpy())

    y_true = np.concatenate(y_true_batches, axis=0)

    y_probs = model.predict(dataset, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    return y_true, y_pred, y_probs


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "Confusion Matrix",
):
    """
    Saves a confusion matrix figure.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Optional cell text for smaller class counts only
    if len(class_names) <= 15:
        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved speech command classification model.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to saved .keras model file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional run name for output folder naming",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    run_name = args.name if args.name else model_path.stem

    # ----------------------------
    # Output directory
    # ----------------------------
    out_dir = Path("outputs") / "evaluation" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load test data
    # ----------------------------
    _, _, test_ds, info = build_datasets(DATASET_PATH)
    class_names = info["label_vocab"]

    print("\n--- Evaluation Setup ---")
    print("Model path:", model_path)
    print("Run name:", run_name)
    print("Num test samples:", info["num_test"])
    print("Num classes:", info["num_classes"])

    # ----------------------------
    # Load model
    # ----------------------------
    if "transfer_mobilenetv2" in model_path.name:
        # Rebuild transfer model architecture, then load saved weights
        example_batch = next(iter(test_ds))
        x_batch, _ = example_batch
        input_shape = tuple(x_batch.shape[1:])

        cfg = TransferConfig(
            input_shape=input_shape,
            num_classes=info["num_classes"],
            image_size=224,
            dropout_rate=0.30,
            dense_units=128,
            learning_rate=1e-3,
            backbone_trainable=False,
        )

        model = build_transfer_model(cfg)
        model.load_weights(str(model_path))
    else:
        model = tf.keras.models.load_model(model_path)
        
    model.summary()

    # ----------------------------
    # Keras evaluation
    # ----------------------------
    print("\n--- Keras Evaluate (Test Set) ---")
    eval_results = model.evaluate(test_ds, verbose=2, return_dict=True)

    # ----------------------------
    # Predictions
    # ----------------------------
    print("\n--- Collecting Predictions ---")
    y_true, y_pred, y_probs = collect_y_true_and_probs(model, test_ds)

    # ----------------------------
    # Metrics
    # ----------------------------
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    summary = {
        "model_path": str(model_path),
        "run_name": run_name,
        "num_test_samples": int(len(y_true)),
        "num_classes": int(info["num_classes"]),
        "test_loss": float(eval_results.get("loss", float("nan"))),
        "test_accuracy": float(eval_results.get("accuracy", float("nan"))),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }

    # ----------------------------
    # Save outputs
    # ----------------------------
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (out_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    (out_dir / "classification_report.json").write_text(
        json.dumps(report_dict, indent=2),
        encoding="utf-8",
    )

    np.save(out_dir / "confusion_matrix.npy", cm)
    np.save(out_dir / "y_true.npy", y_true)
    np.save(out_dir / "y_pred.npy", y_pred)
    np.save(out_dir / "y_probs.npy", y_probs)

    save_confusion_matrix_plot(
        cm=cm,
        class_names=class_names,
        save_path=out_dir / "confusion_matrix.png",
        title=f"Confusion Matrix - {run_name}",
    )

    # ----------------------------
    # Print summary
    # ----------------------------
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n--- Classification Report ---")
    print(report_text)

    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    main()