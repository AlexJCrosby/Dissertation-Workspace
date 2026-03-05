import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/baseline_training_log.csv")

plt.figure()
plt.plot(df["accuracy"], label="train_accuracy")
plt.plot(df["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/training_accuracy.png")

plt.figure()
plt.plot(df["loss"], label="train_loss")
plt.plot(df["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/training_loss.png")

plt.show()