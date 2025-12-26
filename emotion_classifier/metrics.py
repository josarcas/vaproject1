
#IMPORTS----------------------------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

#FUNCTIONS----------------------------------------------------------------------------------------------------------------
def save_training_curves(history: Dict[str, List[float]], path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    
    print(f"DEBUG: Plotting curves for {len(epochs)} epochs")
    print(f"DEBUG: Train loss: {history['train_loss']}")
    print(f"DEBUG: Val loss: {history['val_loss']}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="train", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="val", marker="o")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="train", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="val", marker="o")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, history["train_f1"], label="train", marker="o")
    axes[2].plot(epochs, history["val_f1"], label="val", marker="o")
    axes[2].set_title("F1 (macro)")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_classification_report(
    *,
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    path: Path,
) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    path.write_text(report, encoding="utf-8")


def _plot_confusion(cm: np.ndarray, class_names: List[str], title: str, path: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if cm.dtype != int else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_confusion_matrices(
    *,
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    abs_path: Path,
    norm_path: Path,
) -> None:
    cm_abs = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    _plot_confusion(cm_abs.astype(int), class_names, "Confusion (abs)", abs_path)
    _plot_confusion(cm_norm, class_names, "Confusion (norm)", norm_path)
