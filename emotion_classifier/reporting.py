import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from collections import Counter

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_confusion(y_true: List[int], y_pred: List[int], class_names: List[str]) -> str:
    cm = confusion_matrix(y_true, y_pred)
    # Zero out diagonal to ignore correct predictions
    np.fill_diagonal(cm, 0)
    
    # Flatten and sort to find max complications
    cm_flat = cm.flatten()
    indices = np.argsort(cm_flat)[::-1] # Descending
    
    report = []
    total_errors = np.sum(cm)
    if total_errors == 0:
        return "No errors found."

    for i in range(min(3, len(indices))):
        idx = indices[i]
        if cm_flat[idx] == 0:
            break
        true_idx = idx // len(class_names)
        pred_idx = idx % len(class_names)
        count = cm_flat[idx]
        percent = (count / len(y_true)) * 100
        report.append(
            f"{i+1}. True: {class_names[true_idx]} -> Pred: {class_names[pred_idx]} "
            f"({count} times, {percent:.2f}% of total data)"
        )
    
    return "\n".join(report)

def generate_report(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    history: Dict[str, List[float]],
    class_names: List[str],
    final_metrics: Any,
    output_path: Path,
) -> None:
    
    lines = []
    lines.append("=" * 60)
    lines.append("       EMOTION CLASSIFIER - TRAINING REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # 1. Dataset Info
    lines.append("1. DATASET INFORMATION")
    lines.append("-" * 40)
    lines.append(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    lines.append(f"Training Images:   {len(train_loader.dataset)}")
    lines.append(f"Validation Images: {len(val_loader.dataset)}")
    lines.append(f"Batch Size:        {train_loader.batch_size}")
    lines.append("")

    # 2. Network Architecture
    lines.append("2. NETWORK ARCHITECTURE & EFFICIENCY")
    lines.append("-" * 40)
    lines.append(f"Total Trainable Parameters: {count_parameters(model):,}")
    lines.append("")
    lines.append("Model Structure:")
    lines.append(str(model))
    lines.append("")

    # 3. Training History
    lines.append("3. TRAINING HISTORY")
    lines.append("-" * 40)
    header = f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Train Acc':^10} | {'Val Acc':^10} | {'Top-2 Acc':^10}"
    lines.append(header)
    lines.append("-" * len(header))
    
    epochs = len(history["train_loss"])
    # Handle cases where top2 might not be in older history dicts if user retrains without clearing
    # But since we just added it, we assume it's there or handle gracefully
    top2_hist = history.get("val_top2", [0.0]*epochs)
    
    for i in range(epochs):
        lines.append(
            f"{i+1:^6} | "
            f"{history['train_loss'][i]:^10.4f} | "
            f"{history['val_loss'][i]:^10.4f} | "
            f"{history['train_acc'][i]:^10.4f} | "
            f"{history['val_acc'][i]:^10.4f} | "
            f"{top2_hist[i]:^10.4f}"
        )
    lines.append("")

    # 4. Final Results
    lines.append("4. FINAL METRICS (Best Model Evaluation)")
    lines.append("-" * 40)
    lines.append(f"Accuracy:       {final_metrics.accuracy:.4f}")
    lines.append(f"Top-2 Accuracy: {final_metrics.top2_accuracy:.4f}")
    lines.append(f"F1 Score (Mac): {final_metrics.f1_macro:.4f}")
    lines.append(f"Loss:           {final_metrics.loss:.4f}")
    lines.append("")
    
    # 5. Confusion Analysis
    lines.append("5. CONFUSION ANALYSIS (Top 3 Errors)")
    lines.append("-" * 40)
    lines.append(analyze_confusion(final_metrics.y_true, final_metrics.y_pred, class_names))
    lines.append("")
    
    # Write to file
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Detailed report saved to: {output_path}")
