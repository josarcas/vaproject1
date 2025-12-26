from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    f1_macro: float
    top2_accuracy: float = 0.0
    y_true: Optional[List[int]] = None
    y_pred: Optional[List[int]] = None


@torch.no_grad()
def _forward_collect(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    losses = []
    y_true: List[int] = []
    y_pred: List[int] = []
    
    total_correct_top2 = 0
    total_samples = 0

    for xb, yb in tqdm(loader, desc="eval", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())
        preds = torch.argmax(logits, dim=1)
        
        # Calculate Top-2
        _, top2_preds = logits.topk(2, dim=1)
        total_correct_top2 += (top2_preds == yb.view(-1, 1)).sum().item()
        total_samples += yb.size(0)

        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    loss_mean = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    f1 = float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0
    top2_acc = float(total_correct_top2 / total_samples) if total_samples > 0 else 0.0

    return EpochMetrics(
        loss=loss_mean,
        accuracy=acc,
        f1_macro=f1,
        top2_accuracy=top2_acc,
        y_true=y_true,
        y_pred=y_pred,
    )


def train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    losses = []
    y_true: List[int] = []
    y_pred: List[int] = []
    
    total_correct_top2 = 0
    total_samples = 0

    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = torch.argmax(logits.detach(), dim=1)
        
        # Calculate Top-2
        _, top2_preds = logits.detach().topk(2, dim=1)
        total_correct_top2 += (top2_preds == yb.view(-1, 1)).sum().item()
        total_samples += yb.size(0)

        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    loss_mean = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    f1 = float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0
    top2_acc = float(total_correct_top2 / total_samples) if total_samples > 0 else 0.0

    return EpochMetrics(
        loss=loss_mean, accuracy=acc, f1_macro=f1, top2_accuracy=top2_acc
    )


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_preds: bool = False,
) -> EpochMetrics:
    metrics = _forward_collect(
        model=model, loader=loader, criterion=criterion, device=device
    )
    if return_preds:
        return metrics
    return EpochMetrics(
        loss=metrics.loss,
        accuracy=metrics.accuracy,
        f1_macro=metrics.f1_macro,
        top2_accuracy=metrics.top2_accuracy,
    )
