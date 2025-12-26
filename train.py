import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from emotion_classifier.data import build_dataloaders
from emotion_classifier.metrics import (
    save_classification_report,
    save_confusion_matrices,
    save_training_curves,
)
from emotion_classifier.models import build_model
from emotion_classifier.reporting import generate_report
from emotion_classifier.train_eval import evaluate, train_epoch
from emotion_classifier.utils import get_device, save_checkpoint, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="scratch")
    parser.add_argument("--feature-extract", action="store_true")
    # Defaults from config.py
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--activation", type=str, default=config.ACTIVATION)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    train_loader, val_loader, class_names = build_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        grayscale=args.grayscale,
    )

    model = build_model(
        name=args.model,
        num_classes=len(class_names),
        feature_extract=args.feature_extract,
        grayscale=args.grayscale,
        activation_name=args.activation,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
        # New: Top-2 tracking for reporting
        "val_top2": [],
    }

    best_val_acc = -1.0
    best_ckpt_path = output_dir / f"best_{args.model}.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_metrics.loss)
        history["val_loss"].append(val_metrics.loss)
        history["train_acc"].append(train_metrics.accuracy)
        history["val_acc"].append(val_metrics.accuracy)
        history["train_f1"].append(train_metrics.f1_macro)
        history["val_f1"].append(val_metrics.f1_macro)
        history["val_top2"].append(val_metrics.top2_accuracy)

        scheduler.step(val_metrics.accuracy)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_metrics.loss:.4f} acc {train_metrics.accuracy:.4f} f1 {train_metrics.f1_macro:.4f} | "
            f"val loss {val_metrics.loss:.4f} acc {val_metrics.accuracy:.4f} f1 {val_metrics.f1_macro:.4f}"
        )

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                model_name=args.model,
                class_names=class_names,
                image_size=args.image_size,
                grayscale=args.grayscale,
            )

    save_training_curves(history, output_dir / f"curves_{args.model}.png")

    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    final_metrics = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        return_preds=True,
    )

    save_classification_report(
        y_true=final_metrics.y_true,
        y_pred=final_metrics.y_pred,
        class_names=class_names,
        path=output_dir / f"classification_report_{args.model}.txt",
    )

    save_confusion_matrices(
        y_true=final_metrics.y_true,
        y_pred=final_metrics.y_pred,
        class_names=class_names,
        abs_path=output_dir / f"confusion_abs_{args.model}.png",
        norm_path=output_dir / f"confusion_norm_{args.model}.png",
    )
    
    # Generate detailed report
    generate_report(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        history=history,
        class_names=class_names,
        final_metrics=final_metrics,
        output_path=output_dir / f"training_summary_{args.model}.txt",
    )

    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
