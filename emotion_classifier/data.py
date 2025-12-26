from typing import List, Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def build_transforms(*, image_size: int, train: bool, grayscale: bool):
    ops = []

    if grayscale:
        ops.append(transforms.Grayscale(num_output_channels=3))

    ops.append(transforms.Resize((image_size, image_size)))

    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ]
        )
        if not grayscale:
            ops.append(
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02
                )
            )

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transforms.Compose(ops)


def build_dataloaders(
    *,
    train_dir: str,
    val_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    grayscale: bool = False,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_tfms = build_transforms(image_size=image_size, train=True, grayscale=grayscale)
    val_tfms = build_transforms(image_size=image_size, train=False, grayscale=grayscale)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    class_names = [c for c, _ in sorted(train_ds.class_to_idx.items(), key=lambda x: x[1])]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_names
