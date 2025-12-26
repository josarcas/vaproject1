from typing import Optional

import torch
from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            activation_fn(inplace=True) if activation_fn == nn.ReLU else activation_fn(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            activation_fn(inplace=True) if activation_fn == nn.ReLU else activation_fn(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            activation_fn(inplace=True) if activation_fn == nn.ReLU else activation_fn(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            activation_fn(inplace=True) if activation_fn == nn.ReLU else activation_fn(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def _set_parameter_requires_grad(model: nn.Module, feature_extract: bool) -> None:
    if not feature_extract:
        return
    for p in model.parameters():
        p.requires_grad = False


def build_pretrained(
    *,
    name: str,
    num_classes: int,
    feature_extract: bool,
    use_pretrained_weights: bool,
) -> nn.Module:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained_weights else None
        m = models.resnet18(weights=weights)
        _set_parameter_requires_grad(m, feature_extract)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "mobilenet_v3_small":
        weights = (
            models.MobileNet_V3_Small_Weights.DEFAULT if use_pretrained_weights else None
        )
        m = models.mobilenet_v3_small(weights=weights)
        _set_parameter_requires_grad(m, feature_extract)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m

    raise ValueError(f"Modelo pre-entrenado no soportado: {name}")


def get_activation_class(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "sigmoid":
        return nn.Sigmoid
    if name == "leaky_relu":
        return nn.LeakyReLU
    raise ValueError(f"Activation not supported: {name}")


def build_model(
    *,
    name: str,
    num_classes: int,
    feature_extract: bool = False,
    grayscale: bool = False,
    use_pretrained_weights: bool = True,
    activation_name: str = "relu",
) -> nn.Module:
    if name == "scratch":
        in_ch = 3
        act_cls = get_activation_class(activation_name)
        return SimpleCNN(
            num_classes=num_classes, in_channels=in_ch, activation_fn=act_cls
        )

    # Note: Pre-trained models have their own fixed activation functions.
    return build_pretrained(
        name=name,
        num_classes=num_classes,
        feature_extract=feature_extract,
        use_pretrained_weights=use_pretrained_weights,
    )
