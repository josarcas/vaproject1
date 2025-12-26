from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from emotion_classifier.models import build_model


@dataclass
class PredictionResult:
    image_path: str
    predicted_label: str
    probabilities: Dict[str, float]


def _haar_crop_rgb(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    center_x, center_y = x + w // 2, y + h // 2
    side = max(w, h)
    half_side = side // 2

    x1 = max(center_x - half_side, 0)
    y1 = max(center_y - half_side, 0)
    x2 = min(center_x + half_side, image_bgr.shape[1])
    y2 = min(center_y + half_side, image_bgr.shape[0])

    cropped = image_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def _build_val_transform(*, image_size: int, grayscale: bool):
    ops = []
    if grayscale:
        ops.append(transforms.Grayscale(num_output_channels=3))

    ops.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return transforms.Compose(ops)


@torch.no_grad()
def predict_images(
    *,
    checkpoint_path: str,
    image_paths: Sequence[str],
    device: str,
    use_haar_crop: bool,
    save_dir: Path,
) -> List[PredictionResult]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name: str = str(ckpt.get("model_name", "scratch"))
    class_names: List[str] = ckpt["class_names"]
    image_size: int = int(ckpt.get("image_size", 224))
    grayscale: bool = bool(ckpt.get("grayscale", False))

    model = build_model(
        name=model_name,
        num_classes=len(class_names),
        grayscale=grayscale,
        feature_extract=False,
        use_pretrained_weights=False,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    model.to(device)

    tfm = _build_val_transform(image_size=image_size, grayscale=grayscale)

    results: List[PredictionResult] = []

    for p in image_paths:
        pth = Path(p)

        orig = Image.open(pth).convert("RGB")
        orig_np = np.array(orig)

        cropped_np = None
        if use_haar_crop:
            bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
            cropped_np = _haar_crop_rgb(bgr)

        model_input_img = Image.fromarray(cropped_np) if cropped_np is not None else orig

        x = tfm(model_input_img).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        idx = int(np.argmax(probs))
        pred_label = class_names[idx]
        prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        ncols = 4 if use_haar_crop else 3
        fig, axes = plt.subplots(1, ncols, figsize=(16, 4))

        axes[0].imshow(orig_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        col = 1
        if use_haar_crop:
            if cropped_np is None:
                axes[1].text(0.5, 0.5, "No se detecto rostro", ha="center", va="center")
                axes[1].axis("off")
            else:
                axes[1].imshow(cropped_np)
                axes[1].set_title("Haar crop")
                axes[1].axis("off")
            col = 2

        preproc = tfm(model_input_img)
        preproc_vis = preproc.detach().cpu().numpy().transpose(1, 2, 0)
        preproc_vis = (preproc_vis * np.array([0.229, 0.224, 0.225])) + np.array(
            [0.485, 0.456, 0.406]
        )
        preproc_vis = np.clip(preproc_vis, 0, 1)

        axes[col].imshow(preproc_vis)
        axes[col].set_title(f"Preprocesada\nPred: {pred_label}")
        axes[col].axis("off")

        ax_scores = axes[col + 1]
        order = np.argsort(probs)[::-1]
        top_labels = [class_names[i] for i in order]
        top_probs = probs[order]
        ax_scores.barh(top_labels[::-1], top_probs[::-1])
        ax_scores.set_xlim(0.0, 1.0)
        ax_scores.set_title("Scores (softmax)")

        fig.tight_layout()
        out_path = save_dir / f"{pth.stem}_pred.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

        results.append(
            PredictionResult(
                image_path=str(pth),
                predicted_label=pred_label,
                probabilities=prob_dict,
            )
        )

    return results
