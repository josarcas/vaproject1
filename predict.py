import argparse
from pathlib import Path

import torch

from emotion_classifier.inference import predict_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--images", type=str, nargs="+", required=True)
    parser.add_argument("--haar", action="store_true")
    parser.add_argument("--save-dir", type=str, default="predictions")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = predict_images(
        checkpoint_path=args.checkpoint,
        image_paths=args.images,
        device=device,
        use_haar_crop=args.haar,
        save_dir=save_dir,
    )

    for r in results:
        print(f"{r.image_path} -> {r.predicted_label} | probs={r.probabilities}")


if __name__ == "__main__":
    main()
