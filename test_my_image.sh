#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Error: You must provide the path to an image."
    exit 1
fi

IMAGE_PATH="$1"

echo "Testing with image: $IMAGE_PATH"
echo "Using model: outputs/best_scratch.pt"

python predict.py --checkpoint outputs/best_scratch.pt --haar --images "$IMAGE_PATH" --save-dir predictions

echo ""
echo "Check the 'predictions' folder for the image with the result."
echo ""
