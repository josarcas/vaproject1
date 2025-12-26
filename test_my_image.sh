#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Error: Debes proporcionar la ruta de una imagen."
    echo "Uso: ./test_my_image.sh camino/a/tu/foto.jpg"
    exit 1
fi

IMAGE_PATH="$1"

echo "Probando con imagen: $IMAGE_PATH"
echo "Usando modelo: outputs/best_scratch.pt"

python predict.py --checkpoint outputs/best_scratch.pt --haar --images "$IMAGE_PATH" --save-dir predictions

echo ""
echo "Revisa la carpeta 'predictions' para ver la imagen con el resultado."
echo ""
