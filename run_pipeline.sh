#!/bin/bash
set -e

echo "=================================================="
echo "     EXEC PIPELINE"
echo "=================================================="

echo ""
echo "[1/3] InstaLL requiriments..."
pip install -r requirements.txt

echo ""
echo "[2/3] Download dataset..."
python download_dataset.py

echo ""
echo "[3/3] Training..."
echo "      Save in 'outputs'"
python train.py --train-dir "datos_zip/dataset_emociones/train" --val-dir "datos_zip/dataset_emociones/validation" --output-dir outputs


# 4. Comparativa de Modelos
echo ""
echo "[4/3] Starting Model Comparison (ResNet18 vs MobileNetV3)..."

echo ">>> (A) Training ResNet18..."
python train.py --train-dir "datos_zip/dataset_emociones/train" --val-dir "datos_zip/dataset_emociones/validation" --model resnet18 --output-dir outputs_resnet --batch-size 32

echo ""
echo ">>> (B) Training MobileNetV3..."
python train.py --train-dir "datos_zip/dataset_emociones/train" --val-dir "datos_zip/dataset_emociones/validation" --model mobilenet_v3_small --output-dir outputs_mobilenet --batch-size 32

echo ""
echo "=================================================="
echo "     FULL PIPELINE COMPLETED"
echo "=================================================="
echo "Reports available at:"
echo "1. Basic:     outputs/training_summary_scratch.txt"
echo "2. ResNet18:  outputs_resnet/training_summary_resnet18.txt"
echo "3. MobileNet: outputs_mobilenet/training_summary_mobilenet_v3_small.txt"
echo ""
