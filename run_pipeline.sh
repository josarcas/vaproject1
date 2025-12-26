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

echo ""
echo "=================================================="
echo "     PIPELINE COMPLETED"
echo "=================================================="
echo "Check 'outputs/training_summary_scratch.txt' for the report."
echo ""
