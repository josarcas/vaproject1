#!/bin/bash
set -e

echo "=================================================="
echo "     EJECUTANDO PIPELINE COMPLETO (LINUX)"
echo "=================================================="

echo ""
echo "[1/3] Instalando dependencias..."
pip install -r requirements.txt

echo ""
echo "[2/3] Descargando Dataset (si no existe)..."
python download_dataset.py

echo ""
echo "[3/3] Iniciando Entrenamiento (usando config.py)..."
echo "      Los resultados se guardaran en la carpeta 'outputs'"
python train.py --train-dir "datos_zip/dataset_emociones/train" --val-dir "datos_zip/dataset_emociones/validation" --output-dir outputs

echo ""
echo "=================================================="
echo "     PROCESO COMPLETADO"
echo "=================================================="
echo "Revisa 'outputs/training_summary_scratch.txt' para el reporte."
echo ""
