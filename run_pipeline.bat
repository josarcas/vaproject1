@echo off
echo ==================================================
echo      EJECUTANDO PIPELINE COMPLETO
echo ==================================================

echo.
echo [1/3] Instalando dependencias...
pip install -r requirements.txt
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [2/3] Descargando Dataset (si no existe)...
python download_dataset.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [3/3] Iniciando Entrenamiento (usando config.py)...
echo       Los resultados se guardaran en la carpeta 'outputs'
python train.py --train-dir datos_zip/dataset_emociones/train --val-dir datos_zip/dataset_emociones/validation --output-dir outputs
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==================================================
echo      PROCESO COMPLETADO
echo ==================================================
echo Revisa 'outputs/training_summary_scratch.txt' para el reporte.
echo.
pause
