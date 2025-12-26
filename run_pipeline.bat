@echo off
echo ==================================================
echo      EXEC PIPELINE
echo ==================================================

echo.
echo [1/3] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [2/3] Download dataset...
python download_dataset.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [3/3] Training...
echo       Save in 'outputs'
python train.py --train-dir datos_zip/dataset_emociones/train --val-dir datos_zip/dataset_emociones/validation --output-dir outputs
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==================================================
echo      PIPELINE COMPLETED
echo ==================================================
echo Check 'outputs/training_summary_scratch.txt' for the report.
echo.
pause
