@echo off
echo ==================================================
echo    COMPARE MODELS
echo ==================================================
echo.

echo [1/2] Training Model 1: ResNet18...
python train.py --train-dir datos_zip/dataset_emociones/train --val-dir datos_zip/dataset_emociones/validation --model resnet18 --output-dir outputs_resnet --batch-size 32
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ResNet18 Completed. Results in 'outputs_resnet/'
echo.

echo [2/2] Training Model 2: MobileNetV3 Small...
python train.py --train-dir datos_zip/dataset_emociones/train --val-dir datos_zip/dataset_emociones/validation --model mobilenet_v3_small --output-dir outputs_mobilenet --batch-size 32
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo ==================================================
echo    OUTPUT
echo ==================================================
echo Revisa y compara los siguientes reportes:
echo 1. outputs_resnet\training_summary_resnet18.txt
echo 2. outputs_mobilenet\training_summary_mobilenet_v3_small.txt
echo.
pause
