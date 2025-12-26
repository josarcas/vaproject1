@echo off
set "IMAGE_PATH=%~1"

if "%IMAGE_PATH%"=="" (
    echo Error: Debes arrastrar una imagen sobre este archivo o escribir su ruta.
    echo Uso: test_image.bat camino/a/tu/foto.jpg
    pause
    exit /b 1
)

echo Probando con imagen: %IMAGE_PATH%
echo Usando modelo: outputs/best_scratch.pt (Si falla, asegura que entrenaste primero)

python predict.py --checkpoint outputs/best_scratch.pt --haar --images "%IMAGE_PATH%" --save-dir predictions

echo.
echo Revisa la carpeta 'predictions' para ver la imagen con el resultado.
pause
