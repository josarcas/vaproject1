@echo off
set "IMAGE_PATH=%~1"

if "%IMAGE_PATH%"=="" (
    echo Error: You must drag an image over this file or write its path.
    echo Usage: test_image.bat path/to/your/photo.jpg
    pause
    exit /b 1
)       

echo Testing with image: %IMAGE_PATH%
echo Using model: outputs/best_scratch.pt (If it fails, make sure you trained first)

python predict.py --checkpoint outputs/best_scratch.pt --haar --images "%IMAGE_PATH%" --save-dir predictions

echo.
echo Check the 'predictions' folder for the image with the result.
pause
