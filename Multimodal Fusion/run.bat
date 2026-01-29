@echo off
echo ========================================
echo Multimodal Emotion Recognition - Run
echo ========================================
echo.

REM Activate venv
if not exist venv (
    echo Virtual environment not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM Run training
python main.py --config configs/eav.yaml

pause
