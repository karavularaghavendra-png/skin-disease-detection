@echo off
color 0A
echo ==========================================
echo   Skin Disease Detection - Auto Setup
echo   Deep Learning Project (MobileNetV2)
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.10+ from https://python.org
    echo IMPORTANT: Check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

echo [1/4] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo       Virtual environment created.
) else (
    echo       Virtual environment already exists. Skipping.
)
echo.

echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat
echo       Activated.
echo.

echo [3/4] Installing dependencies (this may take 3-5 minutes)...
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install some dependencies.
    echo Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
echo.
echo       All dependencies installed successfully.
echo.

echo [4/4] Checking model file...
if exist "model\skin_model.h5" (
    echo       [OK] Model file found: model\skin_model.h5
) else (
    echo.
    echo       [WARNING] Model file NOT found!
    echo       Please place 'skin_model.h5' in the 'model' folder.
    echo       Without it, the app cannot make predictions.
    echo.
)

if exist "model\class_indices.json" (
    echo       [OK] Class indices found: model\class_indices.json
) else (
    echo       [WARNING] class_indices.json NOT found in model folder.
)

echo.
echo ==========================================
echo   Setup Complete!
echo ==========================================
echo.
echo   To run the app:
echo     Double-click 'run.bat'
echo     OR run: streamlit run app.py
echo.
echo   The app will open at:
echo     http://localhost:8501
echo.
pause
