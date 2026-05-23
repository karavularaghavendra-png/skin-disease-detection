@echo off
color 0E
echo ==========================================
echo   Creating Portable ZIP Package
echo ==========================================
echo.

REM Set the output filename
set ZIP_NAME=Skin_Disease_Detection_Project.zip

REM Delete old ZIP if exists
if exist "%ZIP_NAME%" del "%ZIP_NAME%"

echo Creating ZIP file: %ZIP_NAME%
echo.
echo Including:
echo   - All Python source files
echo   - Model files (skin_model.h5, best_model.h5)
echo   - Utils, evaluation, static folders
echo   - Setup and run scripts
echo   - Documentation
echo.

REM Use PowerShell to create the ZIP (available on all modern Windows)
powershell -Command ^
  "Compress-Archive -Path @( ^
    'app.py', ^
    'api.py', ^
    'predict.py', ^
    'preprocess.py', ^
    'train_model.py', ^
    'dataset_loader.py', ^
    'explainability.py', ^
    'logger.py', ^
    'utils', ^
    'model', ^
    'evaluation', ^
    'static', ^
    'tests', ^
    'tools', ^
    'requirements.txt', ^
    'README.md', ^
    'TROUBLESHOOTING.md', ^
    'CONTRIBUTING.md', ^
    'CHANGELOG.md', ^
    'LICENSE', ^
    'Dockerfile', ^
    'setup.bat', ^
    'run.bat', ^
    'test_skin_sample.jpg' ^
  ) -DestinationPath '%ZIP_NAME%' -CompressionLevel Optimal -Force"

if %errorlevel% equ 0 (
    echo.
    echo ==========================================
    echo   ZIP Created Successfully!
    echo ==========================================
    echo   File: %ZIP_NAME%
    echo.
    echo   Share this via:
    echo     - Google Drive
    echo     - WhatsApp
    echo     - Pen Drive
    echo     - Email (if under 25 MB)
    echo.
    echo   The recipient should:
    echo     1. Extract the ZIP
    echo     2. Double-click setup.bat
    echo     3. Double-click run.bat
    echo.
) else (
    echo.
    echo [ERROR] Failed to create ZIP file.
    echo Try creating it manually using File Explorer.
)

pause
