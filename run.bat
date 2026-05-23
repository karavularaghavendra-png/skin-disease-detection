@echo off
color 0B
echo ==========================================
echo   Skin Disease Detection App
echo   Starting Streamlit Server...
echo ==========================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run 'setup.bat' first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo   App will open at: http://localhost:8501
echo   Press Ctrl+C to stop the server.
echo.
streamlit run app.py
pause
