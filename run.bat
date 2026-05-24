@echo off
color 0B
echo ==========================================
echo   Skin Disease Detection App
echo   Starting Server...
echo ==========================================
echo.

cd /d "%~dp0"

echo   Project: %cd%
echo.
echo   App will open at: http://localhost:8080/static/index.html
echo   Press Ctrl+C to stop the server.
echo.

REM Open browser automatically after 5 seconds
start "" timeout /t 5 /nobreak >nul & start "" http://localhost:8080/static/index.html#predict

REM Use python -m uvicorn (works even when uvicorn is not on PATH)
python -m uvicorn api:app --reload --port 8080

pause
