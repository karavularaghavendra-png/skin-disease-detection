@echo off
color 0B
echo ==========================================
echo   Skin Disease Detection App
echo   Starting FastAPI Server...
echo ==========================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    REM Try system Python if no venv
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Python not found! Please run 'setup.bat' first.
        pause
        exit /b 1
    )
    echo   Using system Python...
    echo.
    echo   App will open at: http://localhost:8080/static/index.html
    echo   Press Ctrl+C to stop the server.
    echo.
    uvicorn api:app --reload --port 8080
) else (
    call .venv\Scripts\activate.bat
    echo   App will open at: http://localhost:8080/static/index.html
    echo   Press Ctrl+C to stop the server.
    echo.
    uvicorn api:app --reload --port 8080
)
pause
