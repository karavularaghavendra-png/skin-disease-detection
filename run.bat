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

REM Automatically detect local network IP address using PowerShell
set LOCAL_IP=localhost
for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Select-Object -First 1).IPAddress"`) do set LOCAL_IP=%%i

echo   [Local Access] http://localhost:8080/static/index.html#predict
if not "%LOCAL_IP%"=="localhost" (
    echo   [Network Access] http://%LOCAL_IP%:8080/static/index.html#predict
    echo   (Share the Network Access link with anyone on the same Wi-Fi)
)
echo.
echo   Press Ctrl+C to stop the server.
echo ==========================================
echo.

REM Open browser automatically after 5 seconds
start "" timeout /t 5 /nobreak >nul & start "" http://localhost:8080/static/index.html#predict

REM Run on 0.0.0.0 so others on same Wi-Fi/LAN can access it
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8080

pause
