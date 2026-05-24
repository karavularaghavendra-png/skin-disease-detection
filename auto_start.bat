@echo off
REM ============================================================
REM   Skin Disease Detection — Auto-Start Script
REM   Launches FastAPI server + ngrok tunnel automatically.
REM   Place a shortcut to this file in your Startup folder.
REM ============================================================

title Skin Disease Detection - Auto Starter
cd /d "%~dp0"

echo.
echo ============================================
echo   Skin Disease Detection - Auto Starter
echo ============================================
echo.

REM ── Step 1: Kill any leftover processes ─────────────────────
echo [1/4] Cleaning up old processes...
taskkill /F /IM ngrok.exe >nul 2>&1
REM Give a moment for port to free up
timeout /t 2 /nobreak >nul

REM ── Step 2: Start FastAPI server (minimized) ────────────────
echo [2/4] Starting FastAPI server on port 8080...
start "SkinDisease-Server" /min cmd /c "cd /d %~dp0 && call .venv\Scripts\activate.bat && python -m uvicorn api:app --reload --host 0.0.0.0 --port 8080"

REM ── Step 3: Wait for server to be ready ─────────────────────
echo [3/4] Waiting for server to start...
set RETRIES=0
:wait_loop
timeout /t 2 /nobreak >nul
set /a RETRIES+=1
curl -s http://localhost:8080/health >nul 2>&1
if %errorlevel%==0 (
    echo       Server is READY!
    goto server_ready
)
if %RETRIES% GEQ 30 (
    echo       [ERROR] Server did not start after 60 seconds.
    echo       Check if Python and dependencies are installed.
    pause
    exit /b 1
)
echo       Attempt %RETRIES%/30 - waiting...
goto wait_loop

:server_ready

REM ── Step 4: Start ngrok tunnel (minimized) ──────────────────
echo [4/4] Starting ngrok tunnel...
start "SkinDisease-Ngrok" /min cmd /c "ngrok http 8080"

timeout /t 3 /nobreak >nul

REM ── Step 5: Display the public URL ──────────────────────────
echo.
echo ============================================
echo   ALL SYSTEMS ONLINE
echo ============================================
echo.
echo   Local:  http://localhost:8080
echo.

REM Try to fetch and display the ngrok URL
for /f "tokens=*" %%a in ('powershell -Command "(Invoke-WebRequest -UseBasicParsing http://localhost:4040/api/tunnels).Content" 2^>nul') do set NGROK_RESPONSE=%%a
if defined NGROK_RESPONSE (
    for /f "tokens=2 delims=," %%u in ('echo %NGROK_RESPONSE% ^| powershell -Command "$input | ConvertFrom-Json | Select-Object -ExpandProperty tunnels | Select-Object -First 1 -ExpandProperty public_url"') do echo   Public: %%u
    powershell -Command "try { $r = (Invoke-WebRequest -UseBasicParsing http://localhost:4040/api/tunnels).Content | ConvertFrom-Json; Write-Host '  Public: ' $r.tunnels[0].public_url } catch { Write-Host '  Check ngrok dashboard at http://localhost:4040' }"
)

echo.
echo   This window can be minimized. Do NOT close it.
echo   To stop everything, close this window.
echo ============================================
echo.

REM Keep this window alive so processes stay running
pause >nul
