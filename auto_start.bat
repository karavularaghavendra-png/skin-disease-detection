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

REM ── Configuration ───────────────────────────────────────────
set APP_PORT=8000

REM ── Step 1: Kill any leftover processes ─────────────────────
echo [1/5] Cleaning up old processes...
taskkill /F /IM ngrok.exe >nul 2>&1

REM Also kill any Python processes on our port
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":%APP_PORT%" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
)

REM Give a moment for port to free up
timeout /t 2 /nobreak >nul

REM ── Step 2: Check ngrok installation and authtoken ──────────
echo [2/5] Checking ngrok setup...

set NGROK_CMD=
set NGROK_AVAILABLE=0

REM Check if ngrok is in PATH
where ngrok >nul 2>&1
if %errorlevel%==0 (
    set NGROK_CMD=ngrok
    set NGROK_AVAILABLE=1
    goto ngrok_found
)

REM Check common locations where ngrok might be
if exist "%LOCALAPPDATA%\ngrok\ngrok.exe" (
    set "NGROK_CMD=%LOCALAPPDATA%\ngrok\ngrok.exe"
    set NGROK_AVAILABLE=1
    goto ngrok_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\Scripts\ngrok.exe" (
    set "NGROK_CMD=%LOCALAPPDATA%\Programs\Python\Python310\Scripts\ngrok.exe"
    set NGROK_AVAILABLE=1
    goto ngrok_found
)
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\ngrok.exe" (
    set "NGROK_CMD=%LOCALAPPDATA%\Microsoft\WindowsApps\ngrok.exe"
    set NGROK_AVAILABLE=1
    goto ngrok_found
)
if exist "%USERPROFILE%\ngrok\ngrok.exe" (
    set "NGROK_CMD=%USERPROFILE%\ngrok\ngrok.exe"
    set NGROK_AVAILABLE=1
    goto ngrok_found
)
if exist "%~dp0ngrok.exe" (
    set "NGROK_CMD=%~dp0ngrok.exe"
    set NGROK_AVAILABLE=1
    goto ngrok_found
)
if exist "%USERPROFILE%\Downloads\ngrok.exe" (
    set "NGROK_CMD=%USERPROFILE%\Downloads\ngrok.exe"
    set NGROK_AVAILABLE=1
    goto ngrok_found
)

REM ── ngrok not found — try to install via winget/choco ────────
echo   ngrok not found. Attempting auto-install...

REM Try winget first (built-in on Win 11 and newer Win 10)
where winget >nul 2>&1
if %errorlevel%==0 (
    echo   Installing ngrok via winget...
    winget install --id ngrok.ngrok -e --accept-source-agreements --accept-package-agreements >nul 2>&1
    where ngrok >nul 2>&1
    if %errorlevel%==0 (
        set NGROK_CMD=ngrok
        set NGROK_AVAILABLE=1
        echo   [OK] ngrok installed via winget!
        goto ngrok_found
    )
)

REM Try chocolatey
where choco >nul 2>&1
if %errorlevel%==0 (
    echo   Installing ngrok via chocolatey...
    choco install ngrok -y >nul 2>&1
    where ngrok >nul 2>&1
    if %errorlevel%==0 (
        set NGROK_CMD=ngrok
        set NGROK_AVAILABLE=1
        echo   [OK] ngrok installed via chocolatey!
        goto ngrok_found
    )
)

REM ── Auto-install failed — give manual instructions ──────────
echo.
echo   [ERROR] ngrok is NOT installed and auto-install failed.
echo.
echo   To fix this (pick ONE method):
echo.
echo   METHOD 1 - winget (recommended):
echo     winget install ngrok
echo.
echo   METHOD 2 - Manual download:
echo     1. Go to https://ngrok.com/download
echo     2. Download ngrok for Windows
echo     3. Extract ngrok.exe to this project folder:
echo        %~dp0
echo     4. Re-run this script.
echo.
echo   Continuing WITHOUT ngrok (local access only)...
echo.
goto start_server

:ngrok_found
echo   [OK] ngrok found: %NGROK_CMD%

REM ── Auto-upgrade config from v2 to v3 if needed ────────────
REM ngrok v3 requires config version 3. Old configs cause silent failures.
"%NGROK_CMD%" config upgrade >nul 2>&1

REM ── Check if authtoken is configured ────────────────────────
set NGROK_CONFIG_FOUND=0

REM Check all known config locations
if exist "%APPDATA%\ngrok\ngrok.yml" set NGROK_CONFIG_FOUND=1
if exist "%LOCALAPPDATA%\ngrok\ngrok.yml" set NGROK_CONFIG_FOUND=1
if exist "%USERPROFILE%\.ngrok2\ngrok.yml" set NGROK_CONFIG_FOUND=1

REM Double-check: even if config file exists, verify it has an authtoken
if %NGROK_CONFIG_FOUND%==1 (
    REM Config file found — assume authtoken is set
    echo   [OK] ngrok authtoken configured.
    goto start_server
)

echo.
echo   [WARNING] ngrok authtoken is NOT configured!
echo.
echo   ══════════════════════════════════════════════
echo   HOW TO FIX (one-time setup, takes 2 minutes):
echo   ══════════════════════════════════════════════
echo.
echo     1. Sign up for free at:
echo        https://dashboard.ngrok.com/signup
echo.
echo     2. Copy your authtoken from:
echo        https://dashboard.ngrok.com/get-started/your-authtoken
echo.
echo     3. Open a terminal and run:
echo        %NGROK_CMD% config add-authtoken YOUR_TOKEN_HERE
echo.
echo     4. Re-run this script.
echo   ══════════════════════════════════════════════
echo.
echo   Continuing WITHOUT ngrok (local access only)...
echo.
set NGROK_AVAILABLE=0

:start_server

REM ── Step 3: Start FastAPI server (minimized) ────────────────
echo [3/5] Starting FastAPI server on port %APP_PORT%...
start "SkinDisease-Server" /min cmd /c "cd /d %~dp0 && call .venv\Scripts\activate.bat && python -m uvicorn api:app --host 0.0.0.0 --port %APP_PORT%"

REM ── Step 4: Wait for server to be ready ─────────────────────
echo [4/5] Waiting for server to start...
set RETRIES=0
:wait_loop
timeout /t 3 /nobreak >nul
set /a RETRIES+=1

REM Use PowerShell for reliable HTTP check (curl may not exist)
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:%APP_PORT%/health' -TimeoutSec 3; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
    echo       Server is READY!
    goto server_ready
)
if %RETRIES% GEQ 40 (
    echo       [ERROR] Server did not start after 2 minutes.
    echo       Check if Python, dependencies, and model are installed.
    echo.
    echo       Try running manually:
    echo         cd /d %~dp0
    echo         .venv\Scripts\activate
    echo         python -m uvicorn api:app --host 0.0.0.0 --port %APP_PORT%
    pause
    exit /b 1
)
echo       Attempt %RETRIES%/40 - waiting...
goto wait_loop

:server_ready

REM ── Step 5: Start ngrok tunnel (if available) ───────────────
if %NGROK_AVAILABLE%==0 goto no_ngrok

echo [5/5] Starting ngrok tunnel...

REM Kill any lingering ngrok processes one more time
taskkill /F /IM ngrok.exe >nul 2>&1
timeout /t 1 /nobreak >nul

REM Start ngrok tunnel (forwards to local FastAPI server)
start "SkinDisease-Ngrok" /min cmd /c "%NGROK_CMD% http %APP_PORT%"

REM ── Wait for ngrok tunnel to establish ──────────────────────
echo       Waiting for ngrok tunnel to establish...
set NGROK_RETRIES=0
:ngrok_wait
timeout /t 2 /nobreak >nul
set /a NGROK_RETRIES+=1

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:4040/api/tunnels' -TimeoutSec 3; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 goto ngrok_success

if %NGROK_RETRIES% GEQ 10 (
    echo.
    echo   [WARNING] ngrok tunnel failed to start after 20 seconds!
    echo.
    echo   Common reasons:
    echo     - Authtoken expired or invalid
    echo     - Another ngrok session running on another machine
    echo     - Free plan allows only 1 tunnel at a time
    echo     - Firewall blocking ngrok
    echo.
    echo   To fix:
    echo     1. Kill all ngrok: taskkill /F /IM ngrok.exe
    echo     2. Get a fresh authtoken from:
    echo        https://dashboard.ngrok.com/get-started/your-authtoken
    echo     3. Run:  %NGROK_CMD% config add-authtoken YOUR_NEW_TOKEN
    echo     4. Re-run this script.
    echo.
    echo   Falling back to local access only.
    goto no_ngrok
)
echo       Attempt %NGROK_RETRIES%/10 - waiting for tunnel...
goto ngrok_wait

:ngrok_success

REM ── Display the public URL ──────────────────────────────────
echo.
echo ============================================
echo   ALL SYSTEMS ONLINE
echo ============================================
echo.
echo   Local:  http://localhost:%APP_PORT%
echo.

REM Fetch and display the ngrok URL
powershell -NoProfile -Command "try { $r = (Invoke-WebRequest -UseBasicParsing http://localhost:4040/api/tunnels).Content | ConvertFrom-Json; $url = $r.tunnels[0].public_url; Write-Host '  Public: ' $url; Write-Host ''; Write-Host '  Share this link with anyone to access your app!' } catch { Write-Host '  Check ngrok dashboard at http://localhost:4040' }"

echo.
echo   ngrok Dashboard: http://localhost:4040
echo.
echo   This window can be minimized. Do NOT close it.
echo   To stop everything, close this window.
echo ============================================
echo.

goto done

:no_ngrok
echo.
echo ============================================
echo   SERVER ONLINE (local access only)
echo ============================================
echo.
echo   Local:  http://localhost:%APP_PORT%
echo.

REM Show network IP for LAN access
for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' } | Select-Object -First 1).IPAddress"`) do (
    echo   LAN:    http://%%i:%APP_PORT%
    echo   ^(Share this link with anyone on the same Wi-Fi^)
)

echo.
echo   No public URL available. To enable public access:
echo     1. Install ngrok: winget install ngrok
echo     2. Sign up at https://dashboard.ngrok.com/signup
echo     3. Run:  ngrok config add-authtoken YOUR_TOKEN
echo     4. Re-run this script.
echo.
echo   This window can be minimized. Do NOT close it.
echo   To stop everything, close this window.
echo ============================================
echo.

:done
REM Open the browser automatically
start "" "http://localhost:%APP_PORT%"

REM Keep this window alive so processes stay running
pause >nul
