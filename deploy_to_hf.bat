@echo off
REM ============================================================
REM   Skin Disease Detection — Deploy to Hugging Face Spaces
REM   Run this once to push your project to the cloud.
REM   After deploying, your app is ALWAYS ONLINE even when
REM   your PC is off.
REM ============================================================

title Deploy to Hugging Face Spaces
color 0B

echo.
echo ============================================
echo   Deploy to Hugging Face Spaces (FREE)
echo ============================================
echo.
echo   Your app will be permanently available at:
echo   https://huggingface.co/spaces/USERNAME/skin-disease-detection
echo.
echo   Prerequisites:
echo     1. A free account at https://huggingface.co
echo     2. Git installed (https://git-scm.com)
echo.

REM ── Step 0: Verify git is installed ─────────────────────────────────────────
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed!
    echo.
    echo   Please install Git from: https://git-scm.com/download/win
    echo   Then re-run this script.
    pause
    exit /b 1
)
echo [OK] Git found.

REM ── Step 1: Get HF username ──────────────────────────────────────────────────
echo.
set /p HF_USERNAME="Enter your Hugging Face username: "
if "%HF_USERNAME%"=="" (
    echo [ERROR] Username cannot be empty.
    pause
    exit /b 1
)

echo.
echo   Using username: %HF_USERNAME%
echo   Space name: skin-disease-detection
echo.

REM ── Step 2: Copy HF Dockerfile to main Dockerfile ───────────────────────────
echo [1/5] Preparing HF Spaces configuration...
copy /Y Dockerfile.hf Dockerfile >nul
echo       Done. HF Spaces Dockerfile is ready.

REM ── Step 3: Copy HF README to main README ───────────────────────────────────
echo [2/5] Preparing README for HF Spaces...
copy /Y README_HF.md README.md >nul
echo       Done.

REM ── Step 4: Set up git and push ─────────────────────────────────────────────
echo [3/5] Configuring git...
git add -A
git commit -m "Deploy: Hugging Face Spaces configuration" --allow-empty

echo [4/5] Adding Hugging Face remote...
git remote remove hf-spaces >nul 2>&1
git remote add hf-spaces https://huggingface.co/spaces/%HF_USERNAME%/skin-disease-detection

echo [5/5] Pushing to Hugging Face Spaces...
echo.
echo   *** You will be prompted for your HF credentials ***
echo   Username: %HF_USERNAME%
echo   Password: Your HF Access Token (get it from https://huggingface.co/settings/tokens)
echo.
git push hf-spaces main

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Push failed. Please check:
    echo   1. Your HF username is correct: %HF_USERNAME%
    echo   2. You have created the Space at:
    echo      https://huggingface.co/new-space
    echo      - Space name: skin-disease-detection
    echo      - SDK: Docker
    echo      - Visibility: Public
    echo   3. Your Access Token has write permissions
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   SUCCESS! Deployment Complete!
echo ============================================
echo.
echo   Your app is now LIVE at:
echo   https://%HF_USERNAME%-skin-disease-detection.hf.space
echo.
echo   It will take 2-5 minutes to build on the server.
echo   Watch the build logs at:
echo   https://huggingface.co/spaces/%HF_USERNAME%/skin-disease-detection
echo.
echo   IMPORTANT: Set these secrets in HF Spaces Settings:
echo     API_KEY = your-secret-key
echo     CORS_ORIGINS = *
echo.
echo ============================================
echo.
pause
