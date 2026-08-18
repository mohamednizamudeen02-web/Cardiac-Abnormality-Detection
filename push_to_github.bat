@echo off
setlocal enabledelayedexpansion
title Push to GitHub - Cardiac Monitor Pro

echo ========================================================
echo   CARDIAC MONITOR PRO - GITHUB REPOSITORY PUSH
echo ========================================================
echo.

cd /d "%~dp0"

REM Ensure Git and GitHub CLI are in PATH
set "PATH=%LOCALAPPDATA%\Programs\MinGit\cmd;%LOCALAPPDATA%\Programs\gh;%PATH%"

echo Checking GitHub connection...
gh auth status >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] You are not logged in to GitHub yet.
    echo Launching quick GitHub browser login...
    echo.
    gh auth login --web -h github.com -p https -w
    if !errorlevel! neq 0 (
        echo.
        echo [INFO] Web login skipped or cancelled. Proceeding with standard push...
    )
)

echo.
echo Pushing commits to branch 'main'...
echo Target: https://github.com/mohamednizamudeen02-web/Cardiac-Abnormality-Detection.git
echo.

git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ========================================================
    echo   [SUCCESS] All files and models pushed to GitHub!
    echo   URL: https://github.com/mohamednizamudeen02-web/Cardiac-Abnormality-Detection
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo   [NOTICE] If you need to authenticate using a Token:
    echo   1. Generate token at: https://github.com/settings/tokens (select 'repo')
    echo   2. Run: git push https://YOUR_TOKEN@github.com/mohamednizamudeen02-web/Cardiac-Abnormality-Detection.git main
    echo ========================================================
)

echo.
echo Press any key to exit...
pause >nul
