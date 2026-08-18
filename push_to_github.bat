@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   PUSH CARDIAC MONITOR PRO TO GITHUB
echo ========================================================
echo.

cd /d "%~dp0"

REM Add MinGit to PATH if not already present
set "PATH=%LOCALAPPDATA%\Programs\MinGit\cmd;%PATH%"

echo Remote: https://github.com/mohamednizamudeen02-web/Cardiac-Abnormality-Detection.git
echo Branch: main
echo.
echo Pushing commits to GitHub...
echo (If prompted for password, enter your GitHub Personal Access Token)
echo.

git push -u origin main

if !errorlevel! equ 0 (
    echo.
    echo ========================================================
    echo   [SUCCESS] Successfully pushed to GitHub!
    echo ========================================================
) else (
    echo.
    echo ========================================================
    echo   [NOTICE] If authentication failed, you can either:
    echo   1. Use a GitHub Personal Access Token (PAT):
    echo      Settings -> Developer settings -> Personal access tokens
    echo   2. Or run:
    echo      git push https://<YOUR_TOKEN>@github.com/mohamednizamudeen02-web/Cardiac-Abnormality-Detection.git main
    echo ========================================================
)

echo.
pause
