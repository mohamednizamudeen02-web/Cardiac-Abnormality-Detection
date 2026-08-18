@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo   CARDIAC MONITOR PRO - INTELLIGENT ECG DIAGNOSIS SYSTEM
echo ========================================================
echo.

cd /d "%~dp0"

REM Locate Python
set "PYTHON_EXE="
if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    py -0 >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py"
    ) else (
        set "PYTHON_EXE=python"
    )
)

echo Using Python launcher: !PYTHON_EXE!
echo Launching server on http://localhost:5000...
echo.

!PYTHON_EXE! app\main_advanced.py

if !errorlevel! neq 0 (
    echo.
    echo Server exited with error code !errorlevel!
    pause
)
