@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   ADVANCED CARDIAC MONITORING SYSTEM
echo ========================================
echo.
echo Starting advanced version with:
echo   - Real-time WebSocket streaming
echo   - Explainable AI (Grad-CAM, SHAP)
echo   - PDF report generation
echo   - Multi-patient monitoring
echo.
echo ========================================
echo.

cd /d "%~dp0\.."

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
echo Starting advanced server...
echo Navigate to: http://localhost:5000
echo.

!PYTHON_EXE! app\main_advanced.py

if !errorlevel! neq 0 (
    echo.
    echo Server exited with error code !errorlevel!
    pause
)
