# Cardiac Abnormality Detection - Startup Script
# Run this script to start the web application

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cardiac Abnormality Detection System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$PythonExe = "py"
if (Test-Path ".\venv\Scripts\python.exe") {
    $PythonExe = ".\venv\Scripts\python.exe"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
}

Write-Host "Using Python: $PythonExe" -ForegroundColor Yellow
Write-Host "Starting web application..." -ForegroundColor Yellow
Write-Host "Navigate to: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

& $PythonExe app/main_advanced.py
