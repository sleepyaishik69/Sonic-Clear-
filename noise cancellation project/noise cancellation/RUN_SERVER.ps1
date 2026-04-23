# NoiseClear - Audio Cleanup API Server (PowerShell)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "NoiseClear - Audio Cleanup API Server" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\activate.ps1")) {
    Write-Host "[!] Virtual environment not found! Creating..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
& ".venv\Scripts\activate.ps1"

# Check if dependencies are installed
try {
    pip show fastapi > $null 2>&1
    $depsInstalled = $?
} catch {
    $depsInstalled = $false
}

if (-not $depsInstalled) {
    Write-Host "[*] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[x] Failed to install dependencies!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "[✓] Environment ready!" -ForegroundColor Green
Write-Host "[*] Starting API server on http://localhost:8000" -ForegroundColor Cyan
Write-Host "[*] Open http://localhost:8000 in your browser" -ForegroundColor Cyan
Write-Host "[*] Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

Read-Host "Press Enter to exit"
