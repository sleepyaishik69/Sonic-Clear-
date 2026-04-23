@echo off
echo ============================================
echo NoiseClear - Audio Cleanup API Server
echo ============================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [!] Virtual environment not found!
    echo [*] Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if dependencies are installed
pip show fastapi > nul 2>&1
if errorlevel 1 (
    echo [*] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [x] Failed to install dependencies!
        pause
        exit /b 1
    )
)

echo.
echo [✓] Environment ready!
echo [*] Starting API server on http://localhost:8000
echo [*] Open http://localhost:8000 in your browser
echo [*] Press Ctrl+C to stop the server
echo.

REM Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
