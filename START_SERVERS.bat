@echo off
echo ============================================================
echo Starting NarrativeNexus Platform
echo ============================================================
echo.

cd /d D:\InfosysNarrativeNexus

echo [1/2] Starting FastAPI Backend (Port 8000)...
start "FastAPI Backend" cmd /k "call venv\Scripts\activate.bat && uvicorn src.main:app --host 127.0.0.1 --port 8000"

timeout /t 10 /nobreak >nul

echo [2/2] Starting Flask UI (Port 5000)...
start "Flask UI" cmd /k "call venv\Scripts\activate.bat && python ui\app.py"

timeout /t 3 /nobreak >nul

echo.
echo ============================================================
echo SERVERS STARTED!
echo ============================================================
echo Frontend UI:  http://127.0.0.1:5000
echo Backend API:  http://127.0.0.1:8000
echo API Docs:     http://127.0.0.1:8000/docs
echo ============================================================
echo.
echo Opening browser...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5000

echo.
echo To stop servers: Close the terminal windows
echo ============================================================
pause
