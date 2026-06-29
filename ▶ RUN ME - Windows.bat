@echo off
title ExoDetect v8.0 - BAH2026 PS7 Launcher
color 0B

echo.
echo  ██████████████████████████████████████████████
echo  ██        EXODETECT v8.0  - BAH2026         ██
echo  ██   AI Exoplanet Detection from TESS Data  ██
echo  ██████████████████████████████████████████████
echo.

python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo  [ERROR] Python not found!
    echo  Install Python 3.10+ from: https://www.python.org/downloads/
    echo  Check "Add Python to PATH" during install.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

echo  [OK] Python found.
echo.
echo  [1/2] Checking dependencies...

pip show streamlit >nul 2>&1
IF ERRORLEVEL 1 (
    echo  Installing packages - takes 2-3 minutes first time...
    pip install -r requirements.txt --quiet
)

echo  [OK] Dependencies ready.
echo.
echo  [2/2] Starting ExoDetect Dashboard...
echo  Browser opens at: http://localhost:8501
echo  To stop: press Ctrl+C
echo.

start "" "http://localhost:8501"
python -m streamlit run dashboard.py --server.port 8501 --browser.gatherUsageStats false

pause
