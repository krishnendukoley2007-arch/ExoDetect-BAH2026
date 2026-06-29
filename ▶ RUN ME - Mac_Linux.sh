#!/bin/bash
clear
echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║        EXODETECT v8.0  —  BAH2026            ║"
echo "  ║  AI Exoplanet Detection from TESS Data       ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    echo "  [ERROR] Python3 not found."
    echo "  Install from: https://www.python.org/downloads/"
    exit 1
fi

echo "  [OK] $(python3 --version)"
echo ""
echo "  [1/2] Checking dependencies..."

if ! python3 -c "import streamlit" &>/dev/null; then
    echo "  Installing packages - takes 2-3 minutes first time..."
    pip3 install -r requirements.txt --quiet
fi

echo "  [OK] Dependencies ready."
echo ""
echo "  [2/2] Starting ExoDetect Dashboard..."
echo "  Browser opens at: http://localhost:8501"
echo "  To stop: press Ctrl+C"
echo ""

(sleep 3 && \
    if command -v open &>/dev/null; then open "http://localhost:8501"
    elif command -v xdg-open &>/dev/null; then xdg-open "http://localhost:8501"
    fi) &

python3 -m streamlit run dashboard.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
