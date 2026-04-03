@echo off
title Katraswing - Swing Trade Analyzer
echo.
echo  ==========================================
echo   KATRASWING - Swing Trade Analyzer
echo  ==========================================
echo.
echo  Starting app... browser will open shortly.
echo  Press Ctrl+C to stop.
echo.
"C:\Program Files\Python312\python.exe" -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
