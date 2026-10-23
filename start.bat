@echo off
title OmniDetect AI - Vision Studio
cd /d "%~dp0"
echo ========================================================
echo   OmniDetect AI - Unified Object Detector Studio
echo   Launching on http://localhost:8000 ...
echo ========================================================
python -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
