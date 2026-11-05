@echo off
title Rathinam AI Assistant - Consolidated Backend
cd /d "%~dp0"
echo Starting Rathinam AI Assistant Central Backend on port 8000...
".\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000

