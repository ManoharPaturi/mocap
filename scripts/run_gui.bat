@echo off
echo Starting MoCap VS5...
cd /d "%~dp0.."
venv\Scripts\python.exe main_gui.py
pause
