@echo off
echo Starting MoCap VS5 MASTER MODE...
cd /d "%~dp0.."
set /p REMOTE_IP="Enter remote PC IP (e.g. 10.137.227.217): "
echo Remote Camera: %REMOTE_IP%
venv\Scripts\python.exe launch_multi_camera.py --mode master --remote-ip %REMOTE_IP%
pause
