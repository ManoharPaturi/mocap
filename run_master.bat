@echo off
echo Starting MoCap VS5 MASTER MODE...
echo Remote Camera: 10.137.227.217
venv\Scripts\python.exe launch_multi_camera.py --mode master --remote-ip 10.137.227.217
pause
