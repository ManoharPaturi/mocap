@echo off
echo Starting MoCap Web App...

:: Start Backend in a new window (Relative to vs3 folder)
echo Starting Backend (FastAPI)...
start "MoCap Backend" cmd /k "venv\Scripts\activate && uvicorn server:app --reload --host 0.0.0.0 --port 8000"

:: Start Frontend in the current window (Relative to vs3 folder)
echo Starting Frontend (React)...
cd frontend
npm run dev
pause
