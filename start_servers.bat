@echo off
echo Starting Descry Backend (FastAPI)...
start /B cmd /c "cd backend && ..\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"

echo Starting Descry Frontend (Vite)...
start /B cmd /c "cd frontend && npx vite --port 5173"

set BACKEND_DIR=C:\Projects\descry\descry\backend
set FRONTEND_DIR=C:\Projects\descry\descry\frontend

echo.
echo Servers are running in the background.
echo Backend Health: http://localhost:8000/health
echo Frontend UI:     http://localhost:5173
echo.
pause
