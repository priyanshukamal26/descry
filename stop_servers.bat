@echo off
set BACKEND_DIR=C:\Projects\descry\descry\backend
set FRONTEND_DIR=C:\Projects\descry\descry\frontend
echo Stopping processes on port 8000 (Backend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing PID %%a
    taskkill /F /PID %%a
)

echo Stopping processes on port 5173 (Frontend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING') do (
    echo Killing PID %%a
    taskkill /F /PID %%a
)

echo.
echo Servers have been stopped.
pause
