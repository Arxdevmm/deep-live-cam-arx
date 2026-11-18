@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Virtual environment is missing. Run install.bat first.
    pause
    exit /b 1
)

"venv\Scripts\python.exe" "run.py" %*
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Deep-Live-Cam exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
