@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv312\Scripts\python.exe" (
    echo Creating a local Python environment...
    py -3.12 -m venv .venv312
    if errorlevel 1 (
        echo Could not create the environment. Install Python 3.12 and run this file again.
        pause
        exit /b 1
    )
)
set "PYTHON=.venv312\Scripts\python.exe"

"%PYTHON%" check_studio_dependencies.py >nul 2>&1
if errorlevel 1 (
    echo Installing SpeechPractice dependencies...
    "%PYTHON%" -m pip install -r requirements-studio.txt
    if errorlevel 1 (
        echo Dependencies could not be installed. Check your internet connection and run this file again.
        pause
        exit /b 1
    )
    "%PYTHON%" check_studio_dependencies.py >nul 2>&1
    if errorlevel 1 (
        echo The installed dependencies do not meet requirements-studio.txt. Review the installation output and try again.
        pause
        exit /b 1
    )
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo ffmpeg was not found on your PATH. Install ffmpeg, then run this file again.
    pause
    exit /b 1
)

echo.
echo SpeechPractice is starting on this computer only.
echo Open http://127.0.0.1:8765/ in your browser.
echo Keep this window open while using the app; press Ctrl+C to stop it.
echo.
"%PYTHON%" manage_studio.py runserver 127.0.0.1:8765 --noreload --insecure
if errorlevel 1 (
    echo.
    echo The server could not start. If port 8765 is in use, close the other app and try again.
    pause
    exit /b 1
)
