@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Start the SpeechPractice Django dev server for this PC and the local network.
cd /d "%~dp0"

set "HOST=0.0.0.0"
set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

set "DJANGO_DEBUG=1"
set "DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0,*"

if exist ".venv312\Scripts\python.exe" (
    set "PYTHON=.venv312\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

echo.
echo Starting SpeechPractice Django server...
echo.
echo Local:
echo   http://127.0.0.1:%PORT%/
echo.
echo Local network:
for /f "tokens=2 delims=:" %%A in ('ipconfig ^| findstr /R /C:"IPv4.*"') do (
    set "IP=%%A"
    set "IP=!IP: =!"
    echo   http://!IP!:%PORT%/
)
echo.
echo Using: %PYTHON%
echo Press Ctrl+C to stop the server.
echo.

"%PYTHON%" manage.py runserver %HOST%:%PORT%
