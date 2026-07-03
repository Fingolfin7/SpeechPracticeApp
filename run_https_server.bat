@echo off
setlocal EnableExtensions

REM Start the SpeechPractice Django dev server over HTTPS for phone/LAN recording.
cd /d "%~dp0"

set "HOST=0.0.0.0"
set "PORT=8443"
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
echo Starting SpeechPractice HTTPS server for browser recording on LAN...
echo.
echo Using: %PYTHON%
echo.

"%PYTHON%" -m speechpractice_web.dev_https_server --host %HOST% --port %PORT%
