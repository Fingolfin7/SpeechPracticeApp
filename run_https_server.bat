@echo off
setlocal
cd /d "%~dp0"
echo SpeechPractice uses localhost, which supports microphone access without HTTPS.
echo Starting the local app...
echo.
call "%~dp0run_server.bat"
