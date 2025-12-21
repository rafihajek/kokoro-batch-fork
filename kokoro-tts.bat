@echo off
cd /d "C:\Users\alias\Documents\Random Shit\Software\Kokoro-TTS-Local"
call venv\Scripts\activate

REM Open browser after short delay (gives server time to start)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:7860"

python gradio_interface.py
pause
