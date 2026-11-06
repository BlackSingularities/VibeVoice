@echo off
rem VibeVoice Studio - uruchamia serwer w osobnym oknie, niezaleznie od innych aplikacji.
rem Gdy serwer padnie, uruchamia go ponownie (zadania z kolejki wznawiaja sie same).
cd /d "%~dp0"
title VibeVoice Studio
:loop
echo [%date% %time%] Start serwera na http://localhost:3000
".venv\Scripts\python.exe" demo\vibevoice_realtime_demo.py --model_path microsoft/VibeVoice-Realtime-0.5B --port 3000
echo [%date% %time%] Serwer zakonczyl dzialanie (kod %errorlevel%). Restart za 5 s... (Ctrl+C, aby przerwac)
timeout /t 5 /nobreak >nul
goto loop
