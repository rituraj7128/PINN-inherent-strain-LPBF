@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0..\torch_gpu;%PYTHONPATH%"
set "ENV=%USERPROFILE%\.conda\envs\notebook-env"
if exist "%ENV%\pythonw.exe" (
  start "" "%ENV%\pythonw.exe" run_bg.py run_revision3.py results\training_logs\revision3_master.log
) else (
  start "revision3" /min "%ENV%\python.exe" run_bg.py run_revision3.py results\training_logs\revision3_master.log
)
echo Started in the background. Progress: results\training_logs\revision3.log
timeout /t 5
