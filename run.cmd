@echo off
setlocal
cd /d "%~dp0"
set "MEDENERGY_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%MEDENERGY_PY%" goto bundled
py -3 --version >nul 2>&1
if not errorlevel 1 goto launcher
python --version >nul 2>&1
if not errorlevel 1 goto standard
echo Python was not found. You can still double-click dashboard.html to use the ready-made demo.
echo To regenerate the demo, install Python 3.10 or newer and then run this file again.
pause
exit /b 1
:bundled
"%MEDENERGY_PY%" medenergy.py
goto finish
:launcher
py -3 medenergy.py
goto finish
:standard
python medenergy.py
:finish
if errorlevel 1 goto failed
start "" "%~dp0dashboard.html"
echo Done. The dashboard is open. Your data and results are saved in this folder.
pause
exit /b 0
:failed
echo Generation failed. Please copy the error shown above.
pause
exit /b 1
