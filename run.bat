@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"

if exist ".venv\Scripts\python.exe" goto run_venv

where py >nul 2>nul
if %ERRORLEVEL% NEQ 0 goto check_python
py -3 -c "import sys" >nul 2>nul
if %ERRORLEVEL% EQU 0 goto run_py

:check_python
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 goto run_python

echo ERROR: Python 3 was not found.
echo Install Python 3.11 or newer, NumPy, and Pillow.
set "EXIT_CODE=9009"
goto failed

:run_venv
".venv\Scripts\python.exe" -m bone_iqa
goto check_exit

:run_py
py -3 -m bone_iqa
goto check_exit

:run_python
python -m bone_iqa
goto check_exit

:check_exit
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" exit /b 0

:failed
echo.
echo ERROR: Bone IQA failed to start or exited with an error.
echo Exit code: %EXIT_CODE%
echo Run this command in PowerShell for more details:
echo   $env:PYTHONPATH="$PWD\src"; python -m bone_iqa
echo.
pause
exit /b %EXIT_CODE%
