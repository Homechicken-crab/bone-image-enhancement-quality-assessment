@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "PYTHONPATH=%PROJECT_ROOT%src"
set "VENV_PYTHON=%PROJECT_ROOT%.venv\Scripts\python.exe"

if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -m bone_iqa
    exit /b %ERRORLEVEL%
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py -3 -m bone_iqa
    exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python -m bone_iqa
    exit /b %ERRORLEVEL%
)

echo 未找到 Python 3。请安装 Python 3.11+，并先执行 pip install -e .
exit /b 1
