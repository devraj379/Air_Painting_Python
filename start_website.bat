@echo off
setlocal
cd /d "%~dp0"

set "PY_EXE="
where python >nul 2>nul && set "PY_EXE=python"
if "%PY_EXE%"=="" if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PY_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
)

if "%PY_EXE%"=="" (
    echo Could not find Python executable. Please install Python 3.12 or add python to PATH.
    pause
    exit /b 1
)

echo Starting Air Painting Studio at http://127.0.0.1:8000
start "" http://127.0.0.1:8000
"%PY_EXE%" server.py
