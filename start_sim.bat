@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe app.py --sim
) else (
    python app.py --sim
)
if errorlevel 1 pause
