@echo off
echo ========================================================
echo  Starting Neural Style Transfer Pro Studio (Port 8000)
echo ========================================================
if exist .venv\Scripts\python.exe (
    start http://localhost:8000
    .\.venv\Scripts\python.exe -m app.server
) else (
    start http://localhost:8000
    python -m app.server
)
pause
