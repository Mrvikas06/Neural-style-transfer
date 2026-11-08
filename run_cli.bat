@echo off
echo ========================================================
echo  Running Neural Style Transfer CLI
echo ========================================================
if exist .venv\Scripts\python.exe (
    .\.venv\Scripts\python.exe -m nst.cli %*
) else (
    python -m nst.cli %*
)
