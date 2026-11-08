@echo off
echo ========================================================
echo  Starting Neural Style Transfer Web Studio
echo ========================================================
if exist .venv\Scripts\streamlit.exe (
    .\.venv\Scripts\streamlit.exe run app/streamlit_app.py
) else (
    streamlit run app/streamlit_app.py
)
pause
