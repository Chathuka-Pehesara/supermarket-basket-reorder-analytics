@echo off
echo ========================================================
echo Launching Supermarket Merchandising Portal (Streamlit)
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
