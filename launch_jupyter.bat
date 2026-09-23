@echo off
echo ===================================================
echo Starting Jupyter Notebook in Google Chrome...
echo ===================================================

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo Creating virtual environment .venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Checking and installing Jupyter dependencies...
pip show jupyter >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Installing Jupyter...
    pip install jupyter
)

echo Opening Jupyter Notebook directly in Chrome...
set "CHROME_BIN=C:\Program Files\Google\Chrome\Application\chrome.exe"
IF NOT EXIST "%CHROME_BIN%" (
    set "CHROME_BIN=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
)

jupyter notebook --browser="%CHROME_BIN% %%s"

pause
