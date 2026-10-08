@echo off
echo ===================================================
echo Starting Jupyter in Google Chrome...
echo ===================================================

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo Creating virtual environment .venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

IF NOT EXIST ".venv\Scripts\jupyter.exe" (
    echo Installing dependencies from requirements.txt...
    call .venv\Scripts\python.exe -m pip install -r requirements.txt
)

set "CHROME_BIN=C:\Program Files\Google\Chrome\Application\chrome.exe"
IF NOT EXIST "%CHROME_BIN%" (
    set "CHROME_BIN=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
)

echo Opening Google Chrome...
IF EXIST "%CHROME_BIN%" (
    start "" "%CHROME_BIN%" "http://localhost:8888/tree"
) ELSE (
    start "" "http://localhost:8888/tree"
)

echo Starting Jupyter Server...
call .venv\Scripts\jupyter.exe lab --no-browser

pause
