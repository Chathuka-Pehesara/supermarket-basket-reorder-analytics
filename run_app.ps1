Write-Host "========================================================" -ForegroundColor Green
Write-Host "Launching Supermarket Merchandising Portal (Streamlit)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Green

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (Test-Path ".\.venv\Scripts\streamlit.exe") {
    & ".\.venv\Scripts\streamlit.exe" run app.py
} else {
    python -m streamlit run app.py
}
