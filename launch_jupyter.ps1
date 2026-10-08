# ===================================================
# Launch JupyterLab / Notebook directly in Google Chrome
# ===================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# 1. Activate or create .venv
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment .venv..." -ForegroundColor Cyan
    python -m venv .venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

if (-not (Test-Path ".\.venv\Scripts\jupyter.exe")) {
    Write-Host "Jupyter not found. Installing requirements from requirements.txt..." -ForegroundColor Yellow
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
}

# 2. Locate Chrome executable
$chromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path $chromePath)) {
    $chromePath = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
}

# 3. Set BROWSER environment variable so Python's webbrowser module picks up Chrome
$env:BROWSER = $chromePath

# 4. Launch Chrome directly to the local server URL
Write-Host "Opening Google Chrome..." -ForegroundColor Green
if (Test-Path $chromePath) {
    Start-Process $chromePath "http://localhost:8888/tree"
} else {
    Start-Process "http://localhost:8888/tree"
}

# 5. Start the Jupyter server
Write-Host "Starting Jupyter Server..." -ForegroundColor Cyan
& ".\.venv\Scripts\jupyter.exe" lab --no-browser
