# ===================================================
# Launch Jupyter Notebook directly in Google Chrome
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

# 2. Check if jupyter is installed in venv
$jupyterInstalled = (& ".\.venv\Scripts\pip.exe" list | Select-String "jupyter")
if (-not $jupyterInstalled) {
    Write-Host "Installing jupyter in .venv..." -ForegroundColor Yellow
    & ".\.venv\Scripts\pip.exe" install jupyter
}

# 3. Locate Chrome executable
$chromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path $chromePath)) {
    $chromePath = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
}

Write-Host "Launching Jupyter Notebook in Google Chrome..." -ForegroundColor Green
jupyter notebook --browser="$chromePath %s"
