# WhatsApp AI Agent Startup Script

Write-Host "Starting WhatsApp AI Agent..." -ForegroundColor Green
Write-Host ""

# Get the directory of this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    .venv\Scripts\pip install -r requirements.txt
    Write-Host ""
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.venv\Scripts\Activate.ps1

# Start the agent
Write-Host "Starting agent in ngrok mode..." -ForegroundColor Green
python run_with_ngrok.py