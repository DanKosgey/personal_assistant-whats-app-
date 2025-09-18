@echo off
echo Starting WhatsApp AI Agent...
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Installing dependencies...
    .venv\Scripts\pip install -r requirements.txt
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\Activate.bat

REM Start the agent
echo Starting agent in ngrok mode...
python run_with_ngrok.py

pause