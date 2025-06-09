@echo off

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Create necessary directories
if not exist generated_assets mkdir generated_assets

REM Start the application
echo Starting application...
python app/main.py 