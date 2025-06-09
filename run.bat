@echo off
echo Starting AI Creative Pipeline...

:: Activate virtual environment
call .venv\Scripts\activate

:: Start the API server in a new window
start cmd /k "call .venv\Scripts\activate && python app/ignite.py"

:: Wait a few seconds for the API server to start
timeout /t 5

:: Start the Streamlit GUI
streamlit run app/gui.py 