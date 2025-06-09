@echo off
echo Installing dependencies...

:: Remove existing virtual environment if it exists
if exist .venv (
    echo Removing existing virtual environment...
    rmdir /s /q .venv
)

:: Create new virtual environment with Python 3.8
echo Creating new virtual environment...
python -m venv .venv --clear

:: Activate virtual environment
call .venv\Scripts\activate

:: Upgrade pip and install build tools
python -m pip install --upgrade pip
pip install wheel setuptools

:: Install core dependencies first
echo Installing core dependencies...
pip install marshmallow==3.19.0
pip install flask==2.3.3
pip install flask-apispec==0.11.4
pip install gevent==22.10.2

:: Install openfabric and other dependencies
echo Installing remaining dependencies...
pip install openfabric-pysdk==0.2.9
pip install transformers==4.30.0
pip install torch==2.0.0
pip install numpy==1.24.0
pip install pillow==9.5.0
pip install requests==2.31.0
pip install python-dotenv==1.0.0
pip install faiss-cpu==1.7.4
pip install sentence-transformers==2.2.2
pip install streamlit==1.24.0
pip install gradio==3.50.2

echo Installation complete!
pause 