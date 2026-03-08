@echo off
echo ================================================
echo   Style Transfer AI - CLI Installation Script
echo ================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✓ Python found: 
python --version

:: Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo ✓ pip found: 
pip --version
echo.

:: Upgrade pip to latest version
echo Upgrading pip to latest version...
python -m pip install --upgrade pip
echo.

:: Install dependencies from requirements.txt
echo Installing Python dependencies...
if exist requirements.txt (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo WARNING: Some dependencies failed to install
        echo You may need to install them manually:
        echo   pip install requests openai google-generativeai
        echo.
    ) else (
        echo ✓ Dependencies installed successfully
    )
) else (
::echo Installing core dependencies manually...
    pip install requests openai google-generativeai
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)
echo.

:: Change to project root directory (parent of install folder)
cd ..
if not exist "src\" (
    echo ERROR: Cannot find project source directory
    echo Make sure you're running this from the install folder
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Install dependencies in virtual environment
echo Installing dependencies in virtual environment...
pip install requests openai google-generativeai

:: Install the CLI tool
echo Installing Style Transfer AI CLI...
pip install -e .
if %errorlevel% neq 0 (
    echo ERROR: Failed to install CLI tool
    echo Try running this script as Administrator
    pause
    exit /b 1
)

echo ✓ CLI tool installed successfully!
echo.

:: Add virtual environment to user PATH
echo Configuring global access...
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "UserPath=%%B"
if not defined UserPath set "UserPath="

:: Check if our path is already in PATH
echo %UserPath% | findstr /C:"%CD%\.venv\Scripts" >nul
if %errorlevel% neq 0 (
    echo Adding virtual environment to user PATH...
    if defined UserPath (
        setx PATH "%UserPath%;%CD%\.venv\Scripts"
    ) else (
        setx PATH "%CD%\.venv\Scripts"
    )
    echo ✓ PATH updated for global access
    echo NOTE: You may need to restart your command prompt for PATH changes to take effect
) else (
    echo ✓ Virtual environment already in PATH
)

:: Test the installation
echo Testing CLI installation...
.venv\Scripts\style-transfer-ai.exe --help >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: CLI test failed
    echo Try: .venv\Scripts\style-transfer-ai.exe
    echo.
) else (
    echo ✓ CLI command working correctly!
    echo.
    echo You can now use: style-transfer-ai (after restarting command prompt)
)

:: Check for Ollama
echo Checking for Ollama installation...
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo NOTICE: Ollama not found
    echo For local AI processing, install Ollama from:
    echo https://ollama.ai/download
    echo.
    echo After installing Ollama, run these commands:
    echo   ollama pull gemma3:1b
    echo   ollama serve
    echo.
) else (
    echo ✓ Ollama found: 
    ollama --version
    echo.
    echo Checking for required AI models...
    
    ollama list | findstr "gemma3:1b" >nul 2>&1
    if %errorlevel% neq 0 (
        echo - gemma3:1b model not found
        echo   Run: ollama pull gemma3:1b
    ) else (
        echo ✓ gemma3:1b model available
    )
)

echo.
echo ================================================
echo            Installation Complete!
echo ================================================
echo.
echo Usage:
echo   style-transfer-ai                    (Interactive mode)
echo   style-transfer-ai --help             (Show all options)
echo   style-transfer-ai --analyze file.txt (Analyze a file)
echo.
echo Sample files provided:
if exist about_my_pet.txt echo   - about_my_pet.txt
if exist about_my_pet_1.txt echo   - about_my_pet_1.txt
echo.
echo Test the installation:
echo   style-transfer-ai --analyze about_my_pet.txt
echo.
echo For API keys setup, see README.md
echo.
pause