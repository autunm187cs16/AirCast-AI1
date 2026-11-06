@echo off
REM AirCast AI launcher (Windows)
cd /d "%~dp0"

echo ============================================================
echo  AirCast AI
echo ============================================================
python --version
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :failed
    echo Upgrading pip...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    echo Installing dependencies ^(this can take a few minutes^)...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
) else (
    echo Using existing virtual environment.
)

echo.
echo Running environment check...
".venv\Scripts\python.exe" doctor.py
if errorlevel 1 goto :failed

echo.
echo Starting AirCast AI in your browser...
".venv\Scripts\python.exe" -m streamlit run app.py
goto :eof

:failed
echo.
echo ------------------------------------------------------------
echo Something went wrong above. Copy this window's output and
echo share it so the issue can be fixed.
echo ------------------------------------------------------------
pause
