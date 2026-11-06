@echo off
setlocal
cd /d "%~dp0"
title AirCast AI - publish to GitHub

echo ============================================================
echo  Publish AirCast AI to GitHub
echo ============================================================
echo.

where git >nul 2>&1
if errorlevel 1 (
    echo Git is not installed.
    echo Install it from https://git-scm.com/download/win and run this again.
    pause
    exit /b 1
)

git --version

REM --- identity (needed once for the first commit) -------------------------
git config --global user.name >nul 2>&1
if errorlevel 1 (
    set /p GITNAME=Enter your name: 
    git config --global user.name "%GITNAME%"
)
git config --global user.email >nul 2>&1
if errorlevel 1 (
    set /p GITMAIL=Enter your GitHub email: 
    git config --global user.email "%GITMAIL%"
)

REM --- repository -----------------------------------------------------------
if not exist ".git" (
    echo.
    echo Initialising repository...
    git init
    git branch -M main
)

echo.
echo Staging project files (.venv and caches are ignored)...
git add -A

git diff --cached --quiet
if errorlevel 1 (
    git commit -m "AirCast AI - PM2.5 prediction with machine learning"
) else (
    echo Nothing new to commit.
)

echo.
echo Create an EMPTY public repository at https://github.com/new
echo   - name: AirCast-AI
echo   - visibility: Public
echo   - do NOT add a README, .gitignore or licence
echo.
set /p REPO=Paste the repository URL here (e.g. https://github.com/USER/AirCast-AI.git): 

if "%REPO%"=="" (
    echo No URL entered - stopping.
    pause
    exit /b 1
)

git remote remove origin >nul 2>&1
git remote add origin %REPO%

echo.
echo Pushing... sign in to GitHub in the window that opens.
git push -u origin main
if errorlevel 1 (
    echo.
    echo ------------------------------------------------------------
    echo Push failed. Common causes:
    echo   1. The repository already contains a README - delete it or
    echo      create a new empty repository, then run this file again.
    echo   2. You are signed in to a different GitHub account.
    echo   3. The URL is wrong (it must end with .git).
    echo ------------------------------------------------------------
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done. Open your repository URL in an incognito window to
echo  confirm it is public, then submit that URL.
echo ============================================================
pause
