@echo off
REM AI Multimedia Platform - Backend Startup Script (Windows)

setlocal enabledelayedexpansion

echo ========================================
echo   AI Multimedia Platform - Backend
echo ========================================

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

cd /d "%PROJECT_ROOT%"

REM Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found, using system Python
)

REM Check GPU availability
echo Checking GPU availability...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
) else (
    echo No NVIDIA GPU detected, will use CPU
)

REM Parse arguments
set "MODE=prod"
set "PORT=8000"

:parse_args
if "%~1"=="" goto start_server
if /i "%~1"=="--dev" (
    set "MODE=dev"
    shift
    goto parse_args
)
if /i "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--help" goto show_help
echo Unknown option: %~1
exit /b 1

:show_help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --dev         Run in development mode with hot reload
echo   --port PORT   Set port (default: 8000)
echo   -h, --help    Show this help message
exit /b 0

:start_server
echo.
if "%MODE%"=="dev" (
    echo Starting backend in DEVELOPMENT mode on port %PORT%...
    python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port %PORT%
) else (
    echo Starting backend in PRODUCTION mode on port %PORT%...
    python -m uvicorn backend.main:app --host 0.0.0.0 --port %PORT% --workers 4
)
