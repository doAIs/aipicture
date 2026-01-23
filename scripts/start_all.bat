@echo off
REM AI Multimedia Platform - Full Stack Startup Script (Windows)

setlocal enabledelayedexpansion

echo ================================================
echo   AI Multimedia Platform - Full Stack Startup
echo ================================================

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Default ports
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=3000"
set "DEV_MODE=true"

REM Parse arguments
:parse_args
if "%~1"=="" goto check_system
if /i "%~1"=="--prod" (
    set "DEV_MODE=false"
    shift
    goto parse_args
)
if /i "%~1"=="--backend-port" (
    set "BACKEND_PORT=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--frontend-port" (
    set "FRONTEND_PORT=%~2"
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
echo   --prod               Run in production mode
echo   --backend-port PORT  Set backend port (default: 8000)
echo   --frontend-port PORT Set frontend port (default: 3000)
echo   -h, --help           Show this help message
exit /b 0

:check_system
echo.
echo Checking system configuration...

REM Check GPU availability
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
) else (
    echo No NVIDIA GPU detected, will use CPU
)

echo.
echo ================================================
echo   Starting Services
echo ================================================

REM Start Backend in new window
echo.
echo Starting Backend Server on port %BACKEND_PORT%...
if "%DEV_MODE%"=="true" (
    start "AI Platform - Backend" cmd /c "%SCRIPT_DIR%start_backend.bat --dev --port %BACKEND_PORT%"
) else (
    start "AI Platform - Backend" cmd /c "%SCRIPT_DIR%start_backend.bat --port %BACKEND_PORT%"
)

REM Wait for backend to initialize
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

REM Start Frontend in new window
echo.
echo Starting Frontend Server on port %FRONTEND_PORT%...
if "%DEV_MODE%"=="true" (
    start "AI Platform - Frontend" cmd /c "%SCRIPT_DIR%start_frontend.bat --port %FRONTEND_PORT%"
) else (
    start "AI Platform - Frontend" cmd /c "%SCRIPT_DIR%start_frontend.bat --preview --port %FRONTEND_PORT%"
)

REM Wait for frontend to initialize
timeout /t 3 /nobreak >nul

REM Display access information
echo.
echo ================================================
echo   All services are starting!
echo ================================================
echo.
echo   Frontend:  http://localhost:%FRONTEND_PORT%
echo   Backend:   http://localhost:%BACKEND_PORT%
echo   API Docs:  http://localhost:%BACKEND_PORT%/docs
echo.
echo   Close the terminal windows to stop services
echo ================================================

pause
