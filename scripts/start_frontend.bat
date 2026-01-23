@echo off
REM AI Multimedia Platform - Frontend Startup Script (Windows)

setlocal enabledelayedexpansion

echo ========================================
echo   AI Multimedia Platform - Frontend
echo ========================================

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "FRONTEND_DIR=%PROJECT_ROOT%\vue-web"

REM Check if frontend directory exists
if not exist "%FRONTEND_DIR%" (
    echo Error: Frontend directory not found at %FRONTEND_DIR%
    exit /b 1
)

cd /d "%FRONTEND_DIR%"

REM Check if node_modules exists
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)

REM Parse arguments
set "BUILD_MODE=dev"
set "PORT=3000"

:parse_args
if "%~1"=="" goto start_frontend
if /i "%~1"=="--build" (
    set "BUILD_MODE=build"
    shift
    goto parse_args
)
if /i "%~1"=="--prod" (
    set "BUILD_MODE=build"
    shift
    goto parse_args
)
if /i "%~1"=="--preview" (
    set "BUILD_MODE=preview"
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
echo   --build, --prod  Build for production
echo   --preview        Preview production build
echo   --port PORT      Set port (default: 3000)
echo   -h, --help       Show this help message
exit /b 0

:start_frontend
echo.
if "%BUILD_MODE%"=="build" (
    echo Building for production...
    call npm run build
    echo Build complete! Output in dist/ directory
) else if "%BUILD_MODE%"=="preview" (
    echo Previewing production build on port %PORT%...
    call npm run preview -- --port %PORT%
) else (
    echo Starting development server on port %PORT%...
    call npm run dev -- --port %PORT%
)
