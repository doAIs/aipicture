#!/bin/bash
# AI Multimedia Platform - Full Stack Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${MAGENTA}================================================${NC}"
echo -e "${MAGENTA}  AI Multimedia Platform - Full Stack Startup   ${NC}"
echo -e "${MAGENTA}================================================${NC}"

# Default ports
BACKEND_PORT=8000
FRONTEND_PORT=3000
DEV_MODE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod)
            DEV_MODE=false
            shift
            ;;
        --backend-port)
            BACKEND_PORT="$2"
            shift 2
            ;;
        --frontend-port)
            FRONTEND_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prod              Run in production mode"
            echo "  --backend-port PORT Set backend port (default: 8000)"
            echo "  --frontend-port PORT Set frontend port (default: 3000)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Services stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check GPU availability
echo -e "${CYAN}Checking system configuration...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}No NVIDIA GPU detected, will use CPU${NC}"
fi

# Start Backend
echo -e "\n${CYAN}Starting Backend Server...${NC}"
cd "$PROJECT_ROOT"

if [ "$DEV_MODE" = true ]; then
    "$SCRIPT_DIR/start_backend.sh" --dev --port $BACKEND_PORT &
else
    "$SCRIPT_DIR/start_backend.sh" --port $BACKEND_PORT &
fi
BACKEND_PID=$!

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to initialize...${NC}"
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Backend failed to start!${NC}"
    exit 1
fi
echo -e "${GREEN}Backend started on port $BACKEND_PORT${NC}"

# Start Frontend
echo -e "\n${CYAN}Starting Frontend Server...${NC}"

if [ "$DEV_MODE" = true ]; then
    "$SCRIPT_DIR/start_frontend.sh" --port $FRONTEND_PORT &
else
    "$SCRIPT_DIR/start_frontend.sh" --preview --port $FRONTEND_PORT &
fi
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Frontend failed to start!${NC}"
    cleanup
    exit 1
fi
echo -e "${GREEN}Frontend started on port $FRONTEND_PORT${NC}"

# Display access information
echo -e "\n${MAGENTA}================================================${NC}"
echo -e "${GREEN}All services are running!${NC}"
echo -e "${MAGENTA}================================================${NC}"
echo -e ""
echo -e "  ${CYAN}Frontend:${NC}  http://localhost:$FRONTEND_PORT"
echo -e "  ${CYAN}Backend:${NC}   http://localhost:$BACKEND_PORT"
echo -e "  ${CYAN}API Docs:${NC}  http://localhost:$BACKEND_PORT/docs"
echo -e ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo -e "${MAGENTA}================================================${NC}"

# Wait for both processes
wait
