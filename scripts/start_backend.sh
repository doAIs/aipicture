#!/bin/bash

# AI Multimedia Platform - Backend Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting AI Multimedia Platform Backend...${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
    pip install -r backend/requirements.txt
fi

# Create necessary directories
mkdir -p outputs/images outputs/videos models

# Set environment variables
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Check for GPU
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}GPU detected and available${NC}"
else
    echo -e "${YELLOW}Warning: GPU not available, running on CPU${NC}"
fi

# Start the backend server
echo -e "${GREEN}Starting FastAPI server on port 8000...${NC}"

# Development mode with hot reload
if [ "$1" == "--dev" ]; then
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
else
    # Production mode
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
fi
