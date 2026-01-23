#!/bin/bash
# AI Multimedia Platform - Frontend Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/vue-web"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  AI Multimedia Platform - Frontend    ${NC}"
echo -e "${CYAN}========================================${NC}"

# Check if frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
    echo -e "${RED}Error: Frontend directory not found at $FRONTEND_DIR${NC}"
    exit 1
fi

cd "$FRONTEND_DIR"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Parse arguments
BUILD_MODE="dev"
PORT=3000

while [[ $# -gt 0 ]]; do
    case $1 in
        --build|--prod)
            BUILD_MODE="build"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --preview)
            BUILD_MODE="preview"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --build, --prod  Build for production"
            echo "  --preview        Preview production build"
            echo "  --port PORT      Set port (default: 3000)"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Start frontend
if [ "$BUILD_MODE" == "build" ]; then
    echo -e "${GREEN}Building for production...${NC}"
    npm run build
    echo -e "${GREEN}Build complete! Output in dist/ directory${NC}"
elif [ "$BUILD_MODE" == "preview" ]; then
    echo -e "${GREEN}Previewing production build on port $PORT...${NC}"
    npm run preview -- --port $PORT
else
    echo -e "${GREEN}Starting development server on port $PORT...${NC}"
    npm run dev -- --port $PORT
fi
