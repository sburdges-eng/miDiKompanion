#!/bin/bash
#
# iDAW - Kelly Project Unified Start Script
# Starts both the Python Music Brain API and Tauri development server
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🎵 iDAW - Kelly Project${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check for Node.js
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed${NC}"
    exit 1
fi

# Check for Rust/Cargo (for Tauri)
if ! command -v cargo &> /dev/null; then
    echo -e "${YELLOW}Warning: Rust/Cargo not found. Tauri commands may not work.${NC}"
    echo -e "${YELLOW}Install Rust from: https://rustup.rs${NC}"
fi

# Install Python dependencies if needed
echo -e "${GREEN}[1/4] Checking Python dependencies...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

# Install npm dependencies if needed
echo -e "${GREEN}[2/4] Checking Node.js dependencies...${NC}"
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Start Python Music Brain API in background
echo -e "${GREEN}[3/4] Starting Music Brain API server...${NC}"
echo -e "       API will be available at: ${BLUE}http://127.0.0.1:8000${NC}"
python -m uvicorn music_brain.api:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# Wait for API to start
sleep 2

# Check if API is running
if ! kill -0 $API_PID 2>/dev/null; then
    echo -e "${RED}Error: Music Brain API failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}       ✓ Music Brain API running (PID: $API_PID)${NC}"

# Start Tauri development server
echo -e "${GREEN}[4/4] Starting Tauri development server...${NC}"
echo -e "       App will be available at: ${BLUE}http://localhost:1420${NC}"
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}✨ iDAW is starting...${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop all services"
echo ""

# Run Tauri dev (this will block)
npm run tauri dev

# Cleanup when done
cleanup
