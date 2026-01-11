#!/bin/bash
# Start the Stock Factor Analysis web interface
# Features:
# - Auto-restart on crash
# - Robust process cleanup (prevents "Address already in use")
# - Hot reloading via Uvicorn

PORT=${1:-8000}
MAX_RESTARTS=10
RESTART_COUNT=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project directory
cd "$(dirname "$0")"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Stoppping server...${NC}"
    
    # Find PID using port
    local pid=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Killing process on port $PORT (PID: $pid)${NC}"
        kill -15 $pid 2>/dev/null # Try SIGTERM first
        sleep 1
        
        # Check if still alive
        if kill -0 $pid 2>/dev/null; then
            echo -e "${RED}Force killing process $pid${NC}"
            kill -9 $pid 2>/dev/null
        fi
    fi
    exit 0
}

# Trap signals
trap cleanup INT TERM

# Initial cleanup
pid=$(lsof -ti:$PORT 2>/dev/null)
if [ -n "$pid" ]; then
    echo -e "${YELLOW}Cleaning up existing process on port $PORT...${NC}"
    kill -9 $pid 2>/dev/null
    sleep 1
fi

# Main loop
while true; do
    echo -e "${GREEN}Starting Stock Factor Analysis on http://localhost:$PORT${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    # Run uvicorn
    # --reload ensures backend reloads on edits
    python -W ignore -m uvicorn backend.web:app --host 0.0.0.0 --port $PORT --reload
    
    EXIT_CODE=$?
    
    # If exit code is 0, user likely stopped it manually (but trap usually catches this)
    # If exit code is 130 (SIGINT), we stop
    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ]; then
        echo -e "${GREEN}Server stopped.${NC}"
        break
    fi
    
    RESTART_COUNT=$((RESTART_COUNT + 1))
    
    if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
        echo -e "${RED}Server crashed too many times ($MAX_RESTARTS). Stopping.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Server crashed (code: $EXIT_CODE). Restarting in 2s... ($RESTART_COUNT/$MAX_RESTARTS)${NC}"
    sleep 2
    
    # Ensure port is free before restart
    pid=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null
    fi
done
