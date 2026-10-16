#!/bin/bash
# Start the Stock Factor Analysis web interface
# Features:
# - Auto-restart on crash
# - Cleanup limited to the launched child process
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

# Stop only the child started by this launcher. Never kill a port's owner.
SERVER_PID=""
cleanup() {
    trap - INT TERM
    if [ -n "$SERVER_PID" ]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup INT TERM

if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $PORT is already in use; choose another port."
    exit 1
fi

# Main loop
while true; do
    echo -e "${GREEN}Starting Stock Factor Analysis on http://localhost:$PORT${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    echo ""
    
    # Run uvicorn
    # --reload ensures backend reloads on edits
    python -W ignore -m uvicorn backend.web:app --host 127.0.0.1 --port "$PORT" --reload &
    SERVER_PID=$!
    wait "$SERVER_PID"
    
    EXIT_CODE=$?
    SERVER_PID=""
    
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
    
    # A different owner may have acquired the port after the child exited.
    if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port $PORT is now in use; stopping without touching its owner."
        exit 1
    fi
done
