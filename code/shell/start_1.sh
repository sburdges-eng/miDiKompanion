#!/bin/bash
# Simple launcher script for 1DAWCURSORV1

PORT=${1:-8000}

echo "Starting 1DAWCURSORV1 on http://localhost:$PORT"
echo "Press Ctrl+C to stop"

# Try Python 3 first
if command -v python3 &> /dev/null; then
    python3 -m http.server "$PORT"
# Fallback to Python 2
elif command -v python &> /dev/null; then
    python -m SimpleHTTPServer "$PORT"
# Fallback to Node.js serve
elif command -v npx &> /dev/null; then
    npx serve -s . -l "$PORT"
else
    echo "Error: No HTTP server found. Please install Python 3 or Node.js"
    echo "Or open index.html directly in your browser"
    exit 1
fi
