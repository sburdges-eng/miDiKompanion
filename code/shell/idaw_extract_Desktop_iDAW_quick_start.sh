#!/bin/bash
# Quick start with API key as argument

if [ -z "$1" ]; then
    echo "Usage: ./quick_start.sh YOUR_API_KEY"
    echo ""
    echo "Get your FREE API key at: https://freesound.org/apiv2/apply/"
    echo ""
    echo "Example:"
    echo "  ./quick_start.sh abc123xyz456..."
    exit 1
fi

API_KEY="$1"

echo "{"\"freesound_api_key"\": "\"$API_KEY"\"}" > freesound_config.json

echo "✓ API key saved!"
echo ""
echo "Starting automatic download..."
echo ""

./auto_emotion_sampler.py start
