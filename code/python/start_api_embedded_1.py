#!/usr/bin/env python3
"""
Embedded Music Brain API Server Launcher

This script is designed to run the Music Brain API server when embedded
in a standalone application bundle. It handles path setup and environment
configuration automatically.
"""
import sys
import os
from pathlib import Path

# Add the script's directory to Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# If we're in an app bundle, add Resources/python to path
if 'Resources' in str(script_dir):
    # We're in an app bundle
    resources_dir = script_dir.parent if script_dir.name == 'python' else script_dir
    python_resources = resources_dir / 'python'
    if python_resources.exists():
        sys.path.insert(0, str(python_resources))

# Also check parent directories for music_brain
current = script_dir
for _ in range(5):  # Check up to 5 levels up
    music_brain_path = current / 'music_brain'
    if music_brain_path.exists() and music_brain_path.is_dir():
        sys.path.insert(0, str(current))
        break
    current = current.parent

# Set up data paths
if 'Resources' in str(script_dir):
    # In app bundle, data might be in Resources/data
    data_paths = [
        script_dir.parent / 'data',
        script_dir / 'data',
        Path(__file__).parent.parent / 'Data_Files',
    ]
    for data_path in data_paths:
        if data_path.exists():
            os.environ['MUSIC_BRAIN_DATA_PATH'] = str(data_path)
            break

# Now import and run the API
try:
    from music_brain.api import app
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("MUSIC_BRAIN_PORT", "8000"))
    host = os.environ.get("MUSIC_BRAIN_HOST", "127.0.0.1")
    
    # Validate port range
    if not (1024 <= port <= 65535):
        print(f"Error: Port {port} is out of valid range (1024-65535)", file=sys.stderr)
        sys.exit(1)
    
    print(f"Starting Music Brain API server on {host}:{port}")
    print(f"Python path: {sys.path[:3]}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
except ImportError as e:
    print(f"Error importing music_brain.api: {e}", file=sys.stderr)
    print(f"Python path: {sys.path}", file=sys.stderr)
    print(f"Current directory: {os.getcwd()}", file=sys.stderr)
    print("\nTroubleshooting:", file=sys.stderr)
    print("1. Ensure music_brain package is in Python path", file=sys.stderr)
    print("2. Check that all dependencies are installed", file=sys.stderr)
    print("3. Verify music_brain/api.py exists", file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f"Error: Invalid configuration - {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error starting server: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
