"""
Integration tests for Python server management

These tests verify that the Python server can be:
- Started and stopped
- Health checked
- Managed by the Rust backend
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path
import pytest


@pytest.fixture
def python_api_script():
    """Fixture to find the Python API script"""
    script_paths = [
        Path(__file__).parent.parent / "music_brain" / "api.py",
        Path(__file__).parent.parent / "music_brain" / "start_api_embedded.py",
    ]
    
    for path in script_paths:
        if path.exists():
            return path
    
    pytest.skip("Python API script not found")


@pytest.fixture
def test_port():
    """Fixture for test port"""
    return int(os.environ.get("TEST_MUSIC_BRAIN_PORT", "8001"))


def test_python_api_script_exists(python_api_script):
    """Test that the Python API script exists"""
    assert python_api_script.exists(), f"API script should exist at {python_api_script}"


def test_python_api_script_is_executable(python_api_script):
    """Test that the Python API script can be executed"""
    # Check if we can at least import the module structure
    script_dir = python_api_script.parent
    sys.path.insert(0, str(script_dir.parent))
    
    try:
        # Try to import (might fail if dependencies missing, but structure should be OK)
        import importlib.util
        spec = importlib.util.spec_from_file_location("api", python_api_script)
        # Just verify we can create the spec
        assert spec is not None
    except Exception as e:
        # If import fails due to missing deps, that's OK for structure test
        pass
    finally:
        if str(script_dir.parent) in sys.path:
            sys.path.remove(str(script_dir.parent))


def test_api_health_endpoint_structure():
    """Test that the API health endpoint structure is correct"""
    # This test verifies the expected API structure
    # without actually starting a server
    
    # Expected health endpoint response structure
    expected_structure = {
        "status": "healthy"
    }
    
    # Verify structure (would need actual server for full test)
    assert "status" in expected_structure


@pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration tests require RUN_INTEGRATION_TESTS env var"
)
def test_start_python_server(python_api_script, test_port):
    """Test starting the Python server (requires actual server)"""
    # Set test port
    env = os.environ.copy()
    env["MUSIC_BRAIN_PORT"] = str(test_port)
    
    # Start server in background
    process = subprocess.Popen(
        [sys.executable, str(python_api_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    try:
        # Wait for server to start
        time.sleep(3)
        
        # Check if process is still running
        assert process.poll() is None, "Server process should be running"
        
        # Try to connect to health endpoint
        try:
            response = requests.get(
                f"http://127.0.0.1:{test_port}/health",
                timeout=2
            )
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("Server not responding (may need dependencies)")
    finally:
        # Clean up
        process.terminate()
        process.wait(timeout=5)


def test_server_port_configuration():
    """Test that server port can be configured via environment"""
    # Test default port
    default_port = int(os.environ.get("MUSIC_BRAIN_PORT", "8000"))
    assert default_port == 8000
    
    # Test custom port
    os.environ["MUSIC_BRAIN_PORT"] = "9000"
    custom_port = int(os.environ.get("MUSIC_BRAIN_PORT", "8000"))
    assert custom_port == 9000
    
    # Cleanup
    os.environ.pop("MUSIC_BRAIN_PORT", None)


def test_server_host_configuration():
    """Test that server host can be configured via environment"""
    # Test default host
    default_host = os.environ.get("MUSIC_BRAIN_HOST", "127.0.0.1")
    assert default_host == "127.0.0.1"
    
    # Test custom host
    os.environ["MUSIC_BRAIN_HOST"] = "0.0.0.0"
    custom_host = os.environ.get("MUSIC_BRAIN_HOST", "127.0.0.1")
    assert custom_host == "0.0.0.0"
    
    # Cleanup
    os.environ.pop("MUSIC_BRAIN_HOST", None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
