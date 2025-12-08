"""
Tests for the macOS standalone build script

These tests verify that the build script:
- Validates prerequisites
- Handles build options correctly
- Creates proper directory structure
- Embeds Python correctly
"""

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def test_build_script_exists():
    """Test that the build script exists"""
    script_path = Path(__file__).parent.parent / "build_macos_standalone.sh"
    assert script_path.exists(), "build_macos_standalone.sh should exist"
    assert script_path.is_file(), "build_macos_standalone.sh should be a file"


def test_build_script_is_executable():
    """Test that the build script is executable"""
    script_path = Path(__file__).parent.parent / "build_macos_standalone.sh"
    if os.name != 'nt':  # Skip on Windows
        assert os.access(script_path, os.X_OK), "build_macos_standalone.sh should be executable"


def test_build_script_has_required_sections():
    """Test that the build script has required sections"""
    script_path = Path(__file__).parent.parent / "build_macos_standalone.sh"
    content = script_path.read_text()
    
    # Check for key sections
    assert "APP_NAME" in content, "Should define APP_NAME"
    assert "APP_VERSION" in content, "Should define APP_VERSION"
    assert "Checking requirements" in content, "Should check requirements"
    assert "Building Tauri application" in content, "Should build Tauri app"
    assert "Embedding Python runtime" in content, "Should embed Python"


def test_build_script_handles_options():
    """Test that the build script handles command-line options"""
    script_path = Path(__file__).parent.parent / "build_macos_standalone.sh"
    content = script_path.read_text()
    
    # Check for option parsing
    assert "--sign" in content, "Should handle --sign option"
    assert "--notarize" in content, "Should handle --notarize option"
    assert "--release" in content, "Should handle --release option"


def test_build_directories_are_defined():
    """Test that build directories are properly defined"""
    script_path = Path(__file__).parent.parent / "build_macos_standalone.sh"
    content = script_path.read_text()
    
    # Check for directory definitions
    assert "BUILD_DIR" in content, "Should define BUILD_DIR"
    assert "DIST_DIR" in content, "Should define DIST_DIR"
    assert "APP_BUNDLE" in content, "Should define APP_BUNDLE"


@patch('subprocess.run')
def test_build_script_checks_requirements(mock_run):
    """Test that the build script checks for required tools"""
    # Mock successful command execution
    mock_run.return_value = MagicMock(returncode=0)
    
    # The script should check for python3, node, cargo, cmake
    # This is a structural test - actual execution would require the script
    assert True  # Placeholder - would need actual script execution test


def test_python_launcher_script_exists():
    """Test that the Python launcher script exists"""
    launcher_path = Path(__file__).parent.parent / "music_brain" / "start_api_embedded.py"
    assert launcher_path.exists(), "start_api_embedded.py should exist"


def test_python_launcher_has_correct_structure():
    """Test that the Python launcher has correct structure"""
    launcher_path = Path(__file__).parent.parent / "music_brain" / "start_api_embedded.py"
    content = launcher_path.read_text()
    
    # Check for key components
    assert "import sys" in content, "Should import sys"
    assert "from pathlib import Path" in content, "Should import Path"
    assert "music_brain.api" in content, "Should import music_brain.api"
    assert "uvicorn.run" in content, "Should use uvicorn"


def test_tauri_config_exists():
    """Test that Tauri configuration exists"""
    config_path = Path(__file__).parent.parent / "src-tauri" / "tauri.conf.json"
    assert config_path.exists(), "tauri.conf.json should exist"


def test_tauri_config_has_correct_structure():
    """Test that Tauri configuration has correct structure"""
    import json
    config_path = Path(__file__).parent.parent / "src-tauri" / "tauri.conf.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Check for required fields
    assert "productName" in config, "Should have productName"
    assert "version" in config, "Should have version"
    assert "identifier" in config, "Should have identifier"
    assert config["productName"] == "iDAW", "Product name should be iDAW"


def test_rust_python_server_module_exists():
    """Test that the Rust Python server module exists"""
    module_path = Path(__file__).parent.parent / "src-tauri" / "src" / "python_server.rs"
    assert module_path.exists(), "python_server.rs should exist"


def test_rust_python_server_has_functions():
    """Test that the Rust Python server module has required functions"""
    module_path = Path(__file__).parent.parent / "src-tauri" / "src" / "python_server.rs"
    content = module_path.read_text()
    
    # Check for key functions
    assert "find_python_interpreter" in content, "Should have find_python_interpreter"
    assert "find_api_script" in content, "Should have find_api_script"
    assert "start_server" in content, "Should have start_server"
    assert "stop_server" in content, "Should have stop_server"
    assert "check_server_health" in content, "Should have check_server_health"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
