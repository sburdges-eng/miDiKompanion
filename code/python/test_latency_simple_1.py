"""
Simple Latency Tests for iDAW System Components

Tests latency of individual components without requiring full server startup.
"""

import time
import sys
from pathlib import Path
import pytest


def test_python_import_latency():
    """Test Python module import latency"""
    start = time.time()
    
    # Test music_brain imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    import_start = time.time()
    try:
        from music_brain import cli
        import_time = time.time() - import_start
        print(f"\n✓ music_brain.cli import: {import_time:.3f}s")
        assert import_time < 1.0, f"Import took {import_time:.3f}s, expected < 1.0s"
    except ImportError:
        # Module might not be fully installed, skip
        pytest.skip("music_brain not fully installed")
    
    total_time = time.time() - start
    assert total_time < 2.0, f"Total import time {total_time:.3f}s too slow"


def test_rust_compilation_check():
    """Test Rust code structure (syntax check)"""
    import subprocess
    
    start = time.time()
    
    # Check if Rust files have correct structure
    rust_checks = {
        "python_server.rs": ["pub async fn start_server", "pub async fn check_server_health"],
        "commands.rs": ["pub async fn generate_music", "pub async fn get_emotions"],
        "main.rs": ["AppState", "python_server::"],
    }
    
    for filename, patterns in rust_checks.items():
        rust_file = Path(__file__).parent.parent / "src-tauri" / "src" / filename
        if rust_file.exists():
            # Check for key patterns
            for pattern in patterns:
                result = subprocess.run(
                    ["grep", "-q", pattern, str(rust_file)],
                    capture_output=True
                )
                assert result.returncode == 0, f"{filename} missing pattern: {pattern}"
        else:
            pytest.fail(f"{filename} not found")
    
    check_time = time.time() - start
    print(f"\n✓ Rust structure check: {check_time:.3f}s")
    assert check_time < 1.0


def test_script_execution_latency():
    """Test script execution startup latency"""
    import subprocess
    
    scripts = [
        Path(__file__).parent.parent / "scripts" / "build_macos.sh",
        Path(__file__).parent.parent / "scripts" / "fork_setup.sh",
    ]
    
    for script in scripts:
        if script.exists():
            start = time.time()
            # Just check syntax, don't execute
            result = subprocess.run(
                ["bash", "-n", str(script)],
                capture_output=True,
                timeout=5
            )
            syntax_time = time.time() - start
            
            assert result.returncode == 0, f"{script.name} has syntax errors"
            assert syntax_time < 1.0, f"Syntax check took {syntax_time:.3f}s"
            print(f"✓ {script.name} syntax check: {syntax_time:.3f}s")


def test_file_access_latency():
    """Test file system access latency for key files"""
    project_root = Path(__file__).parent.parent
    
    key_files = [
        "src-tauri/src/python_server.rs",
        "src-tauri/src/commands.rs",
        "music_brain/start_api_embedded.py",
        "scripts/build_macos.sh",
        "src/hooks/useMusicBrain.ts",
    ]
    
    start = time.time()
    accessed = 0
    
    for file_path in key_files:
        full_path = project_root / file_path
        if full_path.exists():
            # Quick read test
            with open(full_path, 'rb') as f:
                f.read(1024)  # Read first 1KB
            accessed += 1
    
    access_time = time.time() - start
    avg_time = access_time / len(key_files) if key_files else 0
    
    print(f"\n✓ File access: {accessed}/{len(key_files)} files, avg={avg_time:.4f}s")
    assert avg_time < 0.01, f"File access too slow: {avg_time:.4f}s"


def test_json_parsing_latency():
    """Test JSON configuration parsing latency"""
    import json
    
    config_files = [
        Path(__file__).parent.parent / "src-tauri" / "tauri.conf.json",
        Path(__file__).parent.parent / "package.json",
    ]
    
    total_time = 0
    parsed = 0
    
    for config_file in config_files:
        if config_file.exists():
            start = time.time()
            with open(config_file) as f:
                data = json.load(f)
            parse_time = time.time() - start
            total_time += parse_time
            parsed += 1
            
            assert parse_time < 0.1, f"{config_file.name} parsing too slow: {parse_time:.3f}s"
    
    if parsed > 0:
        avg_time = total_time / parsed
        print(f"\n✓ JSON parsing: {parsed} files, avg={avg_time:.4f}s")


def test_integration_structure():
    """Test that integration structure is correct"""
    project_root = Path(__file__).parent.parent
    
    # Check integration points exist
    integration_points = {
        "Rust Python Server": project_root / "src-tauri" / "src" / "python_server.rs",
        "Rust Commands": project_root / "src-tauri" / "src" / "commands.rs",
        "Rust Main": project_root / "src-tauri" / "src" / "main.rs",
        "Python Launcher": project_root / "music_brain" / "start_api_embedded.py",
        "Frontend Hook": project_root / "src" / "hooks" / "useMusicBrain.ts",
        "Build Script": project_root / "scripts" / "build_macos.sh",
    }
    
    start = time.time()
    missing = []
    
    for name, path in integration_points.items():
        if not path.exists():
            missing.append(name)
    
    check_time = time.time() - start
    
    assert len(missing) == 0, f"Missing integration points: {', '.join(missing)}"
    print(f"\n✓ Integration structure: All {len(integration_points)} points present ({check_time:.3f}s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
