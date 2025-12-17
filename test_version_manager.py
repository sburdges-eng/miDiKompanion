#!/usr/bin/env python3
"""
Tests for version_manager.py

Basic tests to ensure version bumping works correctly.
"""

import tempfile
import shutil
from pathlib import Path
from version_manager import Version, VersionManager


def test_version_parsing():
    """Test version string parsing."""
    v1 = Version.from_string("1.2.3")
    assert v1.major == 1
    assert v1.minor == 2
    assert v1.patch == 3
    assert str(v1) == "1.2.3"
    
    v2 = Version.from_string("v1.2.3")
    assert v2.major == 1
    assert v2.minor == 2
    assert v2.patch == 3
    
    print("✓ Version parsing tests passed")


def test_version_bumps():
    """Test version bump logic."""
    v = Version(1, 2, 3)
    
    # Test patch bump
    v_patch = v.bump_patch()
    assert v_patch.major == 1
    assert v_patch.minor == 2
    assert v_patch.patch == 4
    
    # Test minor bump
    v_minor = v.bump_minor()
    assert v_minor.major == 1
    assert v_minor.minor == 3
    assert v_minor.patch == 0
    
    # Test major bump
    v_major = v.bump_major()
    assert v_major.major == 2
    assert v_major.minor == 0
    assert v_major.patch == 0
    
    print("✓ Version bump tests passed")


def test_version_manager():
    """Test VersionManager with temporary files."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)
        
        # Create test files
        version_file = repo_root / "VERSION"
        version_file.write_text("1.0.0\n")
        
        pyproject_file = repo_root / "pyproject.toml"
        pyproject_file.write_text("""
[project]
name = "test"
version = "1.0.0"
description = "Test"
""")
        
        package_file = repo_root / "package.json"
        package_file.write_text('{\n  "name": "test",\n  "version": "1.0.0"\n}\n')
        
        # Create Version.h directory and file
        version_h_dir = repo_root / "iDAW_Core" / "include"
        version_h_dir.mkdir(parents=True)
        version_h_file = version_h_dir / "Version.h"
        version_h_file.write_text("""
#define IDAW_VERSION_MAJOR 1
#define IDAW_VERSION_MINOR 0
#define IDAW_VERSION_PATCH 0
#define IDAW_VERSION_STRING "1.0.0"
""")
        
        # Test VersionManager
        manager = VersionManager(repo_root)
        
        # Test get current version
        current = manager.get_current_version()
        assert current.major == 1
        assert current.minor == 0
        assert current.patch == 0
        
        # Test patch bump
        new_version = manager.bump_version('patch', 'Test patch bump')
        assert new_version.major == 1
        assert new_version.minor == 0
        assert new_version.patch == 1
        
        # Verify all files were updated
        assert version_file.read_text().strip() == "1.0.1"
        assert '"version": "1.0.1"' in package_file.read_text()
        assert 'version = "1.0.1"' in pyproject_file.read_text()
        assert '#define IDAW_VERSION_PATCH 1' in version_h_file.read_text()
        assert '#define IDAW_VERSION_STRING "1.0.1"' in version_h_file.read_text()
        
        # Test minor bump
        new_version = manager.bump_version('minor', 'Test minor bump')
        assert new_version.major == 1
        assert new_version.minor == 1
        assert new_version.patch == 0
        assert version_file.read_text().strip() == "1.1.0"
        
        # Test major bump
        new_version = manager.bump_version('major', 'Test major bump')
        assert new_version.major == 2
        assert new_version.minor == 0
        assert new_version.patch == 0
        assert version_file.read_text().strip() == "2.0.0"
        
        print("✓ VersionManager tests passed")


def main():
    """Run all tests."""
    print("Running version_manager tests...\n")
    
    test_version_parsing()
    test_version_bumps()
    test_version_manager()
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    main()
