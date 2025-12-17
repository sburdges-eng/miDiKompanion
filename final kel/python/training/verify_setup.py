#!/usr/bin/env python3
"""
Verify training environment setup.

This script checks that all required dependencies are installed
and the environment is ready for model training.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ required")
        print(f"  Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True


def check_dependency(module_name, package_name=None):
    """Check if a Python module is available."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✓ {package_name} installed")
        return True
    except ImportError:
        print(f"✗ {package_name} not installed")
        print(f"  Install with: pip install {package_name}")
        return False


def main():
    print("=" * 60)
    print("Training Environment Verification")
    print("=" * 60)
    print()

    all_ok = True

    # Check Python version
    print("1. Python version:")
    if not check_python_version():
        all_ok = False
    print()

    # Check dependencies
    print("2. Required dependencies:")
    dependencies = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
    ]

    for module, package in dependencies:
        if not check_dependency(module, package):
            all_ok = False
    print()

    # Check optional dependencies
    print("3. Optional dependencies:")
    optional = [
        ('mido', 'mido'),
    ]

    for module, package in optional:
        check_dependency(module, package)
    print()

    # Check script files
    print("4. Training scripts:")
    scripts = [
        'train_emotion_model.py',
        'export_to_rtneural.py',
        'test_emotion_model.py',
        'create_test_model.py',
        'test_plugin_integration.py',
    ]

    script_dir = Path(__file__).parent
    for script in scripts:
        script_path = script_dir / script
        if script_path.exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} not found")
            all_ok = False
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed! Environment is ready for training.")
        print()
        print("Next steps:")
        print("1. Create test model: python create_test_model.py")
        print("2. Or train real model: python train_emotion_model.py --create-dummy")
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print()
        print("Install all dependencies:")
        print("  pip install -r ../requirements.txt")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())

