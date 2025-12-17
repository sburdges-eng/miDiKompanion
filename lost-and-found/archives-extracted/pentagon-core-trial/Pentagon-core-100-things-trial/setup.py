"""
Setup script for Bulling - Bowling Scoring Game

This script can be used to:
1. Install Bulling as a command-line executable: pip install .
2. Create a macOS application bundle (.app): python setup.py py2app

After installation, run Bulling with: bulling
"""

import sys
from setuptools import setup

# For macOS app bundle
APP = ['bulling_qt.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'app_icon.icns',  # Will be created
    'plist': {
        'CFBundleName': 'Bulling',
        'CFBundleDisplayName': 'Bulling',
        'CFBundleGetInfoString': 'Bowling Scoring Game - Bull Head Edition',
        'CFBundleIdentifier': 'com.bulling.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHumanReadableCopyright': '© 2025 Bulling',
        'NSHighResolutionCapable': True,
    },
    'packages': ['PySide6'],
    'includes': [],
    'excludes': ['tkinter', 'matplotlib', 'numpy'],
    'strip': True,
    'optimize': 2,
}

setup(
    name='bulling',
    version='1.0.0',
    description='Bowling Scoring Game - Bull Head Edition',
    author='Bulling Team',
    py_modules=['bulling_qt'],
    install_requires=[
        'PySide6>=6.5.0',
    ],
    entry_points={
        'console_scripts': [
            'bulling=bulling_qt:main',
        ],
    },
    python_requires='>=3.9',
    # For macOS app bundle (optional)
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'] if 'py2app' in sys.argv else [],
)
