#!/usr/bin/env python3
"""
Create organized zip files of code, markdown, MIDI, and WAV files.
Max 25MB per zip, named reposean1.zip, reposean2.zip, etc.
"""

import os
import zipfile
from pathlib import Path
from collections import defaultdict

# Configuration
MAX_ZIP_SIZE = 25 * 1024 * 1024  # 25 MB
OUTPUT_DIR = Path("/Users/seanburdges/Desktop")
OUTPUT_PREFIX = "reposean"

# Directories to scan
SCAN_DIRS = [
    "/Users/seanburdges/Desktop",
    "/Users/seanburdges/Documents"
]

# File extensions to include
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.swift', '.rs', '.cpp', '.c', '.h', '.hpp',
    '.java', '.html', '.css', '.scss', '.json', '.yaml', '.yml', '.sh', '.sql',
    '.xml', '.toml', '.cfg', '.ini', '.gradle', '.cmake', '.mk', '.spec',
    '.plist', '.entitlements', '.storyboard', '.xib', '.pbxproj', '.xcscheme',
    '.m', '.mm', '.r', '.rb', '.go', '.kt', '.scala', '.php', '.pl', '.lua',
    '.dockerfile', '.gitignore', '.env', '.txt'  # including txt for config/readme type files
}
MARKDOWN_EXTENSIONS = {'.md', '.markdown', '.rst'}
MIDI_EXTENSIONS = {'.mid', '.midi'}
AUDIO_EXTENSIONS = {'.wav'}

ALL_EXTENSIONS = CODE_EXTENSIONS | MARKDOWN_EXTENSIONS | MIDI_EXTENSIONS | AUDIO_EXTENSIONS

# Directories/patterns to exclude
EXCLUDE_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env',
    '.pytest_cache', '.mypy_cache', '.tox', 'build', 'dist', '.eggs',
    # System frameworks to skip
    'Python.framework', 'Python3.framework', 'PythonT.framework',
    'JavaScriptCore.framework', '_WebKit_SwiftUI.framework', 
    'PlugInKitDaemon.framework',
    # Third-party frameworks to skip
    'JUCE 4', 'JUCE 2', 'JUCE',
    # Skip existing organized zip folders
    'ALL_ZIPS_ORGANIZED', 'ORGANIZED_ZIPS', 'ZIP_ORGANIZED_FINAL',
    # Skip .tmp folders
    '.tmp.drivedownload', '.tmp.driveupload',
    # Skip extracted archives
    'Archive_2_copy_extracted',
    # Skip git-core and man (system tools)
    'git-core', 'man'
}

# Skip files with these patterns
SKIP_PATTERNS = {'.zip', '.pyc', '.pyo', '.so', '.dylib', '.o', '.a'}

def should_skip_dir(dir_path):
    """Check if directory should be skipped."""
    dir_name = os.path.basename(dir_path)
    # Skip if directory name matches exclude list
    if dir_name in EXCLUDE_DIRS:
        return True
    # Skip if it's a .framework directory (system framework)
    if dir_name.endswith('.framework'):
        return True
    # Skip if path contains excluded directories
    parts = Path(dir_path).parts
    for part in parts:
        if part in EXCLUDE_DIRS or part.endswith('.framework'):
            return True
    return False

def should_include_file(file_path):
    """Check if file should be included."""
    path = Path(file_path)
    ext = path.suffix.lower()
    
    # Skip zip files
    if ext == '.zip':
        return False
    
    # Skip binary/compiled files
    if ext in SKIP_PATTERNS:
        return False
    
    # Include if extension matches
    if ext in ALL_EXTENSIONS:
        return True
    
    # Include files without extension that might be scripts (like 'gradlew', 'Makefile')
    if ext == '' and path.name in {'Makefile', 'Dockerfile', 'gradlew', 'LICENSE', 'VERSION', 'README'}:
        return True
    
    return False

def get_file_size(file_path):
    """Get file size, return 0 if error."""
    try:
        return os.path.getsize(file_path)
    except:
        return 0

def collect_files():
    """Collect all files to be zipped."""
    files = []
    
    for scan_dir in SCAN_DIRS:
        if not os.path.exists(scan_dir):
            continue
            
        for root, dirs, filenames in os.walk(scan_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not should_skip_dir(os.path.join(root, d))]
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                
                if should_include_file(file_path):
                    size = get_file_size(file_path)
                    if size > 0 and size < MAX_ZIP_SIZE:  # Skip empty and too-large files
                        files.append((file_path, size))
    
    return files

def create_zips(files):
    """Create zip files, each under MAX_ZIP_SIZE."""
    # Sort files by directory to keep related files together
    files.sort(key=lambda x: x[0])
    
    zip_number = 1
    current_zip_size = 0
    current_zip_files = []
    created_zips = []
    
    for file_path, file_size in files:
        # If adding this file would exceed limit, create zip and start new one
        if current_zip_size + file_size > MAX_ZIP_SIZE and current_zip_files:
            zip_path = create_single_zip(current_zip_files, zip_number)
            created_zips.append(zip_path)
            zip_number += 1
            current_zip_size = 0
            current_zip_files = []
        
        current_zip_files.append(file_path)
        current_zip_size += file_size
    
    # Create final zip with remaining files
    if current_zip_files:
        zip_path = create_single_zip(current_zip_files, zip_number)
        created_zips.append(zip_path)
    
    return created_zips

def create_single_zip(files, zip_number):
    """Create a single zip file."""
    zip_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}{zip_number}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            # Create archive name relative to home directory
            try:
                arcname = os.path.relpath(file_path, "/Users/seanburdges")
            except ValueError:
                arcname = os.path.basename(file_path)
            
            try:
                zf.write(file_path, arcname)
            except Exception as e:
                print(f"  Warning: Could not add {file_path}: {e}")
    
    return zip_path

def main():
    print("=" * 60)
    print("REPOSEAN ZIP CREATOR")
    print("=" * 60)
    print()
    
    print("Collecting files...")
    files = collect_files()
    
    total_size = sum(f[1] for f in files)
    print(f"Found {len(files)} files ({total_size / (1024*1024):.2f} MB total)")
    print()
    
    # Show breakdown by extension
    ext_counts = defaultdict(int)
    ext_sizes = defaultdict(int)
    for file_path, size in files:
        ext = Path(file_path).suffix.lower() or '(no ext)'
        ext_counts[ext] += 1
        ext_sizes[ext] += size
    
    print("File breakdown:")
    for ext in sorted(ext_counts.keys()):
        print(f"  {ext}: {ext_counts[ext]} files ({ext_sizes[ext] / (1024*1024):.2f} MB)")
    print()
    
    print("Creating zip files...")
    created_zips = create_zips(files)
    
    print()
    print("=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print()
    print(f"Created {len(created_zips)} zip files:")
    for zip_path in created_zips:
        size = os.path.getsize(zip_path)
        print(f"  {zip_path.name}: {size / (1024*1024):.2f} MB")
    print()
    print(f"Location: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
