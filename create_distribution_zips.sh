#!/bin/bash
#
# Master script to create both macOS and iOS distribution zip files
#
# This script creates:
# 1. Bulling-macOS.zip - macOS application bundle (requires macOS to build)
# 2. Bulling-iOS.zip - iOS source files package
#

set -e  # Exit on error

echo "======================================="
echo "Creating Distribution Zip Files"
echo "======================================="
echo ""

# Check if running on macOS for the macOS build
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🖥️  Detected macOS - will create both zips"
    echo ""
    
    # Create macOS zip
    echo "Creating macOS distribution zip..."
    ./create_macos_zip.sh
    echo ""
    
    # Create iOS zip
    echo "Creating iOS source files zip..."
    ./create_ios_zip.sh
    echo ""
    
else
    echo "⚠️  Not running on macOS - can only create iOS zip"
    echo "   (macOS app build requires macOS)"
    echo ""
    
    # Only create iOS zip
    echo "Creating iOS source files zip..."
    ./create_ios_zip.sh
    echo ""
fi

echo "======================================="
echo "Distribution Files Created"
echo "======================================="
echo ""

# Show what was created
if [ -d "dist" ]; then
    echo "Files in dist/ directory:"
    ls -lh dist/*.zip 2>/dev/null || echo "No zip files found"
    echo ""
    
    # Calculate total size
    if ls dist/*.zip &> /dev/null; then
        total_size=$(du -ch dist/*.zip | tail -1 | cut -f1)
        echo "Total size: $total_size"
    fi
fi

echo ""
echo "✅ Done!"
echo ""
