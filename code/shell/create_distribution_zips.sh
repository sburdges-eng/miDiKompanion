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
    
    # Check if scripts exist
    if [ ! -f "create_macos_zip.sh" ]; then
        echo "❌ create_macos_zip.sh not found"
        exit 1
    fi
    
    if [ ! -f "create_ios_zip.sh" ]; then
        echo "❌ create_ios_zip.sh not found"
        exit 1
    fi
    
    # Make scripts executable if needed
    [ ! -x "create_macos_zip.sh" ] && chmod +x create_macos_zip.sh
    [ ! -x "create_ios_zip.sh" ] && chmod +x create_ios_zip.sh
    
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
    
    # Check if iOS script exists
    if [ ! -f "create_ios_zip.sh" ]; then
        echo "❌ create_ios_zip.sh not found"
        exit 1
    fi
    
    # Make script executable if needed
    [ ! -x "create_ios_zip.sh" ] && chmod +x create_ios_zip.sh
    
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
    if find dist -name "*.zip" -print0 2>/dev/null | grep -qz .; then
        ls -lh dist/*.zip 2>/dev/null
    else
        echo "No zip files found"
    fi
    echo ""
    
    # Calculate total size
    if find dist -name "*.zip" -print0 2>/dev/null | grep -qz .; then
        total_size=$(du -ch dist/*.zip 2>/dev/null | tail -1 | cut -f1)
        echo "Total size: $total_size"
    fi
fi

echo ""
echo "✅ Done!"
echo ""
