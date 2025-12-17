#!/bin/bash
#
# Script to build the macOS app and create a distribution zip file
#
# This script:
# 1. Builds the Bulling.app using py2app
# 2. Creates a zip file for distribution
#

set -e  # Exit on error

echo "================================"
echo "Creating macOS Distribution Zip"
echo "================================"
echo ""

# Build the macOS app first
echo "🔨 Building macOS application..."

# Check if build script exists
if [ ! -f "build_macos_app.sh" ]; then
    echo "❌ build_macos_app.sh not found"
    exit 1
fi

if [ ! -x "build_macos_app.sh" ]; then
    echo "⚠️  build_macos_app.sh is not executable, making it executable..."
    chmod +x build_macos_app.sh
fi

./build_macos_app.sh
echo ""

# Check if build was successful
if [ ! -d "dist/Bulling.app" ]; then
    echo "❌ Build failed. Bulling.app not found in dist/"
    exit 1
fi

# Create zip file
echo "📦 Creating distribution zip..."
cd dist
zip -r -q Bulling-macOS.zip Bulling.app
cd ..

# Verify zip was created
if [ -f "dist/Bulling-macOS.zip" ]; then
    echo "   ✓ Zip created successfully!"
    echo ""
    echo "✅ SUCCESS!"
    echo ""
    echo "Distribution file: dist/Bulling-macOS.zip"
    echo ""
    echo "To distribute:"
    echo "  - Share dist/Bulling-macOS.zip"
    echo "  - Recipients: unzip and drag to Applications folder"
    echo ""
    # Show file size
    zip_size=$(du -h "dist/Bulling-macOS.zip" | cut -f1)
    echo "File size: $zip_size"
    echo ""
else
    echo "❌ Failed to create zip file"
    exit 1
fi

echo "================================"
