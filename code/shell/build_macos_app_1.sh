#!/bin/bash
#
# Build script for creating the Dart Strike macOS application
#
# This script automates the process of building a standalone .app bundle
# that can be distributed and run on macOS without any dependencies.
#

set -e  # Exit on error

echo "================================"
echo "Bulling macOS App Builder"
echo "================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is designed for macOS."
    echo "   It may not work correctly on other platforms."
    echo ""
fi

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install --upgrade -r requirements.txt
echo "   ✓ Dependencies installed"
echo ""

# Create app icon if it doesn't exist
if [ ! -f "app_icon.icns" ]; then
    echo "🎨 Creating app icon..."
    # Create a simple icon using iconutil (macOS only)
    # For now, we'll create a placeholder that can be replaced
    mkdir -p Bulling.iconset
    
    # Create a simple 1024x1024 PNG (placeholder - will be replaced with actual icon)
    # Users can replace app_icon.icns with their own icon file
    
    echo "   ⚠️  Using placeholder icon. Replace app_icon.icns with your own icon."
    echo ""
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build dist
echo "   ✓ Cleaned"
echo ""

# Build the app
echo "🔨 Building macOS application..."
python3 setup.py py2app
echo "   ✓ Build complete!"
echo ""

# Check if build was successful
if [ -d "dist/Bulling.app" ]; then
    echo "✅ SUCCESS!"
    echo ""
    echo "Your app is ready: dist/Bulling.app"
    echo ""
    echo "To install:"
    echo "  1. Copy 'Bulling.app' to your Applications folder"
    echo "  2. Double-click to run"
    echo ""
    echo "To distribute:"
    echo "  - Zip the .app file"
    echo "  - Share the zip file"
    echo "  - Recipients: unzip and copy to Applications"
    echo ""
else
    echo "❌ Build failed. Check errors above."
    exit 1
fi

echo "================================"
