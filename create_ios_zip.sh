#!/bin/bash
#
# Script to create a distribution zip file of iOS source files
#
# This script:
# 1. Packages all iOS source files into a zip
# 2. Includes setup instructions
#

set -e  # Exit on error

echo "================================"
echo "Creating iOS Source Files Zip"
echo "================================"
echo ""

# Check if iOS directory exists
if [ ! -d "iOS/Bulling" ]; then
    echo "❌ iOS/Bulling directory not found"
    exit 1
fi

# Create a temporary directory for packaging
temp_dir=$(mktemp -d)
package_dir="$temp_dir/Bulling-iOS"

echo "📦 Packaging iOS source files..."

# Create package structure
mkdir -p "$package_dir"

# Copy iOS source files
cp -r iOS/Bulling/*.swift "$package_dir/"

# Copy documentation relevant to iOS
cp iOS_SETUP_GUIDE.md "$package_dir/"
cp README.md "$package_dir/"

# Create a simple setup instruction file
cat > "$package_dir/SETUP.txt" << 'EOF'
Bulling iOS App - Source Files
================================

Quick Setup:
1. Open Xcode
2. Create new iOS App project
3. Name it "Bulling"
4. Set deployment target to iOS 15.0+
5. Copy all .swift files from this folder to your project
6. Build and run!

For detailed instructions, see iOS_SETUP_GUIDE.md

Files included:
- BullingApp.swift - Main app entry point
- GameModel.swift - Game logic and scoring
- ContentView.swift - Main view coordinator
- GameView.swift - Game interface
- ScorecardView.swift - Score display
- SplashScreen.swift - Animated splash screen
- iOS_SETUP_GUIDE.md - Complete setup guide
- README.md - Project overview

App Features:
- Traditional 10-pin bowling scoring
- Multi-player support (up to 8 players)
- Beautiful bull-themed design
- Animated splash screen
- Real-time score tracking

For questions or issues, visit the repository.
EOF

# Create the zip file
cd "$temp_dir"
zip -r -q Bulling-iOS.zip Bulling-iOS
cd - > /dev/null

# Move zip to dist directory
mkdir -p dist
mv "$temp_dir/Bulling-iOS.zip" "dist/"

# Clean up
rm -rf "$temp_dir"

# Verify zip was created
if [ -f "dist/Bulling-iOS.zip" ]; then
    echo "   ✓ Zip created successfully!"
    echo ""
    echo "✅ SUCCESS!"
    echo ""
    echo "Distribution file: dist/Bulling-iOS.zip"
    echo ""
    echo "To use:"
    echo "  - Share dist/Bulling-iOS.zip"
    echo "  - Recipients: unzip and follow SETUP.txt instructions"
    echo ""
    
    # Show file size
    zip_size=$(du -h "dist/Bulling-iOS.zip" | cut -f1)
    echo "File size: $zip_size"
    echo ""
    
    # List contents
    echo "Contents:"
    unzip -l "dist/Bulling-iOS.zip" | tail -n +4 | head -n -2
    echo ""
else
    echo "❌ Failed to create zip file"
    exit 1
fi

echo "================================"
