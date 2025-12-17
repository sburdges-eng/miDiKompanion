#!/bin/bash
#
# Icon Creator Helper Script for Dart Strike
#
# This script helps create a proper .icns icon file from a PNG image
# for the Dart Strike macOS application.
#

echo "================================"
echo "Bulling Icon Creator"
echo "================================"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script only works on macOS (requires iconutil)."
    exit 1
fi

# Check for input file
if [ $# -eq 0 ]; then
    echo "Usage: ./create_icon.sh <path-to-icon.png>"
    echo ""
    echo "Requirements:"
    echo "  - PNG image (ideally 1024×1024 pixels)"
    echo "  - Square aspect ratio"
    echo ""
    echo "Example:"
    echo "  ./create_icon.sh my_bowling_icon.png"
    exit 1
fi

input_file="$1"

# Check if input file exists
if [ ! -f "$input_file" ]; then
    echo "❌ Error: File '$input_file' not found."
    exit 1
fi

# Check if it's a PNG
if [[ "$input_file" != *.png ]]; then
    echo "⚠️  Warning: File should be a PNG image."
fi

echo "📥 Input file: $input_file"
echo ""

# Create iconset directory
iconset_dir="Bulling.iconset"
rm -rf "$iconset_dir"
mkdir "$iconset_dir"

echo "🎨 Creating icon sizes..."

# Check if sips is available (macOS image utility)
if ! command -v sips &> /dev/null; then
    echo "❌ Error: 'sips' command not found. This script requires macOS."
    exit 1
fi

# Generate all required icon sizes
sizes=(16 32 64 128 256 512 1024)

for size in "${sizes[@]}"; do
    echo "   Creating ${size}×${size}..."
    sips -z $size $size "$input_file" --out "$iconset_dir/icon_${size}x${size}.png" > /dev/null 2>&1
    
    # Create @2x versions (except for 1024)
    if [ $size -ne 1024 ]; then
        double_size=$((size * 2))
        echo "   Creating ${size}×${size}@2x (${double_size}×${double_size})..."
        sips -z $double_size $double_size "$input_file" --out "$iconset_dir/icon_${size}x${size}@2x.png" > /dev/null 2>&1
    fi
done

echo "   ✓ All sizes created"
echo ""

# Convert iconset to icns
echo "🔨 Converting to .icns format..."
iconutil -c icns "$iconset_dir" -o app_icon.icns

if [ -f "app_icon.icns" ]; then
    echo "   ✓ Created app_icon.icns"
    echo ""
    echo "✅ SUCCESS!"
    echo ""
    echo "Icon file created: app_icon.icns"
    echo ""
    echo "Next steps:"
    echo "  1. Rebuild the app: ./build_macos_app.sh"
    echo "  2. Your new icon will be used!"
    echo ""
    
    # Clean up iconset directory
    rm -rf "$iconset_dir"
    echo "🧹 Cleaned up temporary files"
else
    echo "❌ Error: Failed to create app_icon.icns"
    exit 1
fi

echo "================================"
