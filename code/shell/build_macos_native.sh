#!/bin/bash

# =============================================================================
# Bulling macOS Native App Build Script
# =============================================================================
# This script builds the Bulling macOS app using xcodebuild (native SwiftUI)
#
# Requirements:
#   - macOS with Xcode 15.0+ installed
#   - Valid macOS SDK
#
# Usage:
#   ./build_macos_native.sh [debug|release|archive]
#
# Examples:
#   ./build_macos_native.sh           # Build Debug (default)
#   ./build_macos_native.sh debug     # Build Debug configuration
#   ./build_macos_native.sh release   # Build Release configuration
#   ./build_macos_native.sh archive   # Create archive for distribution
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project settings
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACOS_DIR="${PROJECT_DIR}/macOS"
PROJECT_NAME="BullingMac"
SCHEME="Bulling"
BUILD_DIR="${PROJECT_DIR}/build/macos"

# Parse command line argument
BUILD_TYPE="${1:-debug}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Bulling macOS Native App Builder${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check for Xcode
if ! command -v xcodebuild &> /dev/null; then
    echo -e "${RED}Error: xcodebuild not found!${NC}"
    echo "Please install Xcode from the Mac App Store."
    exit 1
fi

# Show Xcode version
echo -e "${YELLOW}Xcode version:${NC}"
xcodebuild -version
echo ""

# Create build directory
mkdir -p "${BUILD_DIR}"

cd "${MACOS_DIR}"

case "${BUILD_TYPE}" in
    "debug")
        echo -e "${YELLOW}Building Debug configuration...${NC}"
        echo ""

        # Build for macOS
        xcodebuild \
            -project "${PROJECT_NAME}.xcodeproj" \
            -scheme "${SCHEME}" \
            -configuration Debug \
            -derivedDataPath "${BUILD_DIR}/DerivedData" \
            CODE_SIGN_IDENTITY="-" \
            CODE_SIGNING_REQUIRED=NO \
            CODE_SIGNING_ALLOWED=NO \
            build

        # Find the built app
        APP_PATH=$(find "${BUILD_DIR}/DerivedData" -name "Bulling.app" -type d | head -1)

        if [ -n "${APP_PATH}" ]; then
            echo ""
            echo -e "${GREEN}Build successful!${NC}"
            echo -e "App location: ${APP_PATH}"
            echo ""
            echo -e "${YELLOW}To run the app:${NC}"
            echo "  open '${APP_PATH}'"
            echo ""
            echo -e "${YELLOW}To copy to Applications:${NC}"
            echo "  cp -R '${APP_PATH}' /Applications/"
        fi
        ;;

    "release")
        echo -e "${YELLOW}Building Release configuration...${NC}"
        echo ""

        # Build for macOS Release
        xcodebuild \
            -project "${PROJECT_NAME}.xcodeproj" \
            -scheme "${SCHEME}" \
            -configuration Release \
            -derivedDataPath "${BUILD_DIR}/DerivedData" \
            CODE_SIGN_IDENTITY="-" \
            CODE_SIGNING_REQUIRED=NO \
            CODE_SIGNING_ALLOWED=NO \
            build

        # Find the built app
        APP_PATH=$(find "${BUILD_DIR}/DerivedData" -name "Bulling.app" -type d | head -1)

        if [ -n "${APP_PATH}" ]; then
            # Copy to dist directory
            mkdir -p "${PROJECT_DIR}/dist"
            rm -rf "${PROJECT_DIR}/dist/Bulling.app"
            cp -R "${APP_PATH}" "${PROJECT_DIR}/dist/"

            echo ""
            echo -e "${GREEN}Build successful!${NC}"
            echo -e "App location: ${PROJECT_DIR}/dist/Bulling.app"
            echo ""
            echo -e "${YELLOW}To run the app:${NC}"
            echo "  open '${PROJECT_DIR}/dist/Bulling.app'"
            echo ""
            echo -e "${YELLOW}To distribute:${NC}"
            echo "  1. Zip the app: cd '${PROJECT_DIR}/dist' && zip -r Bulling-macOS.zip Bulling.app"
            echo "  2. Share the zip file"
        fi
        ;;

    "archive")
        echo -e "${YELLOW}Creating archive for distribution...${NC}"
        echo ""
        echo -e "${YELLOW}Note: For App Store distribution, you need a valid Apple Developer account.${NC}"
        echo ""

        ARCHIVE_PATH="${BUILD_DIR}/Bulling.xcarchive"

        # Create archive
        xcodebuild \
            -project "${PROJECT_NAME}.xcodeproj" \
            -scheme "${SCHEME}" \
            -configuration Release \
            -archivePath "${ARCHIVE_PATH}" \
            CODE_SIGN_IDENTITY="-" \
            archive

        if [ -d "${ARCHIVE_PATH}" ]; then
            echo ""
            echo -e "${GREEN}Archive created successfully!${NC}"
            echo -e "Archive location: ${ARCHIVE_PATH}"
            echo ""
            echo -e "${YELLOW}To export the app:${NC}"
            echo "  1. Open the archive in Xcode Organizer"
            echo "  2. Select 'Distribute App'"
            echo "  3. Choose 'Copy App' for direct distribution"
        fi
        ;;

    "clean")
        echo -e "${YELLOW}Cleaning build artifacts...${NC}"
        rm -rf "${BUILD_DIR}"
        xcodebuild \
            -project "${PROJECT_NAME}.xcodeproj" \
            -scheme "${SCHEME}" \
            clean
        echo -e "${GREEN}Clean complete!${NC}"
        ;;

    *)
        echo -e "${RED}Unknown build type: ${BUILD_TYPE}${NC}"
        echo ""
        echo "Usage: ./build_macos_native.sh [debug|release|archive|clean]"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
