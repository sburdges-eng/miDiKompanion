#!/bin/bash

# =============================================================================
# Bulling iOS App Build Script
# =============================================================================
# This script builds the Bulling iOS app using xcodebuild
#
# Requirements:
#   - macOS with Xcode 15.0+ installed
#   - Valid iOS SDK
#
# Usage:
#   ./build_ios_app.sh [simulator|device|archive]
#
# Examples:
#   ./build_ios_app.sh             # Build for simulator (default)
#   ./build_ios_app.sh simulator   # Build for iOS simulator
#   ./build_ios_app.sh device      # Build for physical device
#   ./build_ios_app.sh archive     # Create archive for distribution
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
IOS_DIR="${PROJECT_DIR}/iOS"
PROJECT_NAME="BullingApp"
SCHEME="Bulling"
BUILD_DIR="${PROJECT_DIR}/build/ios"

# Parse command line argument
BUILD_TYPE="${1:-simulator}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Bulling iOS App Builder${NC}"
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

cd "${IOS_DIR}"

case "${BUILD_TYPE}" in
    "simulator")
        echo -e "${YELLOW}Building for iOS Simulator...${NC}"
        echo ""

        # Build for simulator
        xcodebuild \
            -project "${PROJECT_NAME}.xcodeproj" \
            -scheme "${SCHEME}" \
            -sdk iphonesimulator \
            -configuration Debug \
            -derivedDataPath "${BUILD_DIR}/DerivedData" \
            build

        # Find the built app
        APP_PATH=$(find "${BUILD_DIR}/DerivedData" -name "Bulling.app" -type d | head -1)

        if [ -n "${APP_PATH}" ]; then
            echo ""
            echo -e "${GREEN}Build successful!${NC}"
            echo -e "App location: ${APP_PATH}"
            echo ""
            echo -e "${YELLOW}To run in simulator:${NC}"
            echo "  1. Open Simulator app"
            echo "  2. Drag the .app file into the simulator"
            echo "  OR"
            echo "  xcrun simctl install booted '${APP_PATH}'"
            echo "  xcrun simctl launch booted com.bulling.app"
        fi
        ;;

    "device")
        echo -e "${YELLOW}Building for iOS Device...${NC}"
        echo ""
        echo -e "${YELLOW}Note: You need a valid Apple Developer account and signing certificate.${NC}"
        echo ""

        # Build for device
        xcodebuild \
            -project "${PROJECT_NAME}.xcodeproj" \
            -scheme "${SCHEME}" \
            -sdk iphoneos \
            -configuration Release \
            -derivedDataPath "${BUILD_DIR}/DerivedData" \
            CODE_SIGN_IDENTITY="" \
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
            echo -e "${YELLOW}Note: To install on a device, you need to sign the app.${NC}"
        fi
        ;;

    "archive")
        echo -e "${YELLOW}Creating archive for distribution...${NC}"
        echo ""
        echo -e "${YELLOW}Note: You need a valid Apple Developer account for archiving.${NC}"
        echo ""

        ARCHIVE_PATH="${BUILD_DIR}/Bulling.xcarchive"

        # Create archive
        xcodebuild \
            -project "${PROJECT_NAME}.xcodeproj" \
            -scheme "${SCHEME}" \
            -sdk iphoneos \
            -configuration Release \
            -archivePath "${ARCHIVE_PATH}" \
            archive

        if [ -d "${ARCHIVE_PATH}" ]; then
            echo ""
            echo -e "${GREEN}Archive created successfully!${NC}"
            echo -e "Archive location: ${ARCHIVE_PATH}"
            echo ""
            echo -e "${YELLOW}To export IPA:${NC}"
            echo "  1. Open the archive in Xcode Organizer"
            echo "  2. Select 'Distribute App'"
            echo "  3. Choose distribution method (App Store, Ad Hoc, etc.)"
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
        echo "Usage: ./build_ios_app.sh [simulator|device|archive|clean]"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
