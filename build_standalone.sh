#!/bin/bash

# =============================================================================
# Bulling Standalone App Builder
# =============================================================================
# Creates standalone executables for immediate private distribution
# No Apple Developer account required - unsigned builds for direct sharing
#
# Usage:
#   ./build_standalone.sh [ios|macos|all]
#
# Examples:
#   ./build_standalone.sh           # Build both iOS and macOS
#   ./build_standalone.sh macos     # Build macOS only
#   ./build_standalone.sh ios       # Build iOS simulator only
#   ./build_standalone.sh all       # Build everything
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project settings
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="${PROJECT_DIR}/dist"
BUILD_DIR="${PROJECT_DIR}/build"

# Parse command line argument
BUILD_TARGET="${1:-all}"

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}     ${CYAN}Bulling Standalone App Builder${NC}                       ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}     ${YELLOW}Private Distribution - No Signing Required${NC}          ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC}     ${RED}⚠️  PERSONAL USE ONLY - No Commercial Use${NC}           ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}This builds unsigned apps for personal use only.${NC}"
    echo -e "${YELLOW}See LICENSE.txt for terms and conditions.${NC}"
    echo ""
}

check_requirements() {
    echo -e "${YELLOW}Checking requirements...${NC}"

    if ! command -v xcodebuild &> /dev/null; then
        echo -e "${RED}Error: Xcode is required but not installed.${NC}"
        echo "Please install Xcode from the Mac App Store."
        exit 1
    fi

    echo -e "${GREEN}✓ Xcode found${NC}"
    xcodebuild -version
    echo ""
}

build_macos() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Building macOS Standalone App${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    local MACOS_BUILD_DIR="${BUILD_DIR}/macos"
    mkdir -p "${MACOS_BUILD_DIR}"
    mkdir -p "${DIST_DIR}"

    cd "${PROJECT_DIR}/macOS"

    echo -e "${YELLOW}Building Release configuration...${NC}"

    xcodebuild \
        -project BullingMac.xcodeproj \
        -scheme Bulling \
        -configuration Release \
        -derivedDataPath "${MACOS_BUILD_DIR}/DerivedData" \
        CODE_SIGN_IDENTITY="-" \
        CODE_SIGNING_REQUIRED=NO \
        CODE_SIGNING_ALLOWED=NO \
        ONLY_ACTIVE_ARCH=NO \
        build 2>&1 | grep -E "(error:|warning:|Building|Linking|Compiling|BUILD)"

    # Find and copy the built app
    APP_PATH=$(find "${MACOS_BUILD_DIR}/DerivedData" -name "Bulling.app" -type d | head -1)

    if [ -n "${APP_PATH}" ] && [ -d "${APP_PATH}" ]; then
        rm -rf "${DIST_DIR}/Bulling-macOS.app"
        cp -R "${APP_PATH}" "${DIST_DIR}/Bulling-macOS.app"

        # Create distribution zip
        cd "${DIST_DIR}"
        rm -f Bulling-macOS-Standalone.zip
        zip -r -q Bulling-macOS-Standalone.zip Bulling-macOS.app

        echo ""
        echo -e "${GREEN}✓ macOS app built successfully!${NC}"
        echo -e "  App: ${DIST_DIR}/Bulling-macOS.app"
        echo -e "  ZIP: ${DIST_DIR}/Bulling-macOS-Standalone.zip"
    else
        echo -e "${RED}✗ macOS build failed${NC}"
        return 1
    fi

    cd "${PROJECT_DIR}"
}

build_ios_simulator() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Building iOS Simulator App${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    local IOS_BUILD_DIR="${BUILD_DIR}/ios"
    mkdir -p "${IOS_BUILD_DIR}"
    mkdir -p "${DIST_DIR}"

    cd "${PROJECT_DIR}/iOS"

    echo -e "${YELLOW}Building for iOS Simulator...${NC}"

    xcodebuild \
        -project BullingApp.xcodeproj \
        -scheme Bulling \
        -sdk iphonesimulator \
        -configuration Release \
        -derivedDataPath "${IOS_BUILD_DIR}/DerivedData" \
        CODE_SIGN_IDENTITY="" \
        CODE_SIGNING_REQUIRED=NO \
        CODE_SIGNING_ALLOWED=NO \
        ONLY_ACTIVE_ARCH=NO \
        build 2>&1 | grep -E "(error:|warning:|Building|Linking|Compiling|BUILD)"

    # Find and copy the built app
    APP_PATH=$(find "${IOS_BUILD_DIR}/DerivedData" -name "Bulling.app" -type d | head -1)

    if [ -n "${APP_PATH}" ] && [ -d "${APP_PATH}" ]; then
        rm -rf "${DIST_DIR}/Bulling-iOS-Simulator.app"
        cp -R "${APP_PATH}" "${DIST_DIR}/Bulling-iOS-Simulator.app"

        # Create distribution zip
        cd "${DIST_DIR}"
        rm -f Bulling-iOS-Simulator.zip
        zip -r -q Bulling-iOS-Simulator.zip Bulling-iOS-Simulator.app

        echo ""
        echo -e "${GREEN}✓ iOS Simulator app built successfully!${NC}"
        echo -e "  App: ${DIST_DIR}/Bulling-iOS-Simulator.app"
        echo -e "  ZIP: ${DIST_DIR}/Bulling-iOS-Simulator.zip"
        echo ""
        echo -e "${YELLOW}To install on simulator:${NC}"
        echo "  xcrun simctl install booted '${DIST_DIR}/Bulling-iOS-Simulator.app'"
        echo "  xcrun simctl launch booted com.bulling.app"
    else
        echo -e "${RED}✗ iOS Simulator build failed${NC}"
        return 1
    fi

    cd "${PROJECT_DIR}"
}

build_ios_device() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Building iOS Device App (Unsigned)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    local IOS_BUILD_DIR="${BUILD_DIR}/ios-device"
    mkdir -p "${IOS_BUILD_DIR}"
    mkdir -p "${DIST_DIR}"

    cd "${PROJECT_DIR}/iOS"

    echo -e "${YELLOW}Building for iOS Device (unsigned)...${NC}"
    echo -e "${YELLOW}Note: This creates an unsigned .app for sideloading${NC}"
    echo ""

    xcodebuild \
        -project BullingApp.xcodeproj \
        -scheme Bulling \
        -sdk iphoneos \
        -configuration Release \
        -derivedDataPath "${IOS_BUILD_DIR}/DerivedData" \
        CODE_SIGN_IDENTITY="" \
        CODE_SIGNING_REQUIRED=NO \
        CODE_SIGNING_ALLOWED=NO \
        ONLY_ACTIVE_ARCH=NO \
        build 2>&1 | grep -E "(error:|warning:|Building|Linking|Compiling|BUILD)"

    # Find and copy the built app
    APP_PATH=$(find "${IOS_BUILD_DIR}/DerivedData" -name "Bulling.app" -type d | head -1)

    if [ -n "${APP_PATH}" ] && [ -d "${APP_PATH}" ]; then
        rm -rf "${DIST_DIR}/Bulling-iOS-Device.app"
        cp -R "${APP_PATH}" "${DIST_DIR}/Bulling-iOS-Device.app"

        # Create IPA-style package
        cd "${DIST_DIR}"
        rm -rf Payload
        mkdir -p Payload
        cp -R Bulling-iOS-Device.app Payload/
        rm -f Bulling-iOS-Unsigned.ipa
        zip -r -q Bulling-iOS-Unsigned.ipa Payload
        rm -rf Payload

        echo ""
        echo -e "${GREEN}✓ iOS Device app built successfully!${NC}"
        echo -e "  App: ${DIST_DIR}/Bulling-iOS-Device.app"
        echo -e "  IPA: ${DIST_DIR}/Bulling-iOS-Unsigned.ipa"
        echo ""
        echo -e "${YELLOW}To install on device:${NC}"
        echo "  Use a sideloading tool like AltStore, Sideloadly, or similar"
        echo "  Or sign with your developer certificate and install via Xcode"
    else
        echo -e "${RED}✗ iOS Device build failed${NC}"
        return 1
    fi

    cd "${PROJECT_DIR}"
}

print_summary() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}                    ${GREEN}Build Complete!${NC}                       ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Distribution files in: ${DIST_DIR}${NC}"
    echo ""
    ls -la "${DIST_DIR}" 2>/dev/null | grep -E "Bulling" || true
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║${NC}                  ${YELLOW}⚠️  IMPORTANT NOTICE${NC}                    ${RED}║${NC}"
    echo -e "${RED}║${NC}                                                          ${RED}║${NC}"
    echo -e "${RED}║${NC}  ${YELLOW}These apps are for PERSONAL USE ONLY${NC}                 ${RED}║${NC}"
    echo -e "${RED}║${NC}  ${YELLOW}Do NOT publish to app stores or use commercially${NC}     ${RED}║${NC}"
    echo -e "${RED}║${NC}  ${YELLOW}See LICENSE.txt for complete terms${NC}                   ${RED}║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Distribution Instructions:${NC}"
    echo ""
    echo -e "  ${GREEN}macOS:${NC}"
    echo "    1. Share Bulling-macOS-Standalone.zip with friends/family"
    echo "    2. Recipients: Unzip and move to Applications"
    echo "    3. First run: Right-click → Open (to bypass Gatekeeper)"
    echo "    4. Remind them: Personal use only!"
    echo ""
    echo -e "  ${GREEN}iOS Simulator:${NC}"
    echo "    1. Open Simulator app on Mac"
    echo "    2. Drag Bulling-iOS-Simulator.app into simulator"
    echo "    OR use: xcrun simctl install booted dist/Bulling-iOS-Simulator.app"
    echo ""
    echo -e "  ${GREEN}iOS Device (Personal Sideloading):${NC}"
    echo "    1. Use Bulling-iOS-Unsigned.ipa with sideloading tools"
    echo "    2. Options: AltStore, Sideloadly, or sign with your certificate"
    echo "    3. NOT for App Store - personal devices only!"
    echo ""
    echo -e "${CYAN}📖 For detailed instructions, see PERSONAL_USE_README.md${NC}"
    echo ""
}

# Main execution
print_header
check_requirements

case "${BUILD_TARGET}" in
    "macos")
        build_macos
        ;;
    "ios")
        build_ios_simulator
        ;;
    "ios-device")
        build_ios_device
        ;;
    "all")
        build_macos || true
        echo ""
        build_ios_simulator || true
        echo ""
        build_ios_device || true
        ;;
    *)
        echo -e "${RED}Unknown target: ${BUILD_TARGET}${NC}"
        echo ""
        echo "Usage: ./build_standalone.sh [macos|ios|ios-device|all]"
        exit 1
        ;;
esac

print_summary
