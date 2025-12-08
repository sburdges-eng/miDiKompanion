#!/bin/bash
# Run iOS tests for DAiW
#
# Usage:
#   ./ios/run_tests.sh
#   ./ios/run_tests.sh --device "iPhone 15 Pro"
#   ./ios/run_tests.sh --simulator "iPhone 15"

set -e

# Default values
DESTINATION="platform=iOS Simulator,name=iPhone 15"
SCHEME="DAiW"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DESTINATION="platform=iOS,name=$2"
            shift 2
            ;;
        --simulator)
            DESTINATION="platform=iOS Simulator,name=$2"
            shift 2
            ;;
        --scheme)
            SCHEME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running iOS tests..."
echo "Scheme: $SCHEME"
echo "Destination: $DESTINATION"
echo ""

# Check if Xcode is available
if ! command -v xcodebuild &> /dev/null; then
    echo "Error: xcodebuild not found. Install Xcode from the App Store."
    exit 1
fi

# Check if project exists
if [ ! -f "ios/DAiW.xcodeproj/project.pbxproj" ]; then
    echo "Warning: Xcode project not found. Creating basic structure..."
    echo "You'll need to create the project in Xcode first."
    exit 1
fi

# Run tests
xcodebuild test \
    -project ios/DAiW.xcodeproj \
    -scheme "$SCHEME" \
    -destination "$DESTINATION" \
    -only-testing:DAiWTests \
    | xcpretty

echo ""
echo "Tests completed!"

