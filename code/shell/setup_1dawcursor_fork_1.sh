#!/bin/bash
# =============================================================================
# Setup 1DAWCURSOR Fork Script
# =============================================================================
# This script prepares the repository for forking to 1DAWCURSOR
# It commits all work and sets up the new remote
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║            Setting up 1DAWCURSOR Fork                        ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Check for uncommitted changes
echo -e "${YELLOW}[1/5] Checking for uncommitted changes...${NC}"
if ! git diff-index --quiet HEAD --; then
    echo "  Found uncommitted changes"
    git status --short
    echo ""
    read -p "Commit all changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        git commit -m "feat: Complete standalone macOS app build system with tests

- Add comprehensive build script (build_macos_standalone.sh)
- Implement Python server management in Rust (python_server.rs)
- Create embedded Python launcher (start_api_embedded.py)
- Add Tauri integration for Python server lifecycle
- Implement comprehensive test suite (32 tests)
- Add test runner script and documentation
- Update Tauri configuration for standalone app
- Add build documentation and guides"
        echo -e "${GREEN}  ✓ Changes committed${NC}"
    else
        echo -e "${YELLOW}  ⚠ Skipping commit${NC}"
    fi
else
    echo -e "${GREEN}  ✓ Working tree clean${NC}"
fi

# Step 2: Create new branch for 1DAWCURSOR
echo -e "${YELLOW}[2/5] Creating 1DAWCURSOR branch...${NC}"
BRANCH_NAME="1dawcursor/main"
if git show-ref --verify --quiet refs/heads/$BRANCH_NAME; then
    echo "  Branch $BRANCH_NAME already exists, checking it out"
    git checkout $BRANCH_NAME
else
    git checkout -b $BRANCH_NAME
    echo -e "${GREEN}  ✓ Created branch $BRANCH_NAME${NC}"
fi

# Step 3: Add all new files
echo -e "${YELLOW}[3/5] Staging all files...${NC}"
git add -A
STAGED_COUNT=$(git diff --cached --numstat | wc -l | tr -d ' ')
echo "  Staged $STAGED_COUNT files"

# Step 4: Commit if there are changes
if [ "$STAGED_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}[4/5] Committing changes...${NC}"
    git commit -m "feat: Complete standalone macOS app build system

Components:
- Build script: build_macos_standalone.sh
- Python server management: src-tauri/src/python_server.rs
- Embedded launcher: music_brain/start_api_embedded.py
- Test suite: 32 tests across Rust and Python
- Documentation: BUILD_MACOS_README.md, README_TESTS.md
- Test runner: run_tests.sh

Features:
- Automated macOS app bundle creation
- Python runtime embedding
- C++ library integration
- Code signing support
- Comprehensive test coverage"
    echo -e "${GREEN}  ✓ Changes committed${NC}"
else
    echo -e "${GREEN}  ✓ No new changes to commit${NC}"
fi

# Step 5: Set up remote
echo -e "${YELLOW}[5/5] Setting up remote repository...${NC}"
echo ""
echo -e "${CYAN}To complete the fork setup, you need to:${NC}"
echo ""
echo "1. Create a new repository on GitHub named '1DAWCURSOR'"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: 1DAWCURSOR"
echo "   - Choose public or private"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. After creating the repository, run:"
echo ""
echo -e "${GREEN}   git remote add 1dawcursor https://github.com/sburdges-eng/1DAWCURSOR.git${NC}"
echo -e "${GREEN}   git push -u 1dawcursor $BRANCH_NAME${NC}"
echo ""
echo "3. Or use this script to set it up automatically:"
echo ""
echo -e "${GREEN}   ./setup_1dawcursor_fork.sh --push${NC}"
echo ""

# Check if --push flag is provided
if [[ "$1" == "--push" ]]; then
    # Check if remote exists
    if git remote get-url 1dawcursor 2>/dev/null; then
        echo -e "${YELLOW}Pushing to 1dawcursor remote...${NC}"
        git push -u 1dawcursor $BRANCH_NAME
        echo -e "${GREEN}✓ Pushed to 1dawcursor/$BRANCH_NAME${NC}"
    else
        echo -e "${RED}Error: 1dawcursor remote not found${NC}"
        echo "Please add the remote first:"
        echo "  git remote add 1dawcursor https://github.com/sburdges-eng/1DAWCURSOR.git"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete!                            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Current branch: $(git branch --show-current)"
echo "Ready to push to: 1DAWCURSOR repository"
echo ""
