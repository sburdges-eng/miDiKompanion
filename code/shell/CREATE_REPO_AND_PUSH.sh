#!/bin/bash
# Create GitHub Repository and Push - Complete Workflow
# This script helps you create the repo and push in one go

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        1DAWCURSOR Repository Setup                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if GitHub CLI is available
if command -v gh &> /dev/null; then
    echo "✓ GitHub CLI found"
    echo ""
    read -p "Create repository '1DAWCURSOR' using GitHub CLI? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating repository..."
        gh repo create sburdges-eng/1DAWCURSOR --public --source=. --remote=1dawcursor --push || {
            echo "Repository might already exist or you need to authenticate"
            echo "Run: gh auth login"
            exit 1
        }
        echo "✓ Repository created and pushed!"
        exit 0
    fi
fi

# Manual instructions
echo "To create the repository manually:"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: 1DAWCURSOR"
echo "3. Description: 'iDAW Standalone macOS Application - Cursor Fork'"
echo "4. Choose Public or Private"
echo "5. DO NOT initialize with README, .gitignore, or license"
echo "6. Click 'Create repository'"
echo ""
echo "Then run:"
echo "  ./COMPLETE_FORK_SETUP.sh"
echo ""
echo "Or manually:"
echo "  git remote add 1dawcursor https://github.com/sburdges-eng/1DAWCURSOR.git"
echo "  git push -u 1dawcursor 1dawcursor/main"
