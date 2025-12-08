#!/bin/bash
# Complete 1DAWCURSOR Fork Setup - Non-interactive version
# Run this after creating the GitHub repository

set -e

echo "Setting up 1DAWCURSOR fork..."

# Check if remote already exists
if git remote get-url 1dawcursor 2>/dev/null; then
    echo "Remote '1dawcursor' already exists"
    git remote set-url 1dawcursor https://github.com/sburdges-eng/1DAWCURSOR.git
else
    echo "Adding remote '1dawcursor'..."
    git remote add 1dawcursor https://github.com/sburdges-eng/1DAWCURSOR.git
fi

# Ensure we're on the right branch
git checkout -b 1dawcursor/main 2>/dev/null || git checkout 1dawcursor/main

# Push to remote
echo "Pushing to 1DAWCURSOR repository..."
git push -u 1dawcursor 1dawcursor/main

echo ""
echo "✓ Fork setup complete!"
echo "Repository: https://github.com/sburdges-eng/1DAWCURSOR"
echo "Branch: 1dawcursor/main"
