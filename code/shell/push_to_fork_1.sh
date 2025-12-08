#!/bin/bash
# =============================================================================
# Push to 1DAWCURSOR Fork
# =============================================================================
# Automatically pushes to the fork once repository exists
# =============================================================================

set -euo pipefail

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

REPO_NAME="1DAWCURSOR"
REMOTE_NAME="1dawcursor"
BRANCH_NAME="1dawcursor/main"

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║              Push to 1DAWCURSOR Fork                          ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if remote exists
if ! git remote get-url "$REMOTE_NAME" &>/dev/null; then
    log_error "Remote '$REMOTE_NAME' not configured"
    log_info "Run: git remote add $REMOTE_NAME https://github.com/sburdges-eng/$REPO_NAME.git"
    exit 1
fi

# Check if repository exists
log_info "Checking if repository exists..."
if git ls-remote "$REMOTE_NAME" HEAD &>/dev/null; then
    log_success "Repository exists on GitHub"
else
    log_error "Repository not found on GitHub"
    echo ""
    echo "Please create the repository first:"
    echo "  1. Go to: https://github.com/new"
    echo "  2. Repository name: $REPO_NAME"
    echo "  3. Description: iDAW Standalone macOS Application - Cursor Fork"
    echo "  4. DO NOT initialize with README/gitignore/license"
    echo "  5. Click 'Create repository'"
    echo ""
    echo "Then run this script again:"
    echo "  ./scripts/push_to_fork.sh"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$BRANCH_NAME" ]; then
    log_info "Switching to branch $BRANCH_NAME..."
    git checkout "$BRANCH_NAME" 2>/dev/null || git checkout -b "$BRANCH_NAME"
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    log_warning "Uncommitted changes detected"
    git status --short
    echo ""
    read -p "Commit changes before pushing? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        git commit -m "chore: Update before push to fork"
    fi
fi

# Push to remote
log_info "Pushing to $REMOTE_NAME/$BRANCH_NAME..."
echo ""

if git push -u "$REMOTE_NAME" "$BRANCH_NAME"; then
    echo ""
    log_success "╔══════════════════════════════════════════════════════════════╗"
    log_success "║                    Push Successful!                          ║"
    log_success "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Repository: https://github.com/sburdges-eng/$REPO_NAME"
    echo "Branch: $BRANCH_NAME"
    echo ""
    COMMIT_COUNT=$(git rev-list --count HEAD)
    echo "Pushed: $COMMIT_COUNT commits"
    echo ""
    log_success "All work is now in the 1DAWCURSOR fork!"
    exit 0
else
    log_error "Push failed"
    exit 1
fi
