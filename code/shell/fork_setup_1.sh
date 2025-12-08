#!/bin/bash
# =============================================================================
# 1DAWCURSOR Fork Setup Script
# =============================================================================
# Consolidated script for setting up the 1DAWCURSOR fork repository
# Handles repository creation, remote setup, and code push
# =============================================================================

set -euo pipefail

# Configuration
REPO_NAME="1DAWCURSOR"
REPO_OWNER="sburdges-eng"
REPO_FULL="${REPO_OWNER}/${REPO_NAME}"
BRANCH_NAME="1dawcursor/main"
REMOTE_NAME="1dawcursor"
REPO_URL="https://github.com/${REPO_FULL}.git"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }

# Header
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║           1DAWCURSOR Fork Setup                              ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Function: Check if repository exists on GitHub
check_repo_exists() {
    if command -v gh &> /dev/null; then
        if gh repo view "$REPO_FULL" &>/dev/null; then
            return 0
        fi
    fi
    # Try to check via git
    if git ls-remote "$REPO_URL" &>/dev/null; then
        return 0
    fi
    return 1
}

# Function: Create repository via GitHub CLI
create_repo_gh_cli() {
    log_info "Attempting to create repository via GitHub CLI..."
    if gh repo create "$REPO_FULL" \
        --public \
        --description "iDAW Standalone macOS Application - Cursor Fork" \
        --source=. \
        --remote="$REMOTE_NAME" 2>&1; then
        log_success "Repository created via GitHub CLI"
        return 0
    else
        log_warning "GitHub CLI creation failed (may need permissions)"
        return 1
    fi
}

# Function: Setup remote
setup_remote() {
    if git remote get-url "$REMOTE_NAME" &>/dev/null; then
        log_info "Remote '$REMOTE_NAME' already exists, updating URL..."
        git remote set-url "$REMOTE_NAME" "$REPO_URL"
    else
        log_info "Adding remote '$REMOTE_NAME'..."
        git remote add "$REMOTE_NAME" "$REPO_URL"
    fi
    log_success "Remote configured: $REPO_URL"
}

# Function: Ensure branch exists
ensure_branch() {
    if git show-ref --verify --quiet "refs/heads/$BRANCH_NAME"; then
        log_info "Branch '$BRANCH_NAME' exists, checking out..."
        git checkout "$BRANCH_NAME"
    else
        log_info "Creating branch '$BRANCH_NAME'..."
        git checkout -b "$BRANCH_NAME"
    fi
    log_success "On branch: $BRANCH_NAME"
}

# Function: Check for uncommitted changes
check_uncommitted() {
    if ! git diff-index --quiet HEAD --; then
        log_warning "Uncommitted changes detected:"
        git status --short | head -10
        echo ""
        read -p "Commit all changes? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git add -A
            git commit -m "chore: Update before fork setup

- Consolidate scripts and documentation
- Improve error handling and validation
- Update build system configuration"
            log_success "Changes committed"
            return 0
        else
            log_warning "Skipping commit"
            return 1
        fi
    fi
    return 0
}

# Function: Push to remote
push_to_remote() {
    log_info "Pushing to $REPO_FULL..."
    if git push -u "$REMOTE_NAME" "$BRANCH_NAME" 2>&1; then
        log_success "Code pushed successfully!"
        return 0
    else
        log_error "Push failed"
        return 1
    fi
}

# Main execution
main() {
    # Step 1: Check repository existence
    log_info "[1/5] Checking repository status..."
    if check_repo_exists; then
        log_success "Repository exists on GitHub"
    else
        log_warning "Repository does not exist yet"
        echo ""
        echo "To create the repository:"
        echo "  1. Go to: https://github.com/new"
        echo "  2. Repository name: $REPO_NAME"
        echo "  3. Description: iDAW Standalone macOS Application - Cursor Fork"
        echo "  4. DO NOT initialize with README/gitignore/license"
        echo "  5. Click 'Create repository'"
        echo ""
        
        # Try GitHub CLI if available
        if command -v gh &> /dev/null && gh auth status &>/dev/null; then
            read -p "Try creating via GitHub CLI? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                if ! create_repo_gh_cli; then
                    log_error "Please create repository manually and run this script again"
                    exit 1
                fi
            else
                log_error "Please create repository manually and run this script again"
                exit 1
            fi
        else
            log_error "Please create repository manually and run this script again"
            exit 1
        fi
    fi
    
    # Step 2: Check uncommitted changes
    log_info "[2/5] Checking for uncommitted changes..."
    check_uncommitted || log_warning "Proceeding with uncommitted changes"
    
    # Step 3: Ensure branch
    log_info "[3/5] Ensuring branch exists..."
    ensure_branch
    
    # Step 4: Setup remote
    log_info "[4/5] Setting up remote..."
    setup_remote
    
    # Step 5: Push
    log_info "[5/5] Pushing to remote..."
    if push_to_remote; then
        echo ""
        log_success "╔══════════════════════════════════════════════════════════════╗"
        log_success "║                    Fork Setup Complete!                      ║"
        log_success "╚══════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Repository: https://github.com/$REPO_FULL"
        echo "Branch: $BRANCH_NAME"
        echo ""
        return 0
    else
        log_error "Fork setup incomplete. Please check errors above."
        exit 1
    fi
}

# Run main function
main "$@"
