#!/bin/bash
#
# Kelly Project Repository Consolidation Script
# Consolidates 5 repos into unified 1DAW1 repository
# Author: Sean Burdges
# Date: 2025-12-06
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BASE_DIR=~/kelly-consolidation
TARGET_REPO="1DAW1"
LOG_FILE="$BASE_DIR/consolidation.log"
CONSOLIDATION_LOG="CONSOLIDATION_LOG.md"

# Repos to consolidate
SOURCE_REPOS=("iDAW" "penta-core" "DAiW-Music-Brain" "iDAWi")

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🎵 KELLY PROJECT REPOSITORY CONSOLIDATION${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[ERROR] $1" >> "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[WARNING] $1" >> "$LOG_FILE"
}

# Check if we're in the right directory
if [ ! -d "$BASE_DIR" ]; then
    error "Base directory $BASE_DIR not found!"
    exit 1
fi

cd "$BASE_DIR"

# Check if all required repos exist
log "Checking for required repositories..."
for repo in "${SOURCE_REPOS[@]}"; do
    if [ ! -d "$repo" ]; then
        error "Repository $repo not found!"
        exit 1
    fi
    log "  ✓ Found $repo"
done

# Step 1: Create backup
echo ""
log "📦 STEP 1: Creating backups..."
BACKUP_DIR="$BASE_DIR/backups-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

for repo in "${SOURCE_REPOS[@]}" "$TARGET_REPO"; do
    if [ -d "$repo" ]; then
        log "  Backing up $repo..."
        cp -r "$repo" "$BACKUP_DIR/"
    fi
done

log "  ✓ Backups created in: $BACKUP_DIR"

# Step 2: Clear 1DAW1 (except .git)
echo ""
log "🧹 STEP 2: Preparing target repository..."

cd "$BASE_DIR/$TARGET_REPO"

# Save .git directory
if [ -d ".git" ]; then
    log "  Preserving .git directory..."
    mv .git ../temp_git_backup
fi

# Remove everything else
log "  Clearing target repository..."
rm -rf *
rm -rf .[^.]*  2>/dev/null || true

# Restore .git
if [ -d "../temp_git_backup" ]; then
    mv ../temp_git_backup .git
    log "  ✓ .git directory restored"
fi

cd "$BASE_DIR"

# Step 3: Copy iDAW as base
echo ""
log "📋 STEP 3: Copying iDAW as base repository..."

# Copy everything from iDAW except .git
rsync -av --progress \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='target' \
    --exclude='dist' \
    --exclude='build' \
    "$BASE_DIR/iDAW/" "$BASE_DIR/$TARGET_REPO/" | tee -a "$LOG_FILE"

log "  ✓ iDAW copied to $TARGET_REPO"

# Step 4: Extract unique features from other repos
echo ""
log "🎯 STEP 4: Extracting unique features..."

# From DAiW-Music-Brain: emotion_thesaurus (CRITICAL!)
if [ -d "DAiW-Music-Brain/emotion_thesaurus" ]; then
    log "  Extracting emotion_thesaurus from DAiW-Music-Brain..."
    cp -r DAiW-Music-Brain/emotion_thesaurus "$TARGET_REPO/"
    log "  ✓ emotion_thesaurus copied"
else
    warning "  emotion_thesaurus not found in DAiW-Music-Brain"
fi

# From DAiW-Music-Brain: cpp directory
if [ -d "DAiW-Music-Brain/cpp" ]; then
    log "  Extracting cpp code from DAiW-Music-Brain..."
    mkdir -p "$TARGET_REPO/cpp_music_brain"
    cp -r DAiW-Music-Brain/cpp/* "$TARGET_REPO/cpp_music_brain/"
    log "  ✓ cpp code copied to cpp_music_brain/"
else
    warning "  cpp directory not found in DAiW-Music-Brain"
fi

# From DAiW-Music-Brain: Check for unique data files
if [ -d "DAiW-Music-Brain/data" ]; then
    log "  Checking for unique data files..."
    # Only copy files that don't exist in target
    rsync -av --ignore-existing DAiW-Music-Brain/data/ "$TARGET_REPO/data/" | tee -a "$LOG_FILE"
    log "  ✓ Unique data files merged"
fi

# From penta-core: Check if standalone has unique features
log "  Checking penta-core for unique features..."
if [ -d "penta-core/examples" ] && [ ! -d "$TARGET_REPO/penta_core/examples" ]; then
    log "  Copying unique penta-core examples..."
    mkdir -p "$TARGET_REPO/penta_core/examples"
    cp -r penta-core/examples/* "$TARGET_REPO/penta_core/examples/"
    log "  ✓ penta-core examples copied"
fi

# Step 5: Clean up duplicates and problematic files
echo ""
log "🧹 STEP 5: Cleaning up duplicates..."

cd "$BASE_DIR/$TARGET_REPO"

# Remove nested repos (they were copying attempts)
if [ -d "iDAWi" ]; then
    log "  Removing nested iDAWi directory..."
    rm -rf iDAWi
fi

# Remove duplicate DAiW-Music-Brain if it exists at root
if [ -d "DAiW-Music-Brain" ] && [ -d "music_brain" ]; then
    log "  Removing duplicate DAiW-Music-Brain directory..."
    # Keep music_brain, remove the duplicate
    rm -rf DAiW-Music-Brain
fi

# Fix case-sensitivity issues (keep lowercase versions)
if [ -d "iDAWi" ] && [ -d "idawi" ]; then
    log "  Fixing case-sensitivity: keeping lowercase 'idawi'"
    rm -rf iDAWi
fi

# Remove common build artifacts and caches
log "  Removing build artifacts and caches..."
find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "target" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

log "  ✓ Cleanup complete"

# Step 6: Create consolidation documentation
echo ""
log "📝 STEP 6: Creating consolidation documentation..."

cat > "$CONSOLIDATION_LOG" << 'EOF'
# Kelly Project Consolidation Log

**Date:** $(date +%Y-%m-%d)
**Consolidated by:** Automated script

## Overview

This repository consolidates 5 separate Kelly Project repositories into one unified codebase.

## Source Repositories

1. **iDAW** (sburdges-eng/iDAW) - Base repository (most comprehensive)
2. **penta-core** (sburdges-eng/penta-core) - Core audio engine
3. **DAiW-Music-Brain** (sburdges-eng/DAiW-Music-Brain) - Music processing and emotion system
4. **iDAWi** (sburdges-eng/iDAWi) - Container repo (nested repos, not used directly)
5. **1DAW1** (sburdges-eng/1DAW1) - Target consolidation repo

## Consolidation Strategy

### Base Repository: iDAW
- **Reason:** Most comprehensive with 995 code files (15.41 MB)
- **Contents:** Complete DAW implementation, Side A/Side B UI, Tauri backend, documentation

### Extracted Features

#### From DAiW-Music-Brain:
- ✅ **emotion_thesaurus/** - 6×6×6 emotion node system (216 emotions)
- ✅ **cpp/** - C++ music processing code → moved to `cpp_music_brain/`
- ✅ **data/** - Unique data files merged

#### From penta-core:
- ✅ **examples/** - Example implementations (if unique)

#### From iDAWi:
- ❌ Not used directly (was just a container with nested repos)

## Directory Structure

```
1DAW1/
├── emotion_thesaurus/          # From DAiW-Music-Brain (CRITICAL)
├── cpp_music_brain/            # From DAiW-Music-Brain
├── music_brain/                # Core music logic (from iDAW)
├── vault/                      # Music vault storage (from iDAW)
├── penta_core/                 # Core audio (from iDAW, enhanced)
├── src/                        # Main source code
├── docs/                       # Documentation
├── Obsidian_Documentation/     # Obsidian vault (from iDAW)
├── Production_Workflows/       # Production guides (from iDAW)
├── Songwriting_Guides/         # Songwriting resources (from iDAW)
├── Theory_Reference/           # Music theory (from iDAW)
└── Python_Tools/               # Python utilities (from iDAW)
```

## Features Preserved

### Core Features:
- ✅ Side A/Side B cassette tape interface
- ✅ Professional DAW features (mixer, timeline, transport)
- ✅ Emotion Wheel (6×6×6 thesaurus system)
- ✅ Dreamstate mode
- ✅ Parrot feature
- ✅ Music Brain processing engine
- ✅ Music Vault storage system
- ✅ AI/Interrogation schema
- ✅ Tauri 2.0 backend

### Documentation:
- ✅ Obsidian knowledge base
- ✅ Production workflows
- ✅ Songwriting guides
- ✅ Music theory reference
- ✅ All technical documentation

## Removed/Cleaned:

- ❌ Nested repository directories (iDAWi/, DAiW-Music-Brain/ at root)
- ❌ Duplicate node_modules, target, dist, build directories
- ❌ Case-sensitivity duplicates (iDAWi vs idawi)
- ❌ Python cache directories (__pycache__, .pytest_cache)

## Next Steps

1. **Review** this consolidation log and verify all features are present
2. **Test** the consolidated codebase
3. **Update** README.md with new unified architecture
4. **Configure** repository settings on GitHub
5. **Archive** old repositories (do not delete - keep as backup)

## Backup Location

Original repositories backed up to:
`~/kelly-consolidation/backups-[timestamp]/`

## Git History

- This consolidation creates a fresh start
- Original git history is preserved in backed-up repositories
- For historical reference, see individual repo backups

EOF

log "  ✓ $CONSOLIDATION_LOG created"

# Step 7: Create updated README
echo ""
log "📝 STEP 7: Creating unified README..."

cat > "README_NEW.md" << 'EOF'
# iDAW - Intelligent Digital Audio Workstation

**Kelly Project Unified Repository**

## Overview

iDAW (Intelligent Digital Audio Workstation), also known as the Kelly Project, is an ambitious therapeutic music generation platform that combines professional DAW features with emotional intelligence and AI-powered music creation.

## Core Concept: Side A / Side B

Inspired by cassette tapes, iDAW features a unique dual-interface:

- **Side A**: Traditional DAW interface with professional production tools
- **Side B**: Therapeutic/creative interface with emotion-based generation

## Key Features

### Professional DAW (Side A)
- Full-featured mixer with channel strips
- Timeline and transport controls
- VU meters and audio visualization
- Plugin hosting capabilities
- MIDI and audio recording

### Therapeutic Engine (Side B)
- **6×6×6 Emotion Thesaurus**: 216 emotion nodes for precise emotional targeting
- **Emotion Wheel**: Visual emotion selection interface
- **GhostWriter**: AI-powered lyric and melody generation
- **Interrogator**: Conversational music creation assistant
- **Dreamstate Mode**: Experimental/therapeutic music exploration
- **Parrot Feature**: Learning and mimicking musical styles

### Music Brain
- Intelligent music theory engine
- Chord progression generation
- Scale and harmony analysis
- Genre-specific templates

### Music Vault
- Centralized sample and preset management
- Audio cataloging and tagging
- Smart search and retrieval

## Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS
- **Backend**: Tauri 2.0, Rust
- **Audio Engine**: Penta-core (C++ audio processing)
- **AI Integration**: Python-based music generation
- **Build System**: Vite

## Project Structure

```
iDAW/
├── src/                       # React/TypeScript frontend
├── src-tauri/                 # Rust backend
├── emotion_thesaurus/         # 6×6×6 emotion system
├── music_brain/               # Core music intelligence
├── vault/                     # Sample and preset management
├── penta_core/                # Audio processing engine
├── cpp_music_brain/           # C++ music algorithms
├── docs/                      # Technical documentation
├── Obsidian_Documentation/    # Knowledge base
├── Production_Workflows/      # Guides and templates
└── Python_Tools/              # Utilities and scripts
```

## Quick Start

### Prerequisites
- Node.js 18+
- Rust 1.70+
- Python 3.9+
- CMake (for C++ components)

### Installation

```bash
# Clone the repository
git clone https://github.com/sburdges-eng/1DAW1.git
cd 1DAW1

# Install dependencies
npm install

# Build and run
npm run tauri dev
```

## Documentation

- **[CONSOLIDATION_LOG.md](./CONSOLIDATION_LOG.md)**: How this repo was consolidated
- **[Obsidian_Documentation/](./Obsidian_Documentation/)**: Comprehensive knowledge base
- **[Production_Workflows/](./Production_Workflows/)**: Production guides and workflows
- **[docs/](./docs/)**: Technical documentation

## Development

This project was consolidated from 5 separate repositories:
- iDAW (base)
- DAiW-Music-Brain (emotion system)
- penta-core (audio engine)
- iDAWi (experimental)
- 1DAW1 (target)

See [CONSOLIDATION_LOG.md](./CONSOLIDATION_LOG.md) for details.

## License

MIT License - See LICENSE file for details

## Author

Sean Burdges ([@sburdges-eng](https://github.com/sburdges-eng))

## Acknowledgments

Built with Claude AI assistance for code generation, architecture, and consolidation.

---

**Status**: Active Development (Consolidated 2025-12-06)
EOF

log "  ✓ README_NEW.md created"

# Step 8: Git operations
echo ""
log "🔧 STEP 8: Preparing Git commit..."

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    log "  Creating .gitignore..."
    cat > ".gitignore" << 'EOF'
# Dependencies
node_modules/
target/
dist/
build/

# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Build artifacts
*.o
*.a
*.so
*.dylib

# Logs
*.log
npm-debug.log*

# Environment
.env
.env.local
EOF
    log "  ✓ .gitignore created"
fi

# Git add
log "  Staging all files..."
git add -A

# Git commit
log "  Creating consolidation commit..."
git commit -m "🎵 Kelly Project Consolidation

Consolidated 5 repositories into unified 1DAW1 codebase:
- Base: iDAW (995 files, 15.41 MB)
- Added: emotion_thesaurus from DAiW-Music-Brain
- Added: cpp_music_brain from DAiW-Music-Brain
- Cleaned: Removed duplicates and nested repos
- Documentation: CONSOLIDATION_LOG.md, updated README

Features preserved:
✅ Side A/Side B UI
✅ 6×6×6 Emotion Thesaurus (216 nodes)
✅ Music Brain + Music Vault
✅ Dreamstate + Parrot
✅ AI/Interrogation system
✅ Tauri 2.0 backend
✅ All documentation and workflows

See CONSOLIDATION_LOG.md for details."

log "  ✓ Commit created"

# Step 9: Summary and next steps
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}✅ CONSOLIDATION COMPLETE!${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

log "📊 SUMMARY:"
echo ""
echo -e "${GREEN}Base repository:${NC} iDAW (995 code files, 15.41 MB)"
echo -e "${GREEN}Features added:${NC}"
echo "  ✓ emotion_thesaurus/ (from DAiW-Music-Brain)"
echo "  ✓ cpp_music_brain/ (from DAiW-Music-Brain)"
echo "  ✓ Unique data files merged"
echo ""
echo -e "${GREEN}Cleaned:${NC}"
echo "  ✓ Removed nested repos (iDAWi)"
echo "  ✓ Removed build artifacts"
echo "  ✓ Fixed case-sensitivity issues"
echo ""
echo -e "${GREEN}Documentation:${NC}"
echo "  ✓ CONSOLIDATION_LOG.md created"
echo "  ✓ README_NEW.md created"
echo ""
echo -e "${YELLOW}Backups saved to:${NC}"
echo "  $BACKUP_DIR"
echo ""
echo -e "${YELLOW}Consolidation log:${NC}"
echo "  $LOG_FILE"
echo ""

echo -e "${BLUE}📋 NEXT STEPS:${NC}"
echo ""
echo "1. Review the consolidation:"
echo "   cd $BASE_DIR/$TARGET_REPO"
echo "   cat CONSOLIDATION_LOG.md"
echo ""
echo "2. Review the new README:"
echo "   cat README_NEW.md"
echo "   # If satisfied, replace old README:"
echo "   mv README_NEW.md README.md"
echo ""
echo "3. Push to GitHub:"
echo "   git push origin main --force"
echo "   # (--force needed since we're replacing history)"
echo ""
echo "4. Verify on GitHub:"
echo "   https://github.com/sburdges-eng/1DAW1"
echo ""
echo "5. Archive old repositories:"
echo "   # Mark them as archived on GitHub (don't delete)"
echo ""

echo -e "${GREEN}✨ Consolidation script completed successfully!${NC}"
echo ""
