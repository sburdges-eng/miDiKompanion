# miDiKompanion - Multi-Component Build System

This repository contains three major components:

1. **Git Multi-Repository Updater** - Modular build system for Git batch operations
2. **Music Brain (DAiW/iDAW)** - Python toolkit for music analysis and composition  
3. **Penta Core** - C++/Python hybrid real-time music analysis engine

📖 **For complete build instructions for all components, see [MULTI_BUILD.md](MULTI_BUILD.md)**

---

## Quick Start: Build All Components

```bash
# Build everything
./build_all.sh

# Or build specific components
./build_all.sh --git-updater
./build_all.sh --music-brain
./build_all.sh --penta-core
```

---

## Git Multi-Repository Updater

Modular build system for creating customized Git repository batch update scripts.

### Quick Start

```bash
# Build standard version
make

# Run it
cd /path/to/your/repos
./dist/git-update.sh
```

## Workflow

See [WORKFLOW.md](WORKFLOW.md) for the canonical setup, build, test, and release flow.

### Core Setup (Required)

```bash
# 1. Install Python dependencies
pip install -e ".[dev]"

# 2. Install Node dependencies (for web interface)
npm install

# 3. Run tests to verify setup
pytest tests/python -v
npm run build
```

### Python CLI

```bash
# List available emotions
kelly list-emotions
## Usage

- `make` - Build standard version
- `make full` - Build full-featured version with all modules
- `make minimal` - Build minimal version
- `make install` - Install to ~/bin
- `make help` - Show all options

## Customization

Edit `profiles/custom.profile` and run `make custom`

## Project Structure

```
git-updater/
├── Makefile           # Build system
├── build.sh           # Build script
├── modules/           # Feature modules
│   ├── colors.sh
│   ├── config.sh
│   ├── verbose.sh
│   └── summary.sh
├── core/              # Core components
│   ├── header.sh
│   ├── main-loop.sh
│   └── footer.sh
├── profiles/          # Build profiles
│   ├── minimal.profile
│   ├── standard.profile
│   ├── full.profile
│   └── custom.profile
└── dist/              # Generated scripts
```

## Build Profiles

### Minimal
Core functionality only - no colors, no config file support.

### Standard (Recommended)
Includes colors and config file support. Best for most users.

### Full
All features: colors, config files, verbose mode, summary reports.

### Custom
Edit `profiles/custom.profile` to pick exactly what you need.

## Configuration File

Create `.git-update-config` in your repos directory:

```bash
BRANCHES=("main" "dev" "staging")
ROOT="/path/to/repos"
EXCLUDE=("old-repo" "archived-project")
```

---

## Other Components

This repository also includes:

### Music Brain (DAiW/iDAW)
Python toolkit for music production intelligence with groove extraction, chord analysis, and AI-assisted songwriting.

- **Documentation**: [README_music-brain.md](README_music-brain.md)
- **Build**: See [MULTI_BUILD.md](MULTI_BUILD.md)

### Penta Core
Professional-grade music analysis engine with hybrid C++/Python architecture for real-time performance.

- **Documentation**: [README_penta-core.md](README_penta-core.md), [BUILD.md](BUILD.md)
- **Build**: See [MULTI_BUILD.md](MULTI_BUILD.md)

---

## Complete Documentation

- **[BUILD_QUICK_REFERENCE.md](BUILD_QUICK_REFERENCE.md)** - One-page cheat sheet for quick builds
- **[MULTI_BUILD.md](MULTI_BUILD.md)** - Comprehensive guide for building all components
- **[README_music-brain.md](README_music-brain.md)** - Music Brain documentation
- **[README_penta-core.md](README_penta-core.md)** - Penta Core documentation
- **[BUILD.md](BUILD.md)** - Detailed Penta Core build instructions

## License

MIT License - See [LICENSE](LICENSE) for details.
