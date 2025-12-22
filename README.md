# Git Multi-Repository Updater

Modular build system for creating customized Git repository batch update scripts.

## Quick Start

```bash
# Build standard version
make

# Run it
cd /path/to/your/repos
./dist/git-update.sh
```

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