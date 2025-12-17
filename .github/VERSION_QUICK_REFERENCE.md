# Version Management Quick Reference

## Quick Commands

```bash
# Check current version
./version_manager.py current

# Bug fix (1.2.3 → 1.2.4)
./version_manager.py bump-patch -m "Fix: description"

# New feature (1.2.3 → 1.3.0)
./version_manager.py bump-minor -m "Feature: description"

# Breaking change (1.2.3 → 2.0.0)
./version_manager.py bump-major -m "Breaking: description"

# Auto-detect (analyzes git changes)
./version_manager.py auto -m "Your commit message"
```

## When to Use Each Version Bump

### PATCH (X.Y.Z+1) - Bug Fixes
- Fixing bugs
- Performance improvements
- Documentation updates
- Code refactoring (no behavior change)
- Security patches
- Dependency updates (minor)

### MINOR (X.Y+1.0) - New Features
- New functionality
- New CLI commands
- New plugins or modules
- Significant enhancements
- New emotion presets
- API additions (non-breaking)

### MAJOR (X+1.0.0) - Breaking Changes
- Incompatible API changes
- Major architectural changes
- Breaking file format changes
- Build system overhauls
- Platform support changes

## Workflow

### Standard Workflow
```bash
# 1. Make your changes
git add <files>

# 2. Bump version appropriately
./version_manager.py bump-{patch|minor|major} -m "Description"

# 3. Commit version files
git add VERSION pyproject.toml package.json iDAW_Core/include/Version.h
git commit -m "chore: Bump version to X.Y.Z"

# 4. Tag the release (optional)
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

### Auto-Detection Workflow
```bash
# 1. Make your changes
git add <files>

# 2. Let the tool decide the bump type
./version_manager.py auto -m "Your commit message"

# 3. Commit everything together
git add VERSION pyproject.toml package.json iDAW_Core/include/Version.h
git commit -m "chore: Bump version based on changes"
```

## Files Updated

The version manager automatically updates:
1. `VERSION` - Main version file
2. `pyproject.toml` - Python package config
3. `package.json` - Node.js package config
4. `iDAW_Core/include/Version.h` - C++ header

## Commit Message Conventions

For auto-detection to work best:
- `fix:` - Bug fixes → patch bump
- `feat:` - New features → minor bump
- `BREAKING:` or `!` - Breaking changes → major bump

Examples:
```
fix: resolve MIDI timing drift
feat: add lo-fi emotion preset
feat!: redesign plugin API (BREAKING)
```

## Troubleshooting

### Versions out of sync
```bash
./version_manager.py current  # Check current
# Manually fix VERSION file if needed
./version_manager.py bump-patch -m "Sync versions"
```

### Wrong bump applied
```bash
git revert HEAD  # Revert the commit
# Try again with correct bump type
```

## See Also
- [VERSIONING.md](../VERSIONING.md) - Full documentation
- [VERSION_HISTORY.md](../VERSION_HISTORY.md) - Change history
- [Semantic Versioning](https://semver.org/)
