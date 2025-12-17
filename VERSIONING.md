# Semantic Versioning Guide

## Overview

miDiKompanion follows [Semantic Versioning 2.0.0](https://semver.org/) for all releases.

Version numbers use the format: **MAJOR.MINOR.PATCH**

## Version Number Components

### MAJOR Version (X.0.0)
Increment when making **breaking changes** or **major builds**:
- Incompatible API changes
- Major architectural changes
- Breaking changes to file formats
- Major feature sets that change core functionality
- Build system overhauls
- Platform support changes

**Example:** `1.0.0` → `2.0.0`

### MINOR Version (X.Y.0)
Increment when adding **new features** in a backward-compatible manner:
- New functionality
- New plugins or modules
- New CLI commands
- Significant enhancements to existing features
- New emotion presets or groove templates
- API additions (non-breaking)

**Example:** `1.2.0` → `1.3.0`

### PATCH Version (X.Y.Z)
Increment when making **bug fixes** and minor updates:
- Bug fixes
- Performance improvements
- Documentation updates
- Minor UI tweaks
- Code refactoring (no behavior change)
- Security patches
- Dependency updates

**Example:** `1.2.3` → `1.2.4`

## Version Management Tool

The project includes `version_manager.py` for consistent version updates across all files.

### Installation

The version manager is a standalone Python script in the repository root:

```bash
# Make executable (first time only)
chmod +x version_manager.py
```

### Usage

#### Check Current Version
```bash
./version_manager.py current
```

#### Manual Version Bumps
```bash
# Bump patch version (bug fixes)
./version_manager.py bump-patch -m "Fix MIDI timing issue"

# Bump minor version (new features)
./version_manager.py bump-minor -m "Add new emotion presets"

# Bump major version (breaking changes)
./version_manager.py bump-major -m "Redesign plugin architecture"
```

#### Automatic Version Detection
```bash
# Analyzes git changes and suggests appropriate bump
./version_manager.py auto -m "Your commit message"
```

The auto mode examines:
- Commit message keywords (feat, fix, breaking, major)
- Changed file types (source files vs tests)
- Build system files (CMakeLists.txt, etc.)

### What Gets Updated

The version manager synchronizes version numbers across:

1. **VERSION** - Main version file (plain text)
2. **pyproject.toml** - Python package configuration
3. **package.json** - Node.js package configuration
4. **iDAW_Core/include/Version.h** - C++ version header

All files are updated atomically to maintain consistency.

## Version Update Workflow

### For Bug Fixes
```bash
# 1. Make your bug fix
git add <files>

# 2. Bump patch version
./version_manager.py bump-patch -m "Fix: Description of bug fix"

# 3. Commit with version files
git add VERSION pyproject.toml package.json iDAW_Core/include/Version.h
git commit -m "chore: Bump version to X.Y.Z+1 - Fix: Description"
```

### For New Features
```bash
# 1. Implement your feature
git add <files>

# 2. Bump minor version
./version_manager.py bump-minor -m "Feature: Description of new feature"

# 3. Commit with version files
git add VERSION pyproject.toml package.json iDAW_Core/include/Version.h
git commit -m "chore: Bump version to X.Y+1.0 - Feature: Description"
```

### For Breaking Changes
```bash
# 1. Make your breaking changes
git add <files>

# 2. Bump major version
./version_manager.py bump-major -m "Breaking: Description of changes"

# 3. Commit with version files
git add VERSION pyproject.toml package.json iDAW_Core/include/Version.h
git commit -m "chore: Bump version to X+1.0.0 - Breaking: Description"
```

## Version History

Current version tracking is maintained in:
- `VERSION` file - Canonical source
- `CHANGELOG.md` - Human-readable history
- Git tags - Release markers

### Tagging Releases

After bumping version and committing:

```bash
# Create annotated tag
git tag -a v1.2.3 -m "Release version 1.2.3"

# Push tag to remote
git push origin v1.2.3
```

## Pre-release Versions

For development versions, append metadata:

- **Alpha:** `1.2.3-alpha.1`
- **Beta:** `1.2.3-beta.1`
- **Release Candidate:** `1.2.3-rc.1`

Example:
```bash
# Manually update VERSION file for pre-release
echo "1.3.0-alpha.1" > VERSION
./version_manager.py current  # Verify
```

## Changelog Maintenance

Update `CHANGELOG.md` (or appropriate changelog file) with each version bump:

```markdown
## [1.2.3] - 2024-01-15

### Fixed
- Fixed MIDI timing issue in groove engine
- Resolved memory leak in audio processor

### Changed
- Improved performance of emotion detection

### Added
- Nothing in this release
```

### Changelog Sections
- **Added** - New features
- **Changed** - Changes to existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security fixes

## Best Practices

1. **Always bump version before release** - Never release with a version that's already been released
2. **One version bump per release** - Don't bump multiple times in one release cycle
3. **Update all version files together** - Use `version_manager.py` to ensure consistency
4. **Tag releases in git** - Makes it easy to track and rollback
5. **Document changes in changelog** - Help users understand what changed
6. **Use descriptive commit messages** - Helps with automatic version detection
7. **Test before bumping major versions** - Breaking changes need extra care

## Commit Message Conventions

To help with automatic version detection, use conventional commits:

- `fix:` - Bug fixes (patch bump)
- `feat:` - New features (minor bump)
- `BREAKING:` or `!` - Breaking changes (major bump)

Examples:
```bash
git commit -m "fix: resolve MIDI note timing drift"
git commit -m "feat: add new lo-fi emotion preset"
git commit -m "feat!: redesign plugin API (BREAKING CHANGE)"
```

## Version File Formats

### VERSION (Plain Text)
```
1.2.3
```

### pyproject.toml (TOML)
```toml
[project]
name = "kelly"
version = "1.2.3"
```

### package.json (JSON)
```json
{
  "name": "idaw",
  "version": "1.2.3"
}
```

### Version.h (C++ Header)
```cpp
#define IDAW_VERSION_MAJOR 1
#define IDAW_VERSION_MINOR 2
#define IDAW_VERSION_PATCH 3
#define IDAW_VERSION_STRING "1.2.3"
```

## Troubleshooting

### Version files out of sync
```bash
# Get current version from VERSION file
./version_manager.py current

# Update all files to match
./version_manager.py bump-patch -m "Sync version files"
```

### Need to rollback a version
```bash
# 1. Edit VERSION file manually
echo "1.2.2" > VERSION

# 2. Update all files to match
./version_manager.py bump-patch -m "Rollback and sync"
# This will set to 1.2.3, so edit VERSION again:
echo "1.2.2" > VERSION
```

Or use git to revert:
```bash
git revert <commit-with-wrong-version>
```

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Keep a Changelog](https://keepachangelog.com/)
