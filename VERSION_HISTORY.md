# Version History

This file tracks major version changes for miDiKompanion.

## [1.1.0] - 2024-12-17

### Added
- Semantic versioning system with `version_manager.py` tool
- Automated version bumping (major, minor, patch)
- Version synchronization across all project files:
  - VERSION
  - pyproject.toml
  - package.json
  - iDAW_Core/include/Version.h
- Comprehensive versioning documentation (VERSIONING.md)
- Test suite for version management
- Automatic change detection for smart version bumping

### Changed
- Fixed VERSION file format from `1.0.04` to proper semver `1.1.0`
- Updated all project version files to use consistent semantic versioning

### Technical
- Version manager supports manual and automatic version bumps
- Git integration for change analysis
- Command-line interface for easy version management

---

## [1.0.4] - Previous

- Legacy version before semantic versioning system implementation
