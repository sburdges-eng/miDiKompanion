# GitHub Copilot Instructions for DAiW-Music-Brain

## Permissions

Copilot is authorized to:
- ✅ **ADD** new files
- ✅ **MODIFY** existing files
- ✅ **RENAME** files
- ❌ **DELETE** files (requires human approval)

## Auto-Approval Rules

PRs from Copilot will be automatically approved if:
1. No files are deleted
2. All CI checks pass
3. Changes are additions or modifications only

PRs with file deletions will:
1. Be flagged with `needs-human-review` label
2. Block auto-approval
3. Require manual review before merge

## Project Context

This is the DAiW (Digital Audio intelligent Workstation) project - an emotion-first music generation system.

### Core Philosophy
- "Interrogate Before Generate"
- "Imperfection is Intentional"  
- "Every Rule-Break Needs Justification"

### Key Directories
- `music_brain/structure/` - Core data models and engines
- `music_brain/modules/` - Generation modules (chord, harmony)
- `music_brain/groove/` - Humanization and groove templates
- `music_brain/session/` - Intent processing, vernacular translation
- `tests/` - Unit tests

### Coding Standards
- Python 3.9+
- Type hints required
- Docstrings for all public functions
- Rich library for console output
- Typer for CLI commands
