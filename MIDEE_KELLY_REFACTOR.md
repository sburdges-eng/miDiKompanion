# miDEE/KELLY Code Refactoring Reference

## Overview

This document provides a comprehensive reference for renaming and reorganizing the codebase:
- **miDEE**: All music-related modules and features
- **KELLY**: All emotion-related modules and features

## Current State Analysis

### Music Modules (to become miDEE)
```
midee/                      → midee/
├── __init__.py
├── cli.py
├── api.py
├── harmony.py
├── groove_engine.py
├── audio/
├── groove/
├── data/
│   ├── chord_progressions.json
│   ├── genre_pocket_maps.json
│   └── music_vernacular_database.md
├── structure/
├── session/
└── utils/

cpp_music_brain/                  → cpp_midee/
├── src/
│   ├── harmony/
│   ├── groove/
│   └── dsp/
└── tests/

penta_core_music-brain/           → penta_core_midee/
examples_music-brain/             → examples_midee/
docs_music-brain/                 → docs_midee/
tests_music-brain/                → tests_midee/
```

### Emotion Modules (to become KELLY)
```
kelly.thesaurus/                → kelly/thesaurus/
├── kelly.thesaurus.py          → thesaurus.py
├── angry.json
├── sad.json
├── happy.json
├── fear.json
├── disgust.json
├── surprise.json
├── blends.json
└── metadata.json

midee/emotion_api.py        → kelly/emotion_api.py
emotional_mapping.py              → kelly/emotional_mapping.py
auto_emotion_sampler.py           → kelly/emotion_sampler.py

python/penta_core/rules/emotion.py → kelly/rules/emotion.py
```

## Refactoring Steps

### Phase 1: Create New Directory Structure

```bash
# Create miDEE directories
mkdir -p midee/{audio,groove,data,structure,session,utils,arrangement,collaboration,daw,integrations,learning,orchestrator,agents}
mkdir -p cpp_midee/{src,tests,benchmarks}
mkdir -p penta_core_midee
mkdir -p examples_midee
mkdir -p docs_midee
mkdir -p tests_midee

# Create KELLY directories
mkdir -p kelly/{thesaurus,rules,presets,api}
mkdir -p kelly/data
mkdir -p tests_kelly
mkdir -p docs_kelly
```

### Phase 2: File Renaming Map

#### Music → miDEE

| Current Path | New Path | Type |
|-------------|----------|------|
| `midee/` | `midee/` | Directory |
| `cpp_music_brain/` | `cpp_midee/` | Directory |
| `penta_core_music-brain/` | `penta_core_midee/` | Directory |
| `examples_music-brain/` | `examples_midee/` | Directory |
| `docs_music-brain/` | `docs_midee/` | Directory |
| `tests_music-brain/` | `tests_midee/` | Directory |
| `pyproject_music-brain.toml` | `pyproject_midee.toml` | File |
| `README_music-brain.md` | `README_midee.md` | File |
| `LICENSE_music-brain` | `LICENSE_midee` | File |
| `.gitignore_music-brain` | `.gitignore_midee` | File |

#### Emotion → KELLY

| Current Path | New Path | Type |
|-------------|----------|------|
| `kelly.thesaurus/` | `kelly/thesaurus/` | Directory |
| `kelly.thesaurus/kelly.thesaurus.py` | `kelly/thesaurus/thesaurus.py` | File |
| `emotional_mapping.py` | `kelly/emotional_mapping.py` | File |
| `auto_emotion_sampler.py` | `kelly/emotion_sampler.py` | File |
| `midee/emotion_api.py` | `kelly/emotion_api.py` | File |

### Phase 3: Import Statement Updates

#### Python Imports - miDEE

Replace all occurrences:
```python
# OLD
from midee import ...
from midee.groove import ...
from midee.harmony import ...
import midee

# NEW
from midee import ...
from midee.groove import ...
from midee.harmony import ...
import midee
```

#### Python Imports - KELLY

Replace all occurrences:
```python
# OLD
from kelly.thesaurus import ...
from emotional_mapping import ...
import kelly.thesaurus

# NEW
from kelly.thesaurus import ...
from kelly.emotional_mapping import ...
import kelly
```

### Phase 4: C++ Namespace Updates

#### C++ Namespaces - miDEE

```cpp
// OLD
namespace midee {
namespace MusicBrain {

// NEW
namespace midee {
namespace miDEE {
```

#### C++ Include Guards

```cpp
// OLD
#ifndef MUSIC_BRAIN_HARMONY_H
#define MUSIC_BRAIN_HARMONY_H

// NEW
#ifndef MIDEE_HARMONY_H
#define MIDEE_HARMONY_H
```

### Phase 5: Configuration File Updates

#### pyproject.toml

```toml
# OLD
[project]
name = "music-brain"
name = "kelly"

# NEW
[project]
name = "midee"
name = "kelly"
```

#### package.json

```json
// OLD
{
  "name": "music-brain",
  "name": "kelly"
}

// NEW
{
  "name": "midee",
  "name": "kelly"
}
```

#### CMakeLists.txt

```cmake
# OLD
project(midee)
project(MusicBrain)

# NEW
project(midee)
project(miDEE)
```

### Phase 6: String Literals and Comments

#### Documentation Strings

```python
# OLD
"""miDEE - Emotion to music translation"""
"""KELLY Thesaurus - 216-node emotion mapping"""

# NEW
"""miDEE - Music generation and processing engine"""
"""KELLY - Emotion understanding and mapping system"""
```

#### Log Messages

```python
# OLD
logger.info("miDEE initialized")
logger.info("Emotion thesaurus loaded")

# NEW
logger.info("miDEE initialized")
logger.info("KELLY thesaurus loaded")
```

### Phase 7: Test File Updates

#### Test Imports

```python
# OLD
from midee.tests import ...
from tests_music-brain import ...

# NEW
from midee.tests import ...
from tests_midee import ...
```

### Phase 8: Documentation Updates

#### README Files

Update all references:
- `midee` → `midee`
- `miDEE` → `miDEE`
- `kelly.thesaurus` → `kelly.thesaurus`
- `KELLY Thesaurus` → `KELLY`

#### API Documentation

Update module names in:
- Docstrings
- API references
- Usage examples
- Installation instructions

## Search and Replace Patterns

### Case-Sensitive Replacements

```bash
# Python module names
midee → midee
kelly.thesaurus → kelly.thesaurus

# C++ namespaces
namespace midee → namespace midee
namespace MusicBrain → namespace miDEE

# Include guards
MUSIC_BRAIN_ → MIDEE_
EMOTION_THESAURUS_ → KELLY_THESAURUS_
```

### Case-Insensitive Replacements

```bash
# Documentation and comments
"miDEE" → "miDEE"
"music brain" → "miDEE"
"KELLY Thesaurus" → "KELLY"
"emotion thesaurus" → "KELLY"
```

### Regex Patterns

```regex
# Python imports
from midee\.(\w+) → from midee.\1
from kelly.thesaurus → from kelly.thesaurus

# Include statements
#include "midee/(\w+)" → #include "midee/\1"

# CMake targets
target_link_libraries.*midee → target_link_libraries.*midee
```

## Versioning Reset

### New Version Schema

After refactoring, reset versions to reflect the new architecture:

```
miDEE: 1.0.0
KELLY: 1.0.0
miDiKompanion (overall): 2.0.0 (major architectural change)
```

### Version File Updates

#### VERSION
```
2.0.0
```

#### pyproject_midee.toml
```toml
[project]
name = "midee"
version = "1.0.0"
description = "miDEE - Music generation and processing engine"
```

#### pyproject_kelly.toml
```toml
[project]
name = "kelly"
version = "1.0.0"
description = "KELLY - Emotion understanding and mapping system"
```

#### pyproject.toml (main)
```toml
[project]
name = "midiKompanion"
version = "2.0.0"
description = "miDiKompanion - Therapeutic iDAW with miDEE and KELLY"

[project.optional-dependencies]
midee = ["midee>=1.0.0"]
kelly = ["kelly>=1.0.0"]
```

#### Version.h (C++)
```cpp
#define IDAW_VERSION_MAJOR 2
#define IDAW_VERSION_MINOR 0
#define IDAW_VERSION_PATCH 0
#define IDAW_VERSION_STRING "2.0.0"

#define MIDEE_VERSION "1.0.0"
#define KELLY_VERSION "1.0.0"
```

## CLI Command Changes

### Old Commands
```bash
# miDEE
daiw extract drums.mid
daiw apply --genre funk track.mid
daiw analyze --chords song.mid

# Emotion
kelly list-emotions
kelly process "grief" --output out.mid
```

### New Commands
```bash
# miDEE
midee extract drums.mid
midee apply --genre funk track.mid
midee analyze --chords song.mid

# KELLY
kelly list-emotions
kelly process "grief" --output out.mid
```

## File Organization Structure

```
miDiKompanion/
├── midee/                    # Music generation engine
│   ├── __init__.py
│   ├── cli.py
│   ├── api.py
│   ├── harmony.py
│   ├── groove/
│   ├── audio/
│   ├── structure/
│   └── utils/
├── kelly/                    # Emotion understanding system
│   ├── __init__.py
│   ├── emotion_api.py
│   ├── emotional_mapping.py
│   ├── thesaurus/
│   │   ├── thesaurus.py
│   │   ├── angry.json
│   │   ├── sad.json
│   │   └── ...
│   └── rules/
├── iDAW_Core/               # Core DAW functionality
├── cpp_midee/               # C++ music engine
├── version_manager.py       # Version management
├── pyproject.toml           # Main config
├── pyproject_midee.toml     # miDEE config
├── pyproject_kelly.toml     # KELLY config
└── README.md                # Main documentation
```

## Migration Checklist

### Pre-Migration
- [ ] Backup current codebase
- [ ] Document all current imports
- [ ] List all external dependencies
- [ ] Identify breaking changes
- [ ] Update issue tracker

### Migration Phase 1: Directory Structure
- [ ] Create new directory structure
- [ ] Move files to new locations
- [ ] Update .gitignore files
- [ ] Update build configurations

### Migration Phase 2: Code Updates
- [ ] Update Python imports
- [ ] Update C++ includes and namespaces
- [ ] Update CMake configurations
- [ ] Update package configurations

### Migration Phase 3: Documentation
- [ ] Update README files
- [ ] Update API documentation
- [ ] Update user guides
- [ ] Update developer guides

### Migration Phase 4: Testing
- [ ] Run all tests
- [ ] Fix broken imports
- [ ] Fix broken tests
- [ ] Verify CLI commands

### Migration Phase 5: Versioning
- [ ] Reset version numbers
- [ ] Update VERSION file
- [ ] Update all pyproject.toml files
- [ ] Update Version.h
- [ ] Tag release with v2.0.0

### Post-Migration
- [ ] Update CI/CD pipelines
- [ ] Update deployment scripts
- [ ] Notify users of changes
- [ ] Update external documentation

## Automated Migration Scripts

### Script 1: Rename Directories

```bash
#!/bin/bash
# rename_directories.sh

# Music → miDEE
mv midee midee
mv cpp_music_brain cpp_midee
mv penta_core_music-brain penta_core_midee
mv examples_music-brain examples_midee
mv docs_music-brain docs_midee
mv tests_music-brain tests_midee

# Emotion → KELLY
mkdir -p kelly/thesaurus
mv kelly.thesaurus/* kelly/thesaurus/
mv emotional_mapping.py kelly/
mv auto_emotion_sampler.py kelly/emotion_sampler.py
```

### Script 2: Update Imports

```python
#!/usr/bin/env python3
# update_imports.py

import os
import re
from pathlib import Path

def update_file_imports(filepath):
    """Update imports in a Python file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Update midee → midee
    content = re.sub(r'from midee', 'from midee', content)
    content = re.sub(r'import midee', 'import midee', content)
    
    # Update kelly.thesaurus → kelly.thesaurus
    content = re.sub(r'from kelly.thesaurus', 'from kelly.thesaurus', content)
    content = re.sub(r'import kelly.thesaurus', 'import kelly.thesaurus', content)
    
    with open(filepath, 'w') as f:
        f.write(content)

def main():
    for py_file in Path('.').rglob('*.py'):
        if '.git' not in str(py_file):
            update_file_imports(py_file)
            print(f"Updated: {py_file}")

if __name__ == '__main__':
    main()
```

### Script 3: Update Version Files

```python
#!/usr/bin/env python3
# reset_versions.py

from version_manager import VersionManager, Version
from pathlib import Path

def reset_versions():
    """Reset all version files to 2.0.0."""
    manager = VersionManager(Path('.'))
    
    # Set main version to 2.0.0
    new_version = Version(2, 0, 0)
    manager.update_all(new_version)
    
    print("✓ Reset all versions to 2.0.0")

if __name__ == '__main__':
    reset_versions()
```

## Testing Strategy

### Unit Tests
- Test all renamed modules individually
- Verify imports work correctly
- Check for circular dependencies

### Integration Tests
- Test miDEE standalone
- Test KELLY standalone
- Test miDEE + KELLY integration

### CLI Tests
- Verify all CLI commands work
- Check output format consistency
- Test error handling

## Rollback Plan

If migration fails:

1. **Restore from backup**
   ```bash
   git checkout <pre-migration-commit>
   ```

2. **Document issues**
   - List all breaking changes
   - Note failed tests
   - Record import errors

3. **Incremental approach**
   - Migrate one module at a time
   - Test after each module
   - Commit working changes

## Success Criteria

- [ ] All tests passing
- [ ] No import errors
- [ ] CLI commands work
- [ ] Documentation updated
- [ ] Version files synchronized
- [ ] CI/CD pipelines green
- [ ] No regression in functionality

## Timeline Estimate

- **Phase 1-2**: 2-3 hours (Directory and file moves)
- **Phase 3-4**: 3-4 hours (Code updates)
- **Phase 5**: 2-3 hours (Documentation)
- **Phase 6**: 2-3 hours (Testing and fixes)
- **Total**: 9-13 hours

## Notes

- This is a **major version bump** (1.x.x → 2.0.0) due to breaking changes
- External users will need to update their imports
- Consider deprecation warnings in a transition period
- Update changelog with migration guide

---

**Document Version**: 1.0
**Created**: 2024-12-17
**Status**: Ready for implementation
