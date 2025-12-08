---
name: Fix Test Config and Imports
overview: Fix the test infrastructure to resolve the "Normalize imports + test all modules" task from the DAiW Task Board. This includes updating pyproject.toml, fixing import errors in test_core_modules.py, and adding missing function aliases.
todos:
  - id: fix-pyproject
    content: Update pyproject.toml testpaths from tests to tests_music-brain
    status: completed
  - id: fix-ci-workflow
    content: Update test.yml Python test path to tests_music-brain/
    status: in_progress
  - id: add-function-alias
    content: Add list_genre_templates alias in groove/templates.py
    status: pending
  - id: fix-boom-bap
    content: Add boom-bap template support or fix test expectation
    status: pending
  - id: verify-tests
    content: Run pytest to verify tests collect and pass
    status: pending
---

# Fix Test Configuration and Import Normalization

## Problem Summary
The CI workflow and pyproject.toml reference `tests/` but Python tests are in `tests_music-brain/`. Additionally, `test_core_modules.py` has import errors for non-existent functions.

## Files to Modify

### 1. Update `pyproject.toml` test configuration
Change testpaths from `["tests"]` to `["tests_music-brain"]`:

```toml:102:104:pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests_music-brain"]
```

### 2. Update `.github/workflows/test.yml` Python test path
Change line 408 from `tests/` to `tests_music-brain/`:

```yaml
pytest tests_music-brain/ -v --cov=music_brain
```

### 3. Add missing function alias in `music_brain/groove/templates.py`
Add `list_genre_templates` as alias to `list_genres()` for backwards compatibility:

```python
# After line 214, add alias
list_genre_templates = list_genres
```

### 4. Add boom-bap template alias in `music_brain/groove/templates.py`
Add `boom-bap` as an alias for `hiphop` in GENRE_TEMPLATES or update get_genre_template to handle it.

### 5. Fix test expectations in `test_core_modules.py` line 188
Update test to check for correct available genres (`hiphop` instead of `boom-bap`).