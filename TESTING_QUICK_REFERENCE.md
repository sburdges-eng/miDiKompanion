# Music Brain Testing Quick Reference

**Quick access guide for developers working on test coverage**

---

## Running Tests

### Quick Commands

```bash
# Run all tests
pytest tests_music-brain/ -v

# Run specific test file
pytest tests_music-brain/test_harmony.py -v

# Run specific test class
pytest tests_music-brain/test_harmony.py::TestHarmonyGeneratorInit -v

# Run specific test
pytest tests_music-brain/test_harmony.py::TestHarmonyGeneratorInit::test_default_initialization -v

# Run with coverage
pytest tests_music-brain/ --cov=music_brain --cov-report=term-missing

# Run fast tests only (skip slow)
pytest tests_music-brain/ -m "not slow" -v

# Run failed tests from last run
pytest tests_music-brain/ --lf -v

# Stop on first failure
pytest tests_music-brain/ -x

# Show print statements
pytest tests_music-brain/ -s
```

### Coverage Reports

```bash
# Terminal coverage report
pytest tests_music-brain/ --cov=music_brain --cov-report=term-missing

# HTML coverage report
pytest tests_music-brain/ --cov=music_brain --cov-report=html
open htmlcov/index.html

# XML for CI/CD
pytest tests_music-brain/ --cov=music_brain --cov-report=xml
```

---

## Test File Structure

### Naming Conventions

```
tests_music-brain/
├── test_<module>.py          # Test module
├── test_<feature>.py          # Test specific feature
└── test_<integration>.py      # Integration test
```

**Examples:**
- `test_harmony.py` - Tests for harmony.py
- `test_emotion_api.py` - Tests for emotion_api.py
- `test_complete_workflows.py` - Integration tests

### Test Class Naming

```python
class TestFeatureName:
    """Test description"""

    def test_specific_behavior(self):
        """Test case description"""
        pass
```

**Examples:**
- `TestHarmonyGeneratorInit` - Initialization tests
- `TestBasicProgression` - Basic progression tests
- `TestModalInterchange` - Modal interchange tests

---

## Writing New Tests

### Basic Test Template

```python
"""
Tests for music_brain.<module> module.

Tests cover:
- Feature 1
- Feature 2
- Edge cases

Run with: pytest tests_music-brain/test_<module>.py -v
"""

import pytest
from music_brain.<module> import <functions>


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"key": "value"}


# ==============================================================================
# BASIC FUNCTIONALITY TESTS
# ==============================================================================

class TestBasicFeature:
    """Test basic feature functionality."""

    def test_basic_case(self, sample_data):
        """Test basic case."""
        result = process(sample_data)
        assert result is not None

    def test_edge_case_empty_input(self):
        """Test empty input."""
        result = process({})
        assert result == expected_default


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================

class TestErrorHandling:
    """Test error conditions."""

    def test_raises_on_invalid_input(self):
        """Test raises ValueError on invalid input."""
        with pytest.raises(ValueError, match="Invalid input"):
            process(invalid_data)


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestIntegration:
    """Test integration with other modules."""

    def test_complete_workflow(self):
        """Test complete workflow."""
        # Multi-step test
        pass
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("C", "C major"),
    ("Am", "A minor"),
    ("F#m", "F# minor"),
])
def test_chord_parsing(input, expected):
    """Test chord parsing with multiple inputs."""
    result = parse_chord(input)
    assert result == expected
```

### Fixtures for Test Data

```python
@pytest.fixture
def midi_file():
    """Create temporary MIDI file."""
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
        temp_path = f.name

    # Create MIDI
    mid = mido.MidiFile()
    mid.save(temp_path)

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)
```

---

## Common Test Patterns

### Testing Exceptions

```python
def test_raises_value_error():
    """Test raises ValueError."""
    with pytest.raises(ValueError, match="Expected error message"):
        function_that_raises()
```

### Testing File Operations

```python
def test_creates_file():
    """Test file creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.mid"
        create_file(str(output_path))
        assert output_path.exists()
```

### Testing with Mocks

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test with mocked dependency."""
    with patch('module.external_function') as mock_func:
        mock_func.return_value = "mocked_result"
        result = function_using_external()
        assert result == "expected_based_on_mock"
```

### Testing Optional Dependencies

```python
pytest.mark.skipif(not MIDO_AVAILABLE, reason="mido not installed")
def test_requires_mido():
    """Test that requires mido."""
    # Test code using mido
    pass
```

---

## Assertion Patterns

### Basic Assertions

```python
# Equality
assert result == expected
assert result != unexpected

# Truth
assert result
assert not result

# Type checking
assert isinstance(result, ExpectedType)

# Container membership
assert "key" in result
assert item in collection

# Numeric comparisons
assert value > 0
assert 0 <= value <= 1
assert abs(actual - expected) < tolerance
```

### Advanced Assertions

```python
# All/any
assert all(item > 0 for item in collection)
assert any(item > 100 for item in collection)

# Length
assert len(result) == expected_length
assert len(result) > 0

# Approximation (floating point)
assert pytest.approx(result, abs=0.01) == expected

# Custom messages
assert result == expected, f"Expected {expected}, got {result}"
```

---

## Test Organization by Priority

### HIGH Priority Tests (Write First)

```python
class TestCriticalPath:
    """Test critical functionality."""

    def test_happy_path(self):
        """Test most common usage."""
        pass

    def test_raises_on_invalid(self):
        """Test error handling."""
        pass
```

### MEDIUM Priority Tests

```python
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_input(self):
        pass

    def test_maximum_input(self):
        pass
```

### LOW Priority Tests

```python
class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.benchmark
    def test_performance_target(self, benchmark):
        result = benchmark(expensive_function)
        assert benchmark.stats.mean < 0.1  # 100ms
```

---

## Coverage Targets by Module

| Module | Target | Command |
|--------|--------|---------|
| harmony.py | 95% | `pytest --cov=music_brain.harmony` |
| emotion_api.py | 95% | `pytest --cov=music_brain.emotion_api` |
| groove/ | 90% | `pytest --cov=music_brain.groove` |
| structure/ | 90% | `pytest --cov=music_brain.structure` |
| audio/ | 85% | `pytest --cov=music_brain.audio` |
| daw/ | 85% | `pytest --cov=music_brain.daw` |

---

## Debugging Failed Tests

### Common Issues

**Import errors:**
```bash
# Check if module is importable
python -c "import music_brain.harmony"

# Install missing dependencies
pip install -e ".[dev]"
```

**Path issues:**
```python
# Use absolute paths
from pathlib import Path
test_file = Path(__file__).parent / "fixtures" / "test.mid"
```

**Fixture issues:**
```bash
# List all fixtures
pytest tests_music-brain/ --fixtures

# Use fixture in test
def test_with_fixture(sample_fixture):
    pass
```

### Debugging Commands

```bash
# Show detailed output
pytest tests_music-brain/test_harmony.py -vv

# Show print statements
pytest tests_music-brain/test_harmony.py -s

# Drop into debugger on failure
pytest tests_music-brain/test_harmony.py --pdb

# Show local variables on failure
pytest tests_music-brain/test_harmony.py -l

# Show test collection
pytest tests_music-brain/ --collect-only
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest tests_music-brain/ -v --cov=music_brain --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

---

## Useful pytest Plugins

```bash
# Install useful plugins
pip install pytest-cov          # Coverage
pip install pytest-benchmark   # Performance
pip install pytest-xdist        # Parallel execution
pip install pytest-watch        # Auto-run on file changes
```

### Using pytest-xdist (Parallel)

```bash
# Run tests in parallel (4 workers)
pytest tests_music-brain/ -n 4
```

### Using pytest-watch

```bash
# Auto-run tests on file changes
ptw tests_music-brain/
```

---

## Quick Reference Checklist

When writing a new test:

- [ ] Test file named `test_<module>.py`
- [ ] Docstring at top with description
- [ ] Fixtures defined if needed
- [ ] Test classes organized by feature
- [ ] Test methods named `test_<behavior>`
- [ ] Docstrings for each test method
- [ ] Happy path tested first
- [ ] Error cases tested
- [ ] Edge cases covered
- [ ] Assertions clear and specific
- [ ] Cleanup handled (files, resources)
- [ ] No hardcoded paths
- [ ] Mocks used for external dependencies
- [ ] Coverage target met (check with `--cov`)

---

## Getting Help

**Documentation:**
- pytest docs: https://docs.pytest.org/
- Coverage docs: https://coverage.readthedocs.io/
- Project docs: `/TEST_COVERAGE_REPORT.md`

**TODOs:**
- See `/TEST_COVERAGE_TODOS.md` for detailed tasks

**Questions:**
- Check existing tests for patterns
- Review CLAUDE.md for project standards
- Ask in team chat or PR comments
