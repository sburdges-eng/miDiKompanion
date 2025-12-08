# macOS App Tests

Comprehensive test suite for the DAiW macOS desktop application.

## Test Files

### `test_macos_app.py`
Main test suite covering:
- **Launcher Tests**: Port management, server startup, webview creation
- **Streamlit App Tests**: App initialization and imports
- **App Pages Tests**: UI component testing
- **Integration Tests**: Full app flow
- **macOS-Specific Tests**: App bundle structure, Info.plist validation
- **Error Handling**: Server failures, timeouts, cleanup
- **Configuration Tests**: App metadata and settings

### `test_app_integration.py`
End-to-end integration tests:
- Complete launch flow
- Dependency checking
- Configuration validation
- Error handling scenarios
- Streamlit command generation

## Running Tests

### Run All macOS App Tests
```bash
pytest tests/test_macos_app.py -v
pytest tests/test_app_integration.py -v
```

### Run Specific Test Classes
```bash
# Test launcher functionality
pytest tests/test_macos_app.py::TestLauncher -v

# Test macOS-specific features
pytest tests/test_macos_app.py::TestMacOSSpecific -v

# Test integration
pytest tests/test_app_integration.py::TestAppLaunchFlow -v
```

### Run with Coverage
```bash
pytest tests/test_macos_app.py --cov=launcher --cov=app --cov-report=html
```

## Test Coverage

### Launcher Module (`launcher.py`)
- ✅ `find_free_port()` - Port discovery
- ✅ `run_streamlit()` - Server startup
- ✅ `wait_for_server()` - Server readiness
- ✅ `start_webview()` - Native window creation
- ✅ `main()` - Complete launch flow
- ✅ Error handling and cleanup

### App Module (`app.py`)
- ✅ Module imports
- ✅ Streamlit availability
- ✅ Page structure validation

### macOS-Specific
- ✅ Platform detection
- ✅ App bundle structure (if built)
- ✅ Info.plist validation
- ✅ PyInstaller spec file

## Requirements

### Required Dependencies
- `pytest>=7.0.0`
- `streamlit` (for app tests)
- `pywebview` (optional, for native window tests)

### Optional Dependencies
- `pytest-cov` (for coverage reports)
- `pytest-mock` (for advanced mocking)

## Test Categories

### Unit Tests
- Individual function testing
- Mock-based isolation
- Fast execution

### Integration Tests
- Full workflow testing
- Real subprocess execution
- End-to-end validation

### Platform-Specific Tests
- macOS-only features
- App bundle validation
- Native window testing

## Skipping Tests

Some tests are automatically skipped if dependencies are missing:
- `pywebview` tests skip if library not installed
- `streamlit` tests skip if not available
- macOS-specific tests skip on non-macOS platforms

## Mocking Strategy

Tests use extensive mocking to:
- Avoid starting actual Streamlit servers
- Prevent opening real windows
- Isolate unit functionality
- Speed up test execution

## Continuous Integration

These tests are designed to run in CI/CD pipelines:
- Fast execution (< 5 seconds)
- No external dependencies required
- Platform-aware skipping
- Clear failure messages

## Troubleshooting

### Tests Fail with Import Errors
```bash
# Install test dependencies
pip install -e ".[dev]"
```

### pywebview Tests Skip
```bash
# Install pywebview (optional)
pip install pywebview
```

### macOS Tests Skip on Linux/Windows
This is expected - macOS-specific tests only run on macOS.

## Future Enhancements

- [ ] Visual regression testing for UI
- [ ] Performance benchmarks
- [ ] Memory leak detection
- [ ] Accessibility testing
- [ ] App Store validation tests

