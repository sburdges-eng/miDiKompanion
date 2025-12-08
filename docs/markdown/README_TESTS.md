# Test Suite Documentation

This document describes the test suite for the iDAW macOS standalone application build system.

## Test Structure

### Rust Tests

Located in `src-tauri/src/` and `src-tauri/tests/`:

- **Unit Tests** (`python_server.rs`): Test Python server management functions
  - `test_find_python_interpreter()` - Verifies Python interpreter discovery
  - `test_find_api_script()` - Verifies API script discovery
  - `test_check_server_health_when_down()` - Tests health check when server is down
  - `test_stop_server_when_not_running()` - Tests graceful stop when no server
  - `test_python_api_port_constant()` - Verifies port constant

- **Integration Tests** (`tests/integration_test.rs`): Test Tauri commands
  - `test_emotional_intent_serialization()` - Tests EmotionalIntent serialization
  - `test_generate_request_serialization()` - Tests GenerateRequest serialization
  - `test_interrogate_request_serialization()` - Tests InterrogateRequest serialization

### Python Tests

Located in `tests_music-brain/`:

- **Embedded Launcher Tests** (`test_embedded_launcher.py`):
  - `test_path_setup_in_development()` - Tests path setup in dev mode
  - `test_path_setup_in_bundle()` - Tests path setup in app bundle
  - `test_environment_variable_handling()` - Tests env var handling
  - `test_data_path_setup()` - Tests data path configuration
  - `test_port_default_value()` - Tests default port
  - `test_host_default_value()` - Tests default host

- **Build Script Tests** (`test_build_script.py`):
  - `test_build_script_exists()` - Verifies script exists
  - `test_build_script_is_executable()` - Verifies script is executable
  - `test_build_script_has_required_sections()` - Verifies script structure
  - `test_build_script_handles_options()` - Tests command-line options
  - `test_python_launcher_script_exists()` - Verifies launcher exists
  - `test_tauri_config_exists()` - Verifies Tauri config exists

- **Python Server Integration Tests** (`test_python_server_integration.py`):
  - `test_python_api_script_exists()` - Verifies API script exists
  - `test_api_health_endpoint_structure()` - Tests API structure
  - `test_start_python_server()` - Tests server startup (requires env var)
  - `test_server_port_configuration()` - Tests port configuration
  - `test_server_host_configuration()` - Tests host configuration

## Running Tests

### Rust Tests

```bash
# Run all Rust tests
cd src-tauri
cargo test

# Run specific test module
cargo test python_server

# Run with output
cargo test -- --nocapture
```

### Python Tests

```bash
# Run all Python tests
pytest tests_music-brain/test_embedded_launcher.py -v
pytest tests_music-brain/test_build_script.py -v
pytest tests_music-brain/test_python_server_integration.py -v

# Run all tests in directory
pytest tests_music-brain/ -v

# Run with coverage
pytest tests_music-brain/ --cov=music_brain --cov-report=term-missing
```

### Integration Tests (Full Server)

Some tests require a running server. Set environment variable:

```bash
# Run integration tests that require server
RUN_INTEGRATION_TESTS=1 pytest tests_music-brain/test_python_server_integration.py -v
```

## Test Coverage

### What's Tested

✅ **Python Server Management (Rust)**
- Interpreter discovery
- Script discovery
- Server health checks
- Server lifecycle (start/stop)

✅ **Tauri Commands**
- Request/response serialization
- Data structure validation

✅ **Embedded Launcher (Python)**
- Path resolution
- Environment variable handling
- Bundle vs development mode

✅ **Build Script**
- Script existence and structure
- Configuration validation
- File presence checks

### What's Not Tested (Yet)

⚠️ **Full Integration**
- End-to-end app bundle creation
- Actual server startup in Tauri context
- Complete build process execution

⚠️ **Error Scenarios**
- Network failures
- File system errors
- Permission issues

## Adding New Tests

### Rust Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function() {
        // Test implementation
        assert!(true);
    }
}
```

### Python Test Example

```python
def test_my_function():
    """Test description"""
    result = my_function()
    assert result == expected
```

## Continuous Integration

Tests should be run in CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run Rust tests
  run: cd src-tauri && cargo test

- name: Run Python tests
  run: pytest tests_music-brain/ -v
```

## Test Maintenance

- Update tests when adding new features
- Keep tests fast and isolated
- Use mocks for external dependencies
- Document test requirements
- Run tests before committing
