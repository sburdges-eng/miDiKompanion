# iDAW Cursor Workflow Prompts

Pre-configured prompts for common development workflows.

---

## Code Review Prompts

### Python Code Review
```
Review this Python code for:
1. Type hint completeness
2. RT-safety (if it interfaces with C++)
3. Proper error handling
4. Adherence to black formatting
5. Test coverage considerations
```

### C++ Code Review
```
Review this C++ code for:
1. RT-safety (no allocations in processAudio)
2. noexcept correctness
3. Thread safety (lock-free patterns)
4. Memory management (std::pmr usage)
5. SIMD optimization opportunities
```

---

## Feature Development Prompts

### Add New Groove Genre
```
I need to add a new groove genre called [GENRE_NAME].
Follow the iDAW pattern:
1. Add to Data_Files/genre_pocket_maps.json
2. Add template in music_brain/groove/templates.py
3. Add CLI choice in music_brain/cli.py
4. Add tests in tests_music-brain/
```

### Add New Rule-Breaking Option
```
I need to add a new rule-breaking option called [RULE_NAME].
Follow the iDAW pattern:
1. Add enum in music_brain/session/intent_schema.py
2. Add to RULE_BREAKING_EFFECTS dict
3. Implement in intent_processor.py
4. Add tests
```

### Add New Penta-Core Engine
```
I need to add a new Penta-Core engine for [SUBSYSTEM_NAME].
Follow the iDAW pattern:
1. Create header in include/penta/[subsystem]/
2. Implement in src_penta-core/[subsystem]/
3. Update CMakeLists.txt
4. Add Python bindings in python/penta_core/
5. Add tests in tests_penta-core/
```

---

## Testing Prompts

### Run Full Test Suite
```
Run the complete test suite:
- pytest tests_music-brain/ -v
- pytest DAiW-Music-Brain/tests/ -v
- For C++: cd build && ctest --output-on-failure
```

### Add Test Coverage
```
Add tests for [FUNCTION/CLASS_NAME]:
1. Unit tests for normal operation
2. Edge case tests
3. Error handling tests
4. Integration tests if applicable
Follow pytest conventions and existing test patterns.
```

---

## Debugging Prompts

### Debug RT-Safety Issue
```
Help me debug an RT-safety issue:
1. Check for memory allocations in audio thread
2. Verify lock-free data structure usage
3. Check for blocking operations
4. Verify noexcept marking
```

### Debug Python Import
```
Help me debug a Python import issue:
1. Check module structure and __init__.py
2. Verify package installation (pip install -e .)
3. Check circular import patterns
4. Verify lazy import patterns for CLI
```

---

## Documentation Prompts

### Document New Feature
```
Document this new feature following iDAW patterns:
1. Update CLAUDE.md if architecture changed
2. Add docstrings with type hints
3. Update relevant vault/ files if applicable
4. Add usage examples
```

---

## Refactoring Prompts

### Refactor for RT-Safety
```
Refactor this code for RT-safety:
1. Move allocations outside audio thread
2. Use lock-free ring buffers for communication
3. Add noexcept to audio callbacks
4. Use std::pmr containers where applicable
```

### Optimize SIMD
```
Optimize this code with SIMD:
1. Use AVX2 intrinsics where available
2. Provide scalar fallback
3. Align data for SIMD operations
4. Benchmark before and after
```

---

## MCP Integration Prompts

### Add MCP Tool
```
Add a new MCP tool to [SERVER_NAME]:
1. Define tool schema
2. Implement handler function
3. Register with MCP server
4. Add to tool documentation
5. Test with multiple AI clients
```

### Cross-AI Task Sync
```
Set up cross-AI task synchronization:
1. Use mcp_todo for task storage
2. Implement task creation in current AI
3. Verify task appears in other AIs
4. Add priority and tags for organization
```
