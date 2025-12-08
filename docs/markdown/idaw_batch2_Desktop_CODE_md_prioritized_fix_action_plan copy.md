# Music Brain Vault - Prioritized Fix Action Plan

## Phase 1: Critical Fixes (Week 1-2)
*These issues completely break functionality and must be fixed first*

### Day 1-2: Core Logic Fixes
- [ ] **Fix chord_to_degree function** (4 hours)
  - Add minor scale support
  - Use negative values for chromatic chords
  - Add mode detection
  - File: `/music_brain/structure/progression.py`

- [ ] **Fix progression matching logic** (4 hours)
  - Remove `a != 0` broken logic
  - Add confidence scoring
  - Fix tolerance calculation
  - File: `/music_brain/structure/progression.py`

### Day 3-4: Data Integrity
- [ ] **Fix duplicate patterns** (1 hour)
  - Correct pop_1546 vs pop_1564
  - Audit all patterns for duplicates
  - File: `/music_brain/structure/progression.py`

- [ ] **Fix timing_map semantics** (3 hours)
  - Convert old timing_map to timing_density + timing_offset
  - Update all genre templates
  - Document units clearly
  - Files: All genre template files

### Day 5: Safety Fixes
- [ ] **Add error handling** (6 hours)
  - Wrap all librosa.load calls
  - Add file validation
  - Fix division by zero issues
  - Handle tempo array vs scalar
  - Files: `/music_brain/audio/feel.py`, all audio files

### Day 6-7: Validation Framework
- [ ] **Implement validate_template()** (4 hours)
  - Create validation function
  - Add to template_storage.py
  - Validate on save/load

- [ ] **Add input sanitization** (4 hours)
  - Path traversal protection
  - File type validation
  - Size limits

### Day 8-9: Testing Foundation
- [ ] **Create test suite skeleton** (8 hours)
  - Set up pytest
  - Create test structure
  - Write tests for fixed functions
  - Minimum 10 tests per module

### Day 10: Documentation Update
- [ ] **Update all docstrings** (4 hours)
  - Document fixed functions
  - Add usage examples
  - Update README with fixes

---

## Phase 2: Functional Improvements (Week 3-4)
*These add missing core functionality*

### Week 3: Music Theory
- [ ] **Implement minor scales** (8 hours)
  - Natural, harmonic, melodic minor
  - Update chord detection
  - Update progression analysis

- [ ] **Add modal support** (8 hours)
  - Dorian, Mixolydian, etc.
  - Mode detection algorithm
  - Modal progression patterns

- [ ] **Key detection improvement** (4 hours)
  - Better parsing regex
  - Handle all common formats
  - Add validation

### Week 4: Instrument Handling
- [ ] **Per-instrument velocity curves** (12 hours)
  - Redesign template structure
  - Add instrument mapping
  - Update all templates

- [ ] **Instrument pocket implementation** (8 hours)
  - Apply pocket offsets
  - Test with MIDI files
  - Validate output

---

## Phase 3: Core Features (Week 5-6)
*Essential missing features*

### Week 5: Groove Application
- [ ] **Create groove applicator** (16 hours)
  - Read MIDI file
  - Apply template
  - Output transformed MIDI
  - Main feature implementation

- [ ] **PPQ scaling** (8 hours)
  - Implement scale_template usage
  - Auto-detect source PPQ
  - Convert between PPQs

### Week 6: MIDI Generation
- [ ] **MIDI output module** (12 hours)
  - Generate MIDI from templates
  - Create from scratch
  - Merge with existing MIDI

- [ ] **Pattern generator** (12 hours)
  - Use timing_density for probability
  - Generate variations
  - Section awareness

---

## Phase 4: Quality Assurance (Week 7-8)
*Testing and robustness*

### Week 7: Comprehensive Testing
- [ ] **Unit tests** (40 hours)
  - 80% code coverage minimum
  - All critical paths tested
  - Edge cases covered

- [ ] **Integration tests** (16 hours)
  - End-to-end workflows
  - File I/O tests
  - Database tests

### Week 8: Performance & Security
- [ ] **Performance optimization** (16 hours)
  - Profile code
  - Optimize hot paths
  - Add caching

- [ ] **Security audit** (8 hours)
  - Path traversal
  - SQL injection
  - File upload limits

---

## Phase 5: Advanced Features (Week 9-10)
*Nice-to-have enhancements*

### Week 9: Analysis Enhancement
- [ ] **Chord inversions** (8 hours)
- [ ] **Modulation detection** (8 hours)
- [ ] **Micro-timing analysis** (8 hours)
- [ ] **Ghost note detection** (8 hours)

### Week 10: UI/UX
- [ ] **Web interface** (24 hours)
  - REST API
  - Basic frontend
  - File upload/download

- [ ] **DAW plugin research** (16 hours)
  - VST/AU feasibility
  - JUCE integration
  - Prototype

---

## Quick Wins (Can do anytime)
*Low effort, high impact*

- [ ] **Add logging** (2 hours)
  - Replace print statements
  - Add log levels
  - Configure output

- [ ] **Fix imports** (1 hour)
  - Standardize style
  - Remove unused imports
  - Order properly

- [ ] **Add type hints** (4 hours)
  - Complete missing types
  - Use modern syntax
  - Add mypy config

- [ ] **Create constants file** (2 hours)
  - Extract magic numbers
  - Central configuration
  - Document meanings

---

## Testing Checklist
For each fixed component:

1. **Unit Test**
   - [ ] Happy path
   - [ ] Error cases
   - [ ] Edge cases
   - [ ] Performance test

2. **Integration Test**
   - [ ] With real files
   - [ ] With other modules
   - [ ] End-to-end

3. **Documentation**
   - [ ] Docstring updated
   - [ ] Example added
   - [ ] README updated

---

## Success Metrics

### Phase 1 Complete When:
- All critical bugs fixed
- No crashes on valid input
- Basic test coverage exists

### Phase 2 Complete When:
- Minor scales work
- All instruments handled separately
- Key detection robust

### Phase 3 Complete When:
- Can transform MIDI files
- Can generate new patterns
- PPQ-independent

### Phase 4 Complete When:
- 80% test coverage
- No security vulnerabilities
- Performance acceptable (<1s for typical file)

### Phase 5 Complete When:
- Advanced features working
- UI prototype exists
- Ready for beta testing

---

## Resource Requirements

### Developer Time:
- Phase 1: 80 hours (2 weeks)
- Phase 2: 80 hours (2 weeks)
- Phase 3: 80 hours (2 weeks)
- Phase 4: 80 hours (2 weeks)
- Phase 5: 80 hours (2 weeks)
- **Total: 400 hours (10 weeks full-time)**

### Dependencies:
- Python 3.8+
- librosa 0.9+
- numpy, scipy
- mido (for MIDI)
- pytest (for testing)
- FastAPI (for web interface)

### Test Data Needed:
- MIDI files (various genres)
- Audio files (for analysis)
- Known good templates
- Edge case examples

---

## Risk Mitigation

### High Risk Areas:
1. **Groove application** - Core feature, no existing implementation
2. **Performance with large files** - Needs streaming
3. **DAW integration** - Complex, platform-specific

### Mitigation Strategies:
1. Build prototype first, iterate
2. Implement streaming early
3. Consider alternatives (standalone first)

---

## Definition of Done

The project is considered "fixed" when:

1. ✅ All Phase 1-3 items complete
2. ✅ Test coverage > 80%
3. ✅ No critical bugs in issue tracker
4. ✅ Documentation complete
5. ✅ Can process real-world files
6. ✅ Performance acceptable
7. ✅ Security audit passed
8. ✅ Beta users successful

---

## Next Steps

1. **Immediate Action:**
   - Fix chord_to_degree function TODAY
   - Set up pytest framework
   - Create GitHub issues for each task

2. **This Week:**
   - Complete Phase 1 critical fixes
   - Write tests for fixed code
   - Update documentation

3. **This Month:**
   - Complete Phases 1-3
   - Have working groove application
   - Begin beta testing

Remember: **Fix the foundations first, then build features!**
