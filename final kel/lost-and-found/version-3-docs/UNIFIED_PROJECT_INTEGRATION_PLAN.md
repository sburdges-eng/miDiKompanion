# Unified Kelly MIDI + iDAW Integration Plan

**Created**: December 15, 2024
**Purpose**: Merge all Kelly MIDI and iDAW projects into one unified codebase

---

## Executive Summary

You currently have **5 separate but related projects** on your desktop:
1. **1DAW1** (508 files) - Full Tauri desktop app
2. **iDAW** (468 files) - Python implementation
3. **kelly-midi-max/kellymidicompanion** - JUCE plugin (WORKING - just fixed UI)
4. **Obsidian_Documentation** - 93+ production guides
5. **TEST UPLOADS** - Test specs and MIDI files

**Goal**: Create ONE unified project that combines the best of all systems.

---

## Core Philosophy (Shared Across All Projects)

**"Interrogate Before Generate"**
- The tool shouldn't finish art for people; it should make them braver
- Created in memory of Kelly, focusing on therapeutic music creation
- 216-node emotion thesaurus (6×6×6 hierarchy)
- Three-phase intent system: Wound → Emotion → Rule-breaking
- Intentional music theory violations for emotional authenticity

---

## What Each Project Brings

### 1. 1DAW1 (KEEP AS PRIMARY BASE)
**Strengths:**
- ✅ Most complete implementation
- ✅ Modern tech stack (Tauri 2.0, React 19, TypeScript)
- ✅ Penta-Core C++ engines (harmony, groove, OSC)
- ✅ Full Python music intelligence (DAiW-Music-Brain)
- ✅ MCP Multi-AI orchestration
- ✅ Complete build system (CMake + Cargo + npm)
- ✅ 18-month roadmap (12 months complete)

**What to Extract:**
- Complete Penta-Core C++ engines → Keep all
- Python music_brain package → Keep all
- React frontend → Keep all
- Tauri desktop wrapper → Keep
- MCP TODO server → Keep
- All documentation (100+ markdown files)
- All data files (JSON databases)

### 2. kellymidicompanion (KEEP AS PLUGIN COMPONENT)
**Strengths:**
- ✅ **WORKING JUCE PLUGIN** (VST3/AU) - Just fixed in Logic Pro
- ✅ Professional plugin architecture
- ✅ Cassette tape UI design (backed up)
- ✅ Emotion wheel component (backed up)
- ✅ v2.0.0 features (voice synthesis, biometric input frameworks)

**What to Extract:**
- JUCE plugin implementation → Keep as separate component
- PluginProcessor.cpp → Integrate with 1DAW1 engines
- Emotion wheel UI → Port to React
- Cassette UI design → Port to React
- All C++ engines (if better than 1DAW1 versions)

### 3. iDAW (MERGE INTO 1DAW1)
**Strengths:**
- ✅ Streamlit web UI (alternative interface)
- ✅ Some unique Python utilities

**What to Extract:**
- Any unique Python utilities not in 1DAW1
- Streamlit UI → Keep as alternative interface option
- Logic Pro integration scripts → Merge with 1DAW1

### 4. Obsidian_Documentation (MERGE AS KNOWLEDGE BASE)
**Strengths:**
- ✅ 93+ production guides (organized and complete)
- ✅ Genre pocket maps (13 genres)
- ✅ Chord progression families (50+ progressions)
- ✅ Python tools (groove extractor, audio cataloger)

**What to Extract:**
- All 93 guides → Create unified /docs folder
- JSON data files → Merge into 1DAW1/Data_Files
- Python tools → Integrate into music_brain package
- Templates → Add to 1DAW1/Templates

### 5. TEST UPLOADS (MERGE AS TEST SUITE)
**Strengths:**
- ✅ Comprehensive test specifications
- ✅ 17 MIDI test files (emotions, grooves, rule-breaking)
- ✅ OpenAPI spec for MCP TODO API
- ✅ Song intent schema with 3 examples
- ✅ 428-line TypeScript test suite

**What to Extract:**
- All MIDI test files → 1DAW1/test_cases/midi/
- unified.test.ts → 1DAW1/src/__tests__/
- OpenAPI spec → 1DAW1/mcp_todo/
- Song intent schema → 1DAW1/Data_Files/

---

## Unified Project Structure

```
kelly-daw-unified/
├── README.md                          # Main project README
├── LICENSE.md
├── ARCHITECTURE.md                    # Complete architecture overview
├── ROADMAP.md                         # Unified roadmap
│
├── docs/                              # Unified documentation
│   ├── Production_Workflows/          # 93+ guides from Obsidian
│   ├── Songwriting_Guides/
│   ├── Theory_Reference/
│   ├── API/                           # API documentation
│   ├── Development/                   # Development guides
│   └── User_Guides/                   # End-user documentation
│
├── data/                              # All JSON databases
│   ├── emotion/                       # Emotion thesaurus data
│   │   ├── anger.json
│   │   ├── sad.json
│   │   ├── fear.json
│   │   └── ...
│   ├── music/                         # Musical databases
│   │   ├── genre_pocket_maps.json
│   │   ├── chord_progression_families.json
│   │   ├── scales_database.json
│   │   └── ...
│   └── rules/                         # Rule-breaking database
│       └── rule_breaking_database.json
│
├── src-core/                          # C++ Core (Penta-Core)
│   ├── include/penta/
│   │   ├── common/                    # RT types, memory pools
│   │   ├── harmony/                   # HarmonyEngine, ChordAnalyzer
│   │   ├── groove/                    # GrooveEngine, OnsetDetector
│   │   ├── osc/                       # OSC communication
│   │   └── diagnostics/               # Performance monitoring
│   ├── src/
│   │   ├── harmony/
│   │   ├── groove/
│   │   ├── osc/
│   │   └── diagnostics/
│   ├── CMakeLists.txt
│   └── tests/                         # C++ tests (Catch2)
│
├── src-plugin/                        # JUCE Plugin (Kelly MIDI Companion)
│   ├── src/
│   │   ├── common/Types.h
│   │   ├── engine/                    # EmotionThesaurus, IntentPipeline
│   │   ├── midi/                      # MIDI generation
│   │   ├── plugin/                    # PluginProcessor, PluginEditor
│   │   ├── ui/                        # CassetteView, EmotionWheel
│   │   ├── voice/                     # VoiceSynthesizer
│   │   └── biometric/                 # BiometricInput
│   ├── CMakeLists.txt
│   ├── build_and_install.sh
│   └── tests/                         # Plugin tests
│
├── python/                            # Python packages
│   ├── music_brain/                   # DAiW-Music-Brain package
│   │   ├── __init__.py
│   │   ├── api.py                     # FastAPI server
│   │   ├── cli.py                     # daiw command
│   │   ├── emotion_mapper.py
│   │   ├── interrogator.py
│   │   ├── groove/                    # Groove extraction/application
│   │   ├── harmony/                   # Chord analysis
│   │   ├── structure/                 # Song structure
│   │   ├── session/                   # Intent processing
│   │   ├── audio/                     # Audio analysis
│   │   └── daw/                       # DAW integration
│   ├── penta_core/                    # Python bindings for C++
│   │   ├── rules/                     # Rule system
│   │   ├── teachers/                  # Interactive teaching
│   │   ├── ml/                        # Machine learning
│   │   └── collaboration/             # WebSocket server
│   ├── mcp_workstation/               # Multi-AI orchestration
│   ├── mcp_todo/                      # Cross-AI TODO server
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── setup.py
│
├── src-tauri/                         # Tauri Desktop App (Rust)
│   ├── src/
│   │   └── main.rs
│   ├── Cargo.toml
│   └── tauri.conf.json
│
├── src/                               # React Frontend
│   ├── components/
│   │   ├── SideA/                     # Professional DAW interface
│   │   │   ├── Timeline.tsx
│   │   │   ├── Mixer.tsx
│   │   │   └── Transport.tsx
│   │   ├── SideB/                     # Therapeutic interface
│   │   │   ├── EmotionWheel.tsx       # Ported from JUCE
│   │   │   ├── Interrogator.tsx
│   │   │   └── CassetteView.tsx       # Ported from JUCE
│   │   ├── Shared/
│   │   └── Bridge/                    # Side A ↔ Side B communication
│   ├── hooks/
│   │   ├── useAudioEngine.ts
│   │   ├── useUnifiedStore.ts
│   │   └── useEmotionMapping.ts
│   ├── utils/
│   ├── App.tsx
│   └── main.tsx
│
├── tests/                             # Unified test suite
│   ├── python/                        # Python tests (pytest)
│   │   ├── test_music_brain/
│   │   ├── test_penta_core/
│   │   └── test_integration/
│   ├── cpp/                           # C++ tests (Catch2)
│   │   ├── test_harmony.cpp
│   │   ├── test_groove.cpp
│   │   └── test_simd.cpp
│   ├── typescript/                    # TypeScript tests (Vitest)
│   │   ├── unified.test.ts            # From TEST UPLOADS
│   │   └── components/
│   └── test_cases/                    # Test data
│       ├── midi/                      # 17 MIDI test files from TEST UPLOADS
│       ├── audio/
│       └── intents/
│
├── mcp_servers/                       # MCP Server implementations
│   ├── todo/                          # MCP TODO server
│   ├── workstation/                   # Multi-AI orchestration
│   └── music_brain/                   # Music intelligence MCP
│
├── templates/                         # DAW templates
│   ├── Logic_Pro/
│   ├── Ableton_Live/
│   ├── FL_Studio/
│   └── ...
│
├── scripts/                           # Build and utility scripts
│   ├── build_all.sh
│   ├── install_plugin.sh
│   ├── run_tests.sh
│   └── deploy_desktop.sh
│
├── .github/workflows/                 # CI/CD
│   ├── build-cpp.yml
│   ├── build-python.yml
│   ├── build-desktop.yml
│   └── build-plugin.yml
│
├── package.json                       # Node.js dependencies
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── CMakeLists.txt                     # Root CMake
├── Cargo.toml                         # Rust workspace
├── pyproject.toml                     # Python package
└── docker-compose.yml
```

---

## Unified Technology Stack

### Languages
- **C++20** - Real-time engines, JUCE plugin
- **Python 3.10+** - Music intelligence, AI, backend
- **Rust** - Tauri desktop backend
- **TypeScript** - React frontend

### Frameworks
- **JUCE 8.0.10** - Audio plugin framework
- **Tauri 2.0** - Desktop application
- **React 19.1** - UI framework
- **FastAPI** - Python REST API
- **Tone.js** - Web Audio synthesis

### Build Systems
- **CMake 3.22+** - C++ builds
- **Cargo** - Rust builds
- **npm/Vite** - Frontend builds
- **setuptools** - Python packaging

---

## Component Mapping (Where Code Goes)

### Emotion System (MERGE ALL VARIANTS)

| Source | Component | Destination |
|--------|-----------|-------------|
| 1DAW1 | EmotionThesaurus (Python) | `python/music_brain/emotion_mapper.py` |
| kellymidi | EmotionThesaurus.cpp/.h | `src-plugin/src/engine/` |
| 1DAW1 | emotion JSON files (6 files) | `data/emotion/` |
| All | 216-node hierarchy | **UNIFIED** single source of truth |

**Action**: Create canonical emotion data format that both C++ and Python read.

### MIDI Generation Engines (CHOOSE BEST VERSION)

| Engine | Best Source | Destination |
|--------|-------------|-------------|
| ChordGenerator | kellymidi (C++) | `src-plugin/src/midi/` |
| MelodyEngine | 1DAW1 (Python) | `python/music_brain/melody/` |
| GrooveEngine | 1DAW1 Penta-Core (C++) | `src-core/src/groove/` |
| BassEngine | kellymidi (C++) | `src-plugin/src/midi/` |
| DrumEngine | 1DAW1 (Python) | `python/music_brain/drums/` |
| ArrangementEngine | kellymidi (C++) | `src-plugin/src/midi/` |
| TensionEngine | 1DAW1 (Python) | `python/music_brain/structure/` |

**Action**: Audit each engine pair, choose best implementation, document decision.

### UI Components (PORT JUCE → REACT)

| Component | Source | Destination |
|-----------|--------|-------------|
| EmotionWheel | kellymidi (JUCE) | `src/components/SideB/EmotionWheel.tsx` |
| CassetteView | kellymidi (JUCE) | `src/components/SideB/CassetteView.tsx` |
| Timeline | 1DAW1 (React) | `src/components/SideA/Timeline.tsx` |
| Mixer | 1DAW1 (React) | `src/components/SideA/Mixer.tsx` |

**Action**: Port JUCE UI components to React using Canvas API or SVG.

### Documentation (CONSOLIDATE ALL)

| Source | Files | Destination |
|--------|-------|-------------|
| Obsidian_Documentation | 93+ guides | `docs/Production_Workflows/`, `docs/Songwriting_Guides/` |
| 1DAW1/docs_penta-core | 25+ guides | `docs/Development/Penta_Core/` |
| 1DAW1/docs_music-brain | 10+ guides | `docs/Development/Music_Brain/` |
| kellymidi/README.md | Plugin docs | `docs/User_Guides/Plugin_Guide.md` |
| 1DAW1/CLAUDE.md | Architecture | `docs/Development/ARCHITECTURE.md` |

### Test Files (MERGE ALL SUITES)

| Source | Test Files | Destination |
|--------|-----------|-------------|
| TEST UPLOADS | 17 MIDI files | `tests/test_cases/midi/` |
| TEST UPLOADS | unified.test.ts | `tests/typescript/` |
| 1DAW1/tests_music-brain | 35 Python tests | `tests/python/test_music_brain/` |
| 1DAW1/tests_penta-core | C++ tests | `tests/cpp/` |
| kellymidi/tests | Plugin tests | `tests/cpp/test_plugin/` |

### Data Files (MERGE AND DEDUPLICATE)

| File | Source Projects | Keep From | Destination |
|------|----------------|-----------|-------------|
| genre_pocket_maps.json | Obsidian, 1DAW1 | **1DAW1** (most complete - 13 genres) | `data/music/` |
| chord_progression_families.json | Obsidian, 1DAW1 | **Merge both** (combine unique progressions) | `data/music/` |
| emotion JSON files | 1DAW1, kellymidi | **1DAW1** (6 base emotions + blends) | `data/emotion/` |
| rule_breaking_database.json | 1DAW1 | 1DAW1 | `data/rules/` |
| scales_database.json | 1DAW1 | 1DAW1 | `data/music/` |

---

## Deduplication Strategy

### Duplicate Code to Merge

1. **EmotionThesaurus** - 3 implementations (C++, Python in 1DAW1, Python in iDAW)
   - **Keep**: C++ for plugin, Python for standalone
   - **Action**: Ensure both read from same JSON files

2. **IntentPipeline** - 2 implementations (C++, Python)
   - **Keep**: Both (C++ for real-time, Python for flexibility)
   - **Action**: Ensure identical behavior via tests

3. **GrooveEngine** - 3 implementations
   - **Keep**: Penta-Core C++ (fastest)
   - **Action**: Python wrapper via pybind11

4. **ChordGenerator** - 2 implementations
   - **Keep**: kellymidi C++ (most complete)
   - **Action**: Port to Penta-Core if better

### Files to Archive (Not Delete)

Create `/archive/` directory with:
- `/archive/iDAW/` - Original standalone Python version
- `/archive/kellymidi-backup/` - Pre-merge plugin state
- `/archive/obsidian-standalone/` - Standalone docs before merge

---

## Migration Steps (Ordered)

### Phase 1: Create Unified Repository (Week 1)

1. **Create new repo:**
   ```bash
   mkdir ~/Desktop/kelly-daw-unified
   cd ~/Desktop/kelly-daw-unified
   git init
   ```

2. **Copy 1DAW1 as base:**
   ```bash
   cp -r ~/Desktop/1DAW1/* .
   ```

3. **Integrate kellymidi plugin:**
   ```bash
   mkdir src-plugin
   cp -r ~/Desktop/kelly-midi-max/kellymidicompanion/kelly-midi-companion/* src-plugin/
   ```

4. **Add documentation:**
   ```bash
   mkdir docs
   cp -r ~/Desktop/Obsidian_Documentation/Production_Workflows docs/
   cp -r ~/Desktop/Obsidian_Documentation/Songwriting_Guides docs/
   ```

5. **Add test files:**
   ```bash
   mkdir -p tests/test_cases/midi
   cp ~/Desktop/TEST\ UPLOADS/*.mid tests/test_cases/midi/
   ```

### Phase 2: Merge Data Files (Week 1)

1. **Consolidate JSON databases:**
   ```bash
   mkdir -p data/{emotion,music,rules}
   # Copy and merge all JSON files
   ```

2. **Deduplicate and validate:**
   ```python
   # Write validation script to check for duplicates
   python scripts/validate_data.py
   ```

### Phase 3: Merge Python Code (Week 2)

1. **Audit music_brain packages:**
   - Compare 1DAW1 vs iDAW implementations
   - Keep most complete version of each module

2. **Merge unique utilities:**
   - Obsidian_Documentation Python tools → `python/music_brain/tools/`

3. **Update imports and tests:**
   ```bash
   python -m pytest tests/python/
   ```

### Phase 4: Merge C++ Code (Week 2)

1. **Compare Penta-Core vs kellymidi engines:**
   - Benchmark performance
   - Choose best implementation

2. **Port kellymidi engines to Penta-Core:**
   - If kellymidi versions are better, copy to `src-core/`

3. **Update CMakeLists.txt:**
   - Unified build for both plugin and core engines

### Phase 5: Port UI Components (Week 3)

1. **Analyze JUCE UI components:**
   - EmotionWheel drawing logic
   - CassetteView animation

2. **Port to React:**
   - Use Canvas API or SVG
   - Maintain visual design

3. **Test in Tauri desktop app:**
   ```bash
   npm run dev
   ```

### Phase 6: Update Build System (Week 3)

1. **Create unified CMake:**
   ```cmake
   # Root CMakeLists.txt builds:
   # - Penta-Core library
   # - Kelly MIDI plugin
   # - Python bindings
   ```

2. **Create build scripts:**
   ```bash
   ./scripts/build_all.sh
   ```

3. **Verify all outputs:**
   - Desktop app (Tauri)
   - VST3/AU plugin
   - Python package

### Phase 7: Consolidate Documentation (Week 4)

1. **Merge all markdown files:**
   - Avoid duplicates
   - Cross-link related docs

2. **Create unified README.md:**
   - Clear project overview
   - Installation instructions
   - Usage examples

3. **Generate API docs:**
   - Python: Sphinx
   - C++: Doxygen
   - TypeScript: TypeDoc

### Phase 8: Testing and Validation (Week 4)

1. **Run all test suites:**
   ```bash
   ./scripts/run_tests.sh
   ```

2. **Validate emotion thesaurus:**
   - C++ and Python produce same results

3. **Integration tests:**
   - Desktop app → Plugin communication
   - Python backend → React frontend
   - MIDI generation end-to-end

---

## Build Instructions (After Merge)

### Full Build (All Components)
```bash
cd kelly-daw-unified
./scripts/build_all.sh
```

### Individual Components

**C++ Core + Plugin:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Python Package:**
```bash
pip install -e ".[all]"
```

**Desktop App:**
```bash
npm install
npm run tauri dev    # Development
npm run tauri build  # Production
```

---

## Deliverables

After complete integration, you will have:

### 1. Kelly DAW Desktop Application
- Tauri 2.0 desktop app (macOS/Windows/Linux)
- Side A: Professional DAW interface
- Side B: Therapeutic emotion-driven interface
- Python backend for music intelligence
- C++ core for real-time processing

### 2. Kelly MIDI Companion Plugin
- VST3/AU plugin
- Works in Logic Pro, Ableton, FL Studio, etc.
- Emotion wheel UI
- Real-time MIDI generation

### 3. Python Library
- `pip install kelly-music-brain`
- CLI: `kelly` command
- API: FastAPI server
- Importable: `from music_brain import EmotionMapper`

### 4. Documentation Site
- 93+ production guides
- API documentation
- User manuals
- Development guides

### 5. MCP Servers
- MCP TODO server (cross-AI task management)
- MCP Workstation (multi-AI orchestration)
- MCP Music Brain (music intelligence)

---

## Success Metrics

- ✅ All tests passing (Python, C++, TypeScript)
- ✅ Plugin loads in Logic Pro without crashes
- ✅ Desktop app builds on macOS
- ✅ Emotion thesaurus consistency across languages
- ✅ MIDI generation produces valid files
- ✅ Documentation is complete and organized
- ✅ Build scripts work on fresh machine

---

## Risk Mitigation

### Risk: Breaking kellymidi plugin
**Mitigation**:
- Keep backup of working plugin
- Test after each change
- Maintain separate plugin build

### Risk: Conflicting dependencies
**Mitigation**:
- Use virtual environment for Python
- Lock dependency versions
- Document conflicts

### Risk: Data inconsistency
**Mitigation**:
- Single source of truth for JSON files
- Validation scripts
- Integration tests

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Repo setup + Data merge | Unified repo, consolidated JSON files |
| 2 | Code merge (Python + C++) | Merged engines, passing unit tests |
| 3 | UI port + Build system | React UI with JUCE components ported |
| 4 | Docs + Testing | Complete documentation, all tests green |

**Total Duration**: 4 weeks for complete integration

---

## Next Steps

1. **Review this plan** - Does it align with your vision?
2. **Choose base directory** - `kelly-daw-unified` or different name?
3. **Archive decision** - Keep originals or delete after merge?
4. **Priority components** - Which to merge first?

**Ready to begin Phase 1?**
