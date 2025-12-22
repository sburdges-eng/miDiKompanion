# Workflow

This document is the canonical, repeatable workflow for working in this repo. The repository contains multiple tools and apps, so use the sections that match what you are working on.

## Environment Setup

### Core tools
- **Git** and **bash** (for repo scripts).
- **Make** (for the git updater build).

### Optional toolchains (only if needed)
- **Python 3.8+** for the automation and analysis tooling.
- **Node.js + npm** for the web/desktop UI.
- **CMake 3.20+** and a **C++20 compiler** for native builds.

### Install dependencies
- **Python**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Node**
  ```bash
  npm install
  ```
- **OS helpers** (optional)
  ```bash
  ./install_macos.sh
  ./install_linux.sh
  # Windows (PowerShell)
  ./install_windows.ps1
  ```

## Run / Build

### Git updater (Makefile build)
```bash
make
./dist/git-update.sh
```

### Quick build helpers
```bash
./build.sh
./build_quick.sh
```

### Windows quick helper
```powershell
./build_quick.bat
```

### Web/desktop UI (Vite/Tauri)
```bash
npm run dev
npm run build
npm run preview
```

### Native app builds
```bash
./build_macos_app.sh
./build_ios_app.sh
./build_standalone.sh
```

## Develop

### Code layout (high-level)
- **Shell automation**: `build.sh`, `build_quick.sh`, `create_distribution_zips.sh`.
- **Python orchestration & tools**: `app.py`, `server.py`, `cli.py`, `launcher.py`, plus modules in `modules/`, `tools/`, and `python/`.
- **C++/native**: core sources under `src/`, `core/`, `include/`, and plugin-related files near `Plugin*`/`WavetableSynth*`.
- **Web/Tauri UI**: `src/`, `src-tauri/`, `public/`, `vite.config.ts`.
- **Tests**: `tests/`, `tests_*` folders, and `test_*.py` at the repo root.

### Conventions
- Prefer existing scripts (`build.sh`, `build_standalone.sh`, `create_distribution_zips.sh`) over ad-hoc commands.
- Keep new automation inside `tools/` or `scripts`-style files; avoid duplicating existing helpers.

## Test

### Python tests
```bash
python -m pytest
```

### C++/CMake tests (if configured)
```bash
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Release / Distribution

### Git updater dist artifacts
```bash
make
ls dist/
```

### Packaged app artifacts
```bash
./create_distribution_zips.sh
./create_macos_zip.sh
./create_ios_zip.sh
```

For standalone packaging, run:
```bash
./build_standalone.py
```

---

Related detailed docs: `BUILD.md`, `SETUP_GUIDE.md`, and `QUICKSTART.md`.
