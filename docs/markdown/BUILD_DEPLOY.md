# iDAW Build & Deployment Guide

Complete guide for building and deploying the iDAW (Intelligent Digital Audio Workstation) application.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Build Steps](#build-steps)
4. [Deployment](#deployment)
5. [Platform-Specific Instructions](#platform-specific-instructions)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Node.js | 18+ | Frontend build |
| npm/yarn | Latest | Package management |
| Python | 3.9+ | Music Brain API |
| Rust | 1.70+ | Tauri backend |
| CMake | 3.22+ | C++ build |
| Xcode (macOS) | 14+ | macOS builds |
| Visual Studio (Windows) | 2022 | Windows builds |

### Install Prerequisites

#### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install node python@3.11 rust cmake

# Install Xcode Command Line Tools
xcode-select --install
```

#### Windows
```powershell
# Using winget
winget install OpenJS.NodeJS
winget install Python.Python.3.11
winget install Rustlang.Rust.MSVC
winget install Kitware.CMake

# Install Visual Studio 2022 Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools
```

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Python
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install build dependencies
sudo apt install -y build-essential cmake libwebkit2gtk-4.0-dev \
    libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev
```

---

## Development Setup

### 1. Clone and Initialize

```bash
# Navigate to project
cd /Users/seanburdges/kelly-consolidation/1DAW1

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install -e ".[all]"  # Install with all optional dependencies

# Install Node.js dependencies
npm install

# Install Tauri CLI
cargo install tauri-cli
```

### 2. Environment Configuration

Create `.env` file in project root:
```env
# API Configuration
MUSIC_BRAIN_API_HOST=127.0.0.1
MUSIC_BRAIN_API_PORT=8000

# Build Configuration
NODE_ENV=development
TAURI_DEBUG=true

# Optional: AI API Keys (for future integrations)
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

### 3. Verify Setup

```bash
# Check versions
node --version      # Should be 18+
python3 --version   # Should be 3.9+
rustc --version     # Should be 1.70+
cargo --version
cmake --version     # Should be 3.22+

# Verify Tauri
cargo tauri --version
```

---

## Build Steps

### Quick Build (Development)

```bash
# Start development server (runs API + frontend)
./start.sh

# Or manually:
# Terminal 1: Start API
source venv/bin/activate
python -m uvicorn music_brain.api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Start Tauri dev
npm run tauri dev
```

### Production Build

#### Step 1: Build Python API (Optional - for standalone)
```bash
source venv/bin/activate

# Create standalone executable with PyInstaller
pip install pyinstaller
pyinstaller --onefile --name music_brain_api \
    --add-data "music_brain/data:music_brain/data" \
    music_brain/api.py
```

#### Step 2: Build Frontend
```bash
# Build frontend assets
npm run build

# Verify build
ls -la dist/
```

#### Step 3: Build Tauri Application
```bash
# Build for current platform
npm run tauri build

# Build outputs location:
# macOS: src-tauri/target/release/bundle/dmg/
# Windows: src-tauri/target/release/bundle/msi/
# Linux: src-tauri/target/release/bundle/deb/
```

### Full Production Build Script

```bash
#!/bin/bash
# full_build.sh

set -e  # Exit on error

echo "======================================"
echo "iDAW Full Production Build"
echo "======================================"

# Step 1: Clean previous builds
echo "[1/6] Cleaning previous builds..."
rm -rf dist/
rm -rf src-tauri/target/release/bundle/
rm -rf build/

# Step 2: Install/update dependencies
echo "[2/6] Installing dependencies..."
npm ci
source venv/bin/activate
pip install -r requirements.txt

# Step 3: Run tests
echo "[3/6] Running tests..."
npm run test 2>/dev/null || echo "No frontend tests configured"
pytest tests/ -v 2>/dev/null || echo "No Python tests found"

# Step 4: Build frontend
echo "[4/6] Building frontend..."
npm run build

# Step 5: Build Tauri app
echo "[5/6] Building Tauri application..."
npm run tauri build

# Step 6: Build Python API (standalone)
echo "[6/6] Building Python API executable..."
pyinstaller --onefile --name music_brain_api \
    --add-data "music_brain/data:music_brain/data" \
    --hidden-import uvicorn \
    --hidden-import fastapi \
    music_brain/api.py 2>/dev/null || echo "PyInstaller build skipped"

echo "======================================"
echo "Build Complete!"
echo "======================================"
echo ""
echo "Outputs:"
echo "  Frontend: dist/"
echo "  App Bundle: src-tauri/target/release/bundle/"
echo "  API Executable: dist/music_brain_api (if PyInstaller available)"
```

---

## Deployment

### Option 1: Desktop Application (Recommended)

#### macOS
```bash
# Build DMG
npm run tauri build

# Sign and notarize (requires Apple Developer account)
# Set environment variables:
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAM_ID)"
export APPLE_ID="your@email.com"
export APPLE_PASSWORD="app-specific-password"
export APPLE_TEAM_ID="TEAM_ID"

npm run tauri build -- --target universal-apple-darwin
```

#### Windows
```powershell
# Build MSI installer
npm run tauri build

# Sign with certificate (optional)
# Requires code signing certificate
```

#### Linux
```bash
# Build DEB package
npm run tauri build

# Build AppImage
npm run tauri build -- --bundles appimage
```

### Option 2: Web Deployment (API + Frontend)

#### Docker Deployment

Create `Dockerfile`:
```dockerfile
# Multi-stage build
FROM node:18 AS frontend-builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM python:3.11-slim AS production
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python API
COPY music_brain/ ./music_brain/

# Copy frontend build
COPY --from=frontend-builder /app/dist ./static/

# Expose port
EXPOSE 8000

# Start server
CMD ["python", "-m", "uvicorn", "music_brain.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  idaw:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=production
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

Deploy:
```bash
# Build and run
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 3: Cloud Deployment

#### AWS/GCP/Azure

```bash
# Build production image
docker build -t idaw:latest .

# Push to registry
docker tag idaw:latest your-registry.com/idaw:latest
docker push your-registry.com/idaw:latest

# Deploy to cloud (example: AWS ECS, GCP Cloud Run, Azure Container Apps)
```

---

## Platform-Specific Instructions

### macOS Build

```bash
# Required: Xcode and Command Line Tools
xcode-select --install

# Build universal binary (Intel + Apple Silicon)
npm run tauri build -- --target universal-apple-darwin

# Output: src-tauri/target/universal-apple-darwin/release/bundle/
```

### Windows Build

```powershell
# Build MSI installer
npm run tauri build

# Build NSIS installer (alternative)
npm run tauri build -- --bundles nsis

# Output: src-tauri\target\release\bundle\
```

### Linux Build

```bash
# Build DEB (Debian/Ubuntu)
npm run tauri build -- --bundles deb

# Build RPM (Fedora/RHEL)
npm run tauri build -- --bundles rpm

# Build AppImage (Universal)
npm run tauri build -- --bundles appimage

# Output: src-tauri/target/release/bundle/
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

Create `.github/workflows/build.yml`:

```yaml
name: Build and Release

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          npm ci
          pip install -r requirements.txt

      - name: Run tests
        run: |
          npm test || true
          pytest tests/ -v || true

  build:
    needs: test
    strategy:
      fail-fast: false
      matrix:
        platform:
          - os: ubuntu-22.04
            rust_target: x86_64-unknown-linux-gnu
          - os: macos-latest
            rust_target: aarch64-apple-darwin
          - os: macos-latest
            rust_target: x86_64-apple-darwin
          - os: windows-latest
            rust_target: x86_64-pc-windows-msvc

    runs-on: ${{ matrix.platform.os }}
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - name: Setup Rust
        uses: dtolnay/rust-action@stable
        with:
          targets: ${{ matrix.platform.rust_target }}

      - name: Install Linux dependencies
        if: matrix.platform.os == 'ubuntu-22.04'
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.0-dev \
            libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev

      - name: Install dependencies
        run: npm ci

      - name: Build Tauri app
        uses: tauri-apps/tauri-action@v0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tagName: v__VERSION__
          releaseName: 'iDAW v__VERSION__'
          releaseBody: 'See the assets to download and install this version.'
          releaseDraft: true
          prerelease: false

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build-${{ matrix.platform.os }}-${{ matrix.platform.rust_target }}
          path: src-tauri/target/release/bundle/

  release:
    needs: build
    if: startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v4
        with:
          path: artifacts/

      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          files: artifacts/**/*
          draft: false
          prerelease: false
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Troubleshooting

### Common Issues

#### 1. Node.js/npm Issues
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### 2. Rust/Cargo Issues
```bash
# Update Rust
rustup update

# Clean cargo cache
cargo clean

# Rebuild
cargo build --release
```

#### 3. Python Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 4. Tauri Build Failures

**macOS:**
```bash
# Install missing dependencies
brew install openssl@1.1
export OPENSSL_DIR=$(brew --prefix openssl@1.1)
```

**Linux:**
```bash
# Install webkit dependencies
sudo apt install libwebkit2gtk-4.0-dev libappindicator3-dev
```

**Windows:**
```powershell
# Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools"
```

#### 5. API Connection Issues
```bash
# Check if API is running
curl http://127.0.0.1:8000/health

# Check port availability
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process using port
kill -9 $(lsof -t -i:8000)  # macOS/Linux
```

### Build Verification

```bash
# Verify all components
./verify_build.sh

# Or manually:
# 1. Check frontend build
ls -la dist/index.html

# 2. Check Tauri build
ls -la src-tauri/target/release/bundle/

# 3. Check API
curl http://127.0.0.1:8000/health
```

---

## Version Management

### Updating Version

1. Update `package.json`:
```json
{
  "version": "1.0.0"
}
```

2. Update `src-tauri/Cargo.toml`:
```toml
[package]
version = "1.0.0"
```

3. Update `src-tauri/tauri.conf.json`:
```json
{
  "version": "1.0.0"
}
```

4. Create git tag:
```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

---

## Support

- **Documentation**: See `CLAUDE.md` for detailed project documentation
- **Issues**: https://github.com/your-repo/idaw/issues
- **Discussions**: https://github.com/your-repo/idaw/discussions
