# Quick Start - Gatekeeper Fix

## 🚀 Launch Standalone App (Easiest)

Just run:
```bash
./scripts/launch_standalone.sh
```

This automatically:
1. ✅ Removes quarantine
2. ✅ Signs the app
3. ✅ Launches it

## 🔧 Fix All Builds

To fix Standalone, VST3, and AU plugins:
```bash
./scripts/fix_gatekeeper.sh
```

## 📋 Available Scripts

| Script | Purpose |
|--------|---------|
| `./scripts/launch_standalone.sh` | Fix & launch standalone app |
| `./scripts/fix_gatekeeper.sh` | Fix all builds (Standalone, VST3, AU) |
| `./scripts/build_and_install.sh` | Build & install plugins |

## 📖 Full Documentation

See `GATEKEEPER_FIX.md` for detailed manual steps and troubleshooting.
