# Distribution Zip Creation Scripts

This directory contains scripts to create distribution packages for both macOS and iOS versions of Bulling.

## Quick Start

### Create Both Distribution Zips (macOS only)
```bash
./create_distribution_zips.sh
```
This creates:
- `dist/Bulling-macOS.zip` - macOS application bundle
- `dist/Bulling-iOS.zip` - iOS source files package

### Create Individual Zips

#### macOS Distribution Zip
```bash
./create_macos_zip.sh
```
**Requirements**: Must run on macOS
**Output**: `dist/Bulling-macOS.zip`
**Contains**: Complete macOS application bundle

#### iOS Source Files Zip
```bash
./create_ios_zip.sh
```
**Requirements**: None (can run on any platform)
**Output**: `dist/Bulling-iOS.zip`
**Contains**: All Swift source files, setup guide, and instructions

## What Each Zip Contains

### Bulling-macOS.zip
- `Bulling.app` - Complete macOS application
- Double-click to run, no installation needed
- Size: ~60-80 MB (compressed)

**For users:**
1. Download and unzip
2. Drag `Bulling.app` to Applications folder
3. Double-click to play

### Bulling-iOS.zip
- All Swift source files (*.swift)
- iOS_SETUP_GUIDE.md
- README.md
- SETUP.txt with quick instructions
- Size: ~20 KB

**For developers:**
1. Download and unzip
2. Open Xcode
3. Create new iOS App project
4. Copy all .swift files to project
5. Build and run

## Scripts Overview

| Script | Purpose | Platform | Output |
|--------|---------|----------|--------|
| `create_distribution_zips.sh` | Create both zips | macOS (both) / Linux (iOS only) | Both zip files |
| `create_macos_zip.sh` | Create macOS zip | macOS only | `dist/Bulling-macOS.zip` |
| `create_ios_zip.sh` | Create iOS zip | Any | `dist/Bulling-iOS.zip` |

## File Locations

All zip files are created in the `dist/` directory:
```
dist/
├── Bulling.app/           # (macOS only - built by create_macos_zip.sh)
├── Bulling-macOS.zip      # macOS distribution
└── Bulling-iOS.zip        # iOS source files
```

## Build Process

### macOS Build Steps
1. Install Python dependencies (`pip install -r requirements.txt`)
2. Run `build_macos_app.sh` to create `Bulling.app`
3. Zip the application bundle
4. Output: `dist/Bulling-macOS.zip`

### iOS Package Steps
1. Copy Swift source files to temporary directory
2. Add documentation (iOS_SETUP_GUIDE.md, README.md)
3. Create SETUP.txt with instructions
4. Zip everything together
5. Output: `dist/Bulling-iOS.zip`

## Distribution

### Sharing the Files

**Via GitHub Releases:**
```bash
# Create both zips
./create_distribution_zips.sh

# Upload dist/Bulling-macOS.zip and dist/Bulling-iOS.zip to GitHub release
```

**Via Direct Download:**
- Upload zip files to cloud storage (Dropbox, Google Drive, etc.)
- Share download links with users

**Via Website:**
- Host zip files on your website
- Provide download buttons/links

## Troubleshooting

### "Permission denied" when running scripts
```bash
chmod +x create_distribution_zips.sh
chmod +x create_macos_zip.sh
chmod +x create_ios_zip.sh
```

### macOS script fails on Linux
This is expected - the macOS build requires macOS. Use `create_ios_zip.sh` instead, or run on a Mac.

### Missing dist/ directory
The scripts automatically create the `dist/` directory if it doesn't exist.

### Zip files are too large
- macOS zip: Normal size is 60-80 MB (contains full Python + Qt libraries)
- iOS zip: Should be ~20 KB (just source files)

## Advanced Usage

### Custom Zip Names
Edit the scripts to change output filenames:
- In `create_macos_zip.sh`: Change `Bulling-macOS.zip`
- In `create_ios_zip.sh`: Change `Bulling-iOS.zip`

### Include Additional Files
Edit `create_ios_zip.sh` to add more files:
```bash
cp YOUR_FILE.txt "$package_dir/"
```

### Exclude Files from iOS Zip
Edit the `cp` command in `create_ios_zip.sh` to be more selective.

## Version Control

The following are excluded from version control (via .gitignore):
- `dist/` directory
- `*.zip` files
- `build/` directory

Zip files should be created during release process, not committed to the repository.

## Related Documentation

- [DISTRIBUTION_GUIDE.md](DISTRIBUTION_GUIDE.md) - Complete distribution guide
- [MACOS_APP_GUIDE.md](MACOS_APP_GUIDE.md) - macOS app setup and usage
- [iOS_SETUP_GUIDE.md](iOS_SETUP_GUIDE.md) - iOS development guide
- [README.md](README.md) - Main project documentation

## Quick Reference

```bash
# Create both zips (macOS only)
./create_distribution_zips.sh

# Create only macOS zip (macOS only)
./create_macos_zip.sh

# Create only iOS zip (any platform)
./create_ios_zip.sh

# View created zips
ls -lh dist/*.zip

# Test iOS zip contents
unzip -l dist/Bulling-iOS.zip
```

## Support

For issues or questions:
1. Check this README
2. Review the main [DISTRIBUTION_GUIDE.md](DISTRIBUTION_GUIDE.md)
3. Open an issue on GitHub

---

**Ready to create distribution packages? Run `./create_distribution_zips.sh`!** 📦
