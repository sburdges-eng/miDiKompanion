# 🚀 Quick Distribution Guide

Get your Bulling app ready for distribution in minutes!

## One-Command Distribution

### Create Both Zips (macOS required for full build)

```bash
./create_distribution_zips.sh
```

**Output:**
- ✅ `dist/Bulling-macOS.zip` - Ready-to-run macOS app
- ✅ `dist/Bulling-iOS.zip` - iOS source files package

---

## Platform-Specific

### macOS App Distribution

```bash
./create_macos_zip.sh
```

**What it does:**
1. Runs `build_macos_app.sh` to create the app
2. Creates `dist/Bulling-macOS.zip`
3. Shows file size and success message

**Share with users:**
- Send them `dist/Bulling-macOS.zip`
- They unzip and drag to Applications
- Done! They can double-click to play

### iOS Source Files Package

```bash
./create_ios_zip.sh
```

**What it does:**
1. Packages all .swift files
2. Includes setup guides (README, iOS_SETUP_GUIDE)
3. Creates SETUP.txt with quick instructions
4. Creates `dist/Bulling-iOS.zip`

**Share with developers:**
- Send them `dist/Bulling-iOS.zip`
- They unzip and follow SETUP.txt
- They build in Xcode
- Done! They have the iOS app

---

## Real-World Examples

### Example 1: GitHub Release

```bash
# On macOS
./create_distribution_zips.sh

# Upload both files to GitHub Release
# Users can download either:
# - Bulling-macOS.zip for Mac users
# - Bulling-iOS.zip for iOS developers
```

### Example 2: Direct User Distribution

```bash
# Build macOS version
./create_macos_zip.sh

# Email dist/Bulling-macOS.zip to users
# Include instructions:
# 1. Unzip the file
# 2. Drag Bulling.app to Applications
# 3. Double-click to play
```

### Example 3: Developer Handoff

```bash
# Package iOS source
./create_ios_zip.sh

# Send dist/Bulling-iOS.zip to iOS developer
# They get:
# - All Swift source files
# - Complete setup guide
# - Quick start instructions
```

---

## File Sizes

| File | Size | Description |
|------|------|-------------|
| `Bulling-macOS.zip` | ~60-80 MB | Complete app with Qt libraries |
| `Bulling-iOS.zip` | ~20 KB | Just source code files |

---

## Verification

### Check What's in the Zips

```bash
# List macOS zip contents
unzip -l dist/Bulling-macOS.zip

# List iOS zip contents
unzip -l dist/Bulling-iOS.zip

# Test iOS zip extraction
cd /tmp
unzip ~/path/to/dist/Bulling-iOS.zip
ls Bulling-iOS/
```

### Test the Zips

```bash
# Test macOS app (on Mac)
cd /tmp
unzip ~/path/to/dist/Bulling-macOS.zip
open Bulling.app

# Test iOS package
cd /tmp
unzip ~/path/to/dist/Bulling-iOS.zip
cat Bulling-iOS/SETUP.txt
```

---

## Distribution Checklist

### Before Creating Zips

- [ ] Code is complete and tested
- [ ] Version numbers updated (if applicable)
- [ ] README and guides are up-to-date
- [ ] All .swift files are in iOS/Bulling/ directory

### After Creating Zips

- [ ] Test unzipping both files
- [ ] Test macOS app launches (if on Mac)
- [ ] Verify iOS zip contains all .swift files
- [ ] Check file sizes are reasonable
- [ ] Upload to distribution platform
- [ ] Share download links

---

## Troubleshooting

### "Permission denied" error

```bash
chmod +x create_distribution_zips.sh
chmod +x create_macos_zip.sh
chmod +x create_ios_zip.sh
```

### macOS script fails on Linux

Expected! macOS app must be built on macOS. Use:
```bash
./create_ios_zip.sh  # This works on any platform
```

### Missing dist/ directory

No problem! Scripts create it automatically.

### Zip files are missing

Check for error messages in the script output. Common causes:
- Not on macOS (for macOS zip)
- Missing dependencies (run `pip install -r requirements.txt`)
- Build failed (check error messages)

---

## Quick Reference

```bash
# Both (macOS only)
./create_distribution_zips.sh

# Just macOS (macOS only)
./create_macos_zip.sh

# Just iOS (any platform)
./create_ios_zip.sh

# View files
ls -lh dist/

# Clean up
rm -rf dist/
```

---

## Next Steps

1. **Create the zips** - Run the appropriate script
2. **Test them** - Unzip and verify contents
3. **Share them** - Upload to GitHub, website, or email
4. **Support users** - Share setup instructions

---

## More Information

- 📖 [DISTRIBUTION_SCRIPTS_README.md](DISTRIBUTION_SCRIPTS_README.md) - Detailed script documentation
- 📦 [DISTRIBUTION_GUIDE.md](DISTRIBUTION_GUIDE.md) - Complete distribution guide
- 🖥️ [MACOS_APP_GUIDE.md](MACOS_APP_GUIDE.md) - macOS app guide
- 📱 [iOS_SETUP_GUIDE.md](iOS_SETUP_GUIDE.md) - iOS setup guide

---

**Ready to distribute? Run `./create_distribution_zips.sh` now!** 🚀
