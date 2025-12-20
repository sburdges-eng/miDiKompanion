# 📦 Bulling - Distribution & Build Guide

Complete guide for building and distributing Bulling apps for macOS and iOS.

---

## ⚠️ **IMPORTANT: Personal Use Only**

**This software is licensed for PERSONAL USE ONLY.**

All distribution methods described in this guide are intended for:
- ✅ Personal entertainment and use
- ✅ Sharing with friends and family for personal use
- ✅ Personal device installation
- ✅ Personal development and learning

**NOT permitted:**
- ❌ Commercial use or distribution
- ❌ Publishing to app stores (Apple App Store, Google Play, etc.)
- ❌ Business or organizational use
- ❌ Monetization in any form

📖 **See [LICENSE.txt](LICENSE.txt) and [PERSONAL_USE_README.md](PERSONAL_USE_README.md) for complete terms**

---

## 🚀 Quick Start - Personal Distribution

### Standalone Build (Recommended for Personal Use)

```bash
# Build unsigned, standalone apps for personal distribution
./build_standalone.sh all

# Or build specific platforms:
./build_standalone.sh macos       # macOS app only
./build_standalone.sh ios         # iOS simulator only
./build_standalone.sh ios-device  # iOS device (unsigned)
```

This creates ready-to-use apps in the `dist/` directory that you can share 
with friends and family for personal use.

### Create Distribution Zips

```bash
# Create both macOS and iOS distribution packages
./create_distribution_zips.sh
```

**Output:**
- `dist/Bulling-macOS.zip` - macOS application bundle (macOS only)
- `dist/Bulling-iOS.zip` - iOS source files package

**Or create individually:**
```bash
# macOS distribution (requires macOS)
./create_macos_zip.sh

# iOS source files package (any platform)
./create_ios_zip.sh
```

📖 **[Distribution Scripts README](DISTRIBUTION_SCRIPTS_README.md)** - Complete documentation for automated zip creation

---

## 🖥️ macOS Distribution

### Prerequisites
- macOS 10.13 or later
- Python 3.9 or later installed
- Terminal access

### Build Process

#### 1. Install Dependencies
```bash
cd Pentagon-core-100-things
pip3 install -r requirements.txt
```

#### 2. Generate App Icon (Optional)
```bash
# Generate SVG icon
python3 generate_icon.py

# Convert SVG to PNG using:
# - Online tool: https://cloudconvert.com/svg-to-png
# - macOS Preview app
# - Command: brew install librsvg && rsvg-convert -w 1024 -h 1024 bulling_icon.svg > bulling_icon.png

# Create .icns file
./create_icon.sh bulling_icon.png
```

#### 3. Build the App
```bash
./build_macos_app.sh
```

This creates: `dist/Bulling.app`

#### 4. Test the App
```bash
# Open directly
open dist/Bulling.app

# Or right-click → Open (first time only)
```

#### 5. Create Distribution Package

**Automated (Recommended):**
```bash
./create_macos_zip.sh
```

**Manual:**
```bash
cd dist
zip -r Bulling-macOS.zip Bulling.app
```

### Distribution Options

#### Option A: Direct File Sharing
1. Share `Bulling-macOS-v1.0.zip`
2. Users unzip and drag to Applications folder
3. Users may need to approve in Security settings (first launch)

#### Option B: Code Signing (Recommended for wider distribution)
```bash
# Requires Apple Developer account ($99/year)
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: YOUR NAME" \
  dist/Bulling.app

# Create notarized package
# Follow Apple's notarization guide
```

#### Option C: DMG Installer (Professional)
```bash
# Install create-dmg
brew install create-dmg

# Create DMG
create-dmg \
  --volname "Bulling Installer" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "Bulling.app" 175 120 \
  --hide-extension "Bulling.app" \
  --app-drop-link 425 120 \
  "Bulling-Installer.dmg" \
  "dist/"
```

### File Sizes
- **Bulling.app**: ~100-150 MB
- **Bulling.zip**: ~60-80 MB (compressed)
- **DMG**: ~70-90 MB

---

## 📱 iOS Distribution

### Prerequisites
- macOS with Xcode 14+ installed
- Apple Developer account (for device testing/App Store)
- iOS device or Simulator

### Source Files Package (Simplified Distribution)

**Quick Method - Automated:**
```bash
# Create iOS source files zip
./create_ios_zip.sh
```

**Output:** `dist/Bulling-iOS.zip`

**Contains:**
- All Swift source files (*.swift)
- iOS_SETUP_GUIDE.md
- README.md
- SETUP.txt with quick instructions

**For developers receiving the zip:**
1. Download and unzip `Bulling-iOS.zip`
2. Follow SETUP.txt instructions
3. Open Xcode and create new iOS App
4. Copy .swift files to project
5. Build and run

### Full Build Process

#### 1. Setup Xcode Project
```bash
# Use automated zip (recommended)
./create_ios_zip.sh

# OR manually follow iOS_SETUP_GUIDE.md
1. Create new iOS App in Xcode
2. Name it "Bulling"
3. Copy all .swift files from iOS/Bulling/
```

#### 2. Configure App Icon
1. Export `bulling_icon.svg` to PNG at multiple sizes:
   - 1024×1024 (App Store)
   - 180×180 (iPhone)
   - 120×120 (smaller)
   - 60×60 (notifications)

2. In Xcode: Assets.xcassets → AppIcon
3. Drag each size to appropriate slot

#### 3. Build Settings
- **Bundle Identifier**: `com.yourname.bulling`
- **Version**: `1.0`
- **Build**: `1`
- **Deployment Target**: iOS 15.0
- **Signing**: Select your team

#### 4. Test Build
```
# For Simulator
⌘R (Command + R)

# For Device
1. Connect device
2. Select device in Xcode
3. Trust computer on device
4. ⌘R to build and run
```

### Distribution Options

#### Option A: TestFlight (Beta Testing)
```
1. Archive: Product → Archive
2. Distribute App → TestFlight & App Store
3. Upload to App Store Connect
4. Add beta testers
5. Share TestFlight link
```

**Pros**: Easy testing, up to 10,000 testers
**Cons**: Requires Apple Developer account

#### Option B: App Store
```
1. Create app in App Store Connect
2. Archive in Xcode
3. Upload to App Store
4. Submit for review
5. Wait for approval (1-3 days typically)
6. Release when approved
```

**Pros**: Professional distribution, auto-updates
**Cons**: Review process, $99/year developer fee

#### Option C: Ad-Hoc Distribution
```
1. Archive your app
2. Export → Ad Hoc
3. Share .ipa file
4. Install using:
   - Apple Configurator
   - Xcode Devices window
   - Enterprise MDM (if corporate)
```

**Pros**: Direct distribution to testers
**Cons**: Limited to 100 devices per year

#### Option D: Enterprise Distribution
- Requires Apple Developer Enterprise account ($299/year)
- Internal distribution only
- Not for public apps

### File Sizes
- **.ipa**: ~5-10 MB
- **App Store listing**: ~8-12 MB (with assets)

---

## 🌐 Platform-Specific Notes

### macOS

**Security Gatekeeper**
- Users will see warning on first launch
- **Solution**: Right-click → Open (first time)
- **Or**: Sign with Developer ID (recommended)

**Rosetta 2 (Apple Silicon)**
- App works on both Intel and Apple Silicon Macs
- Universal binary can be created with py2app options

**Minimum macOS Version**
- Built app targets: macOS 10.13+
- Can be adjusted in setup.py

### iOS

**Device Requirements**
- iPhone/iPad running iOS 15.0+
- Works on all screen sizes
- Optimized for both iPhone and iPad

**App Store Requirements**
- Privacy policy (if collecting data)
- Screenshots for each device type
- App description and keywords
- Age rating

**Permissions**
- No special permissions needed
- Fully offline app

---

## 📊 Version Naming

### Semantic Versioning
```
MAJOR.MINOR.PATCH

1.0.0 - Initial release
1.0.1 - Bug fix
1.1.0 - New feature
2.0.0 - Major changes
```

### Current Versions
- **macOS**: v1.0.0
- **iOS**: v1.0.0

---

## 🔐 Code Signing (Advanced)

### macOS Code Signing
```bash
# Sign the app
codesign --deep --force --verify --verbose \
  --sign "Developer ID Application: YOUR NAME (TEAM_ID)" \
  --options runtime \
  dist/Bulling.app

# Verify signature
codesign --verify --deep --strict --verbose=2 dist/Bulling.app
spctl -a -t exec -vv dist/Bulling.app
```

### iOS Code Signing
- Automatic in Xcode with team selected
- Uses provisioning profiles
- Managed in Xcode: Signing & Capabilities

---

## 🎨 App Store Assets

### macOS (if distributing via Mac App Store)
- **Icon**: 1024×1024 PNG
- **Screenshots**: Various Mac screen sizes
- **Description**: 170 char subtitle + 4000 char description
- **Keywords**: Up to 100 characters

### iOS App Store
- **Icon**: 1024×1024 PNG (no alpha channel)
- **Screenshots**: Required for:
  - 6.7" iPhone (iPhone 14 Pro Max)
  - 5.5" iPhone (or larger)
  - 12.9" iPad Pro
- **App Previews**: Optional videos
- **Privacy Info**: Required

---

## 📝 Release Checklist

### macOS Release
- [ ] Code complete and tested
- [ ] Version number updated in setup.py
- [ ] App icon created (app_icon.icns)
- [ ] Build with `./build_macos_app.sh`
- [ ] Test on clean Mac (if possible)
- [ ] Create zip archive with `./create_macos_zip.sh`
- [ ] (Optional) Code sign
- [ ] (Optional) Create DMG
- [ ] Upload to distribution platform
- [ ] Update README with download link

### iOS Release
- [ ] Code complete and tested
- [ ] Version and build numbers updated
- [ ] All .swift files in iOS/Bulling/ directory
- [ ] Create source package with `./create_ios_zip.sh`
- [ ] (Optional) Build in Xcode for App Store:
  - [ ] App icon added to Assets.xcassets
  - [ ] Screenshots prepared
  - [ ] Privacy policy (if needed)
  - [ ] Archive in Xcode
  - [ ] Upload to App Store Connect
  - [ ] Fill in App Store metadata
  - [ ] Submit for review
  - [ ] Monitor review status
  - [ ] Release when approved

---

## 🚀 Quick Distribution Commands

### Both Platforms (Automated)
```bash
# Create both distribution zips at once
./create_distribution_zips.sh

# Creates:
# - dist/Bulling-macOS.zip (macOS app)
# - dist/Bulling-iOS.zip (iOS source files)
```

### macOS Only
```bash
# Automated (recommended)
./create_macos_zip.sh
echo "✅ Ready to distribute: dist/Bulling-macOS.zip"

# Manual
./build_macos_app.sh
cd dist
zip -r Bulling-macOS.zip Bulling.app
echo "✅ Ready to distribute: Bulling-macOS.zip"
```

### iOS Only
```bash
# Source files package (automated)
./create_ios_zip.sh
echo "✅ Ready to distribute: dist/Bulling-iOS.zip"

# OR build in Xcode for App Store
1. Product → Archive
2. Distribute App
3. Choose distribution method
4. Follow prompts
```

---

## 📈 Distribution Platforms

### For macOS

**Direct Download**
- GitHub Releases
- Your own website
- Cloud storage (Dropbox, Google Drive)

**Mac App Store** (requires paid developer account)
- Professional distribution
- Auto-updates
- User trust

**Homebrew Cask** (advanced)
- Popular for developer tools
- Requires maintaining cask formula

### For iOS

**App Store** (primary)
- Official Apple distribution
- Required for public distribution

**TestFlight**
- Beta testing
- Internal and external testers

**Enterprise** (internal company apps)
- Requires Enterprise account
- Internal distribution only

---

## 💰 Cost Summary

### Free Options
- ✅ Direct macOS app sharing
- ✅ iOS Simulator testing
- ✅ Development and building

### Paid Options
- **Apple Developer** ($99/year)
  - iOS device testing
  - App Store distribution
  - TestFlight
  - macOS code signing

- **Apple Developer Enterprise** ($299/year)
  - Internal iOS distribution
  - Not for public apps

---

## 🎯 Best Practices

1. **Version Control**
   - Tag releases in Git
   - Keep changelog updated

2. **Testing**
   - Test on multiple devices/macOS versions
   - Beta test with real users

3. **Updates**
   - Regular bug fixes
   - Feature additions based on feedback

4. **Documentation**
   - Keep guides up to date
   - Include troubleshooting tips

5. **Support**
   - Respond to user feedback
   - Fix critical bugs quickly

---

## 🆘 Common Distribution Issues

### macOS

**"App is damaged" error**
- **Cause**: Gatekeeper blocking unsigned app
- **Fix**: Right-click → Open, or sign with Developer ID

**App won't launch**
- **Cause**: Missing dependencies or architecture mismatch
- **Fix**: Rebuild with proper py2app settings

### iOS

**"Unable to install"**
- **Cause**: Provisioning profile issues
- **Fix**: Check signing in Xcode, regenerate profiles

**App Store rejection**
- **Cause**: Various (review guidelines)
- **Fix**: Address specific feedback from Apple

---

## 📞 Support Resources

### macOS
- [py2app Documentation](https://py2app.readthedocs.io/)
- [PySide6 Documentation](https://doc.qt.io/qtforpython/)
- [Code Signing Guide](https://developer.apple.com/support/code-signing/)

### iOS
- [App Distribution Guide](https://developer.apple.com/documentation/xcode/distributing-your-app-for-beta-testing-and-releases)
- [App Store Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [TestFlight Documentation](https://developer.apple.com/testflight/)

---

**Ready to distribute? Follow this guide step by step!** 📦🚀
