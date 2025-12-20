# 🎳 Bulling - Personal Use Distribution Complete

**Version:** 1.0.0  
**License:** Personal Use Only (Non-Commercial)  
**Last Updated:** December 2024

---

## ✅ Project Status: COMPLETE

This repository now contains a **complete standalone app distribution system** for 
personal, non-commercial use only.

---

## 📦 What's Included

### Applications
- ✅ **macOS Desktop App** (Native Swift or Python/Qt6)
- ✅ **iOS Mobile App** (Native Swift/SwiftUI)
- ✅ **Standalone Build System** (Unsigned, personal use)

### Documentation
- ✅ **LICENSE.txt** - Personal use license
- ✅ **PERSONAL_USE_README.md** - Complete personal use guide
- ✅ **QUICK_DOWNLOAD.md** - Fast download & install guide
- ✅ **README.md** - Main project documentation
- ✅ **MACOS_APP_GUIDE.md** - macOS setup and usage
- ✅ **iOS_SETUP_GUIDE.md** - iOS setup and sideloading
- ✅ **DISTRIBUTION_GUIDE.md** - Build and distribution guide

### Build System
- ✅ **build_standalone.sh** - One-command build for all platforms
- ✅ **build_macos_app.sh** - macOS Python app builder
- ✅ **build_macos_native.sh** - macOS Swift app builder
- ✅ **build_ios_app.sh** - iOS app builder
- ✅ **create_distribution_zips.sh** - Distribution package creator

---

## 🚀 Quick Start for Users

### Download and Install (Easiest)

**macOS:**
```bash
# Build standalone app
./build_standalone.sh macos

# App created at: dist/Bulling-macOS.app
# Share the .app or .zip with friends/family
```

**iOS Simulator:**
```bash
# Build for testing
./build_standalone.sh ios

# Install to simulator
xcrun simctl install booted dist/Bulling-iOS-Simulator.app
```

**iOS Device (Personal Sideloading):**
```bash
# Build unsigned IPA
./build_standalone.sh ios-device

# Use AltStore, Sideloadly, or your Apple Developer certificate
# to install: dist/Bulling-iOS-Unsigned.ipa
```

---

## ⚠️ IMPORTANT: Personal Use Only

### What You CAN Do ✅
- Install on your personal devices (Mac, iPhone, iPad)
- Share with friends and family for their personal use
- Modify the code for personal learning and use
- Create personal backups
- Use for home entertainment

### What You CANNOT Do ❌
- Publish to App Store or Google Play
- Use for commercial purposes or business
- Sell or monetize the software
- Distribute via TestFlight for public/commercial use
- Remove the personal use license

**Why?** This software is provided free for personal enjoyment only. Commercial 
use would violate the license terms.

---

## 📋 License Summary

**Full License:** [LICENSE.txt](LICENSE.txt)

**Type:** Personal Use Only (Non-Commercial)

**Third-Party Components:**
- **PySide6 (Qt6):** LGPL licensed - must comply with LGPL terms
- **SwiftUI:** Part of Apple's SDK - subject to Apple's developer agreement

**Copyright:** © 2025 Bulling Project. Personal Use Only.

---

## 🛠️ Technical Details

### System Requirements

**macOS App:**
- macOS 10.13 High Sierra or later
- No installation or dependencies needed for users
- ~100-150 MB file size (standalone bundle)

**iOS App:**
- iOS 15.0 or later
- iPhone and iPad compatible
- ~5-10 MB file size

### Build Requirements (Developers)
- macOS with Xcode 14+ (for building)
- Python 3.9+ (for Python-based builds)
- Git (for cloning repository)

---

## 📖 Key Documentation Files

### For End Users (Non-Technical)
1. **[QUICK_DOWNLOAD.md](QUICK_DOWNLOAD.md)** ⭐ Start here!
   - Simple download and install instructions
   - Troubleshooting for common issues
   - How to play guide

2. **[PERSONAL_USE_README.md](PERSONAL_USE_README.md)**
   - Complete personal use guide
   - All installation methods
   - iOS sideloading instructions

### For Developers
1. **[README.md](README.md)** - Main project overview
2. **[MACOS_APP_GUIDE.md](MACOS_APP_GUIDE.md)** - macOS build guide
3. **[iOS_SETUP_GUIDE.md](iOS_SETUP_GUIDE.md)** - iOS build guide
4. **[DISTRIBUTION_GUIDE.md](DISTRIBUTION_GUIDE.md)** - Distribution methods

### Legal
1. **[LICENSE.txt](LICENSE.txt)** - Complete license terms

---

## 🎯 Distribution Methods

### macOS
- **Direct sharing:** Share .app or .zip file
- **Cloud storage:** Upload to Dropbox, Google Drive, etc.
- **USB drive:** Copy and share offline
- **GitHub Releases:** Official distribution (if set up)

**Note:** Apps are unsigned. Users must right-click → Open first time.

### iOS
- **AltStore:** Free, 7-day signing, auto-refresh
- **Sideloadly:** Free, 7-day signing, simple drag-and-drop
- **Personal Dev Certificate:** $99/year, 1-year signing, no re-signing
- **Direct Xcode:** Free, 7-day signing, requires Mac connection

**Note:** Cannot publish to App Store due to personal use license.

---

## 🔒 Security & Privacy

### What Makes This Safe?
- ✅ **Open Source:** All code is visible and auditable
- ✅ **No Internet:** Completely offline, no network access
- ✅ **No Data Collection:** Zero telemetry or tracking
- ✅ **No Accounts:** No sign-up or login required
- ✅ **Local Storage:** Your data stays on your device
- ✅ **No Ads:** Clean, ad-free experience

### Why Unsigned?
- Apps are unsigned because this is for personal use only
- Signing requires Apple Developer account ($99/year)
- Not needed for personal distribution to friends/family
- Users can easily bypass security warning (right-click → Open)

---

## 🎮 Features

### Gameplay
- ✅ Traditional 10-pin bowling rules
- ✅ Strikes, spares, and proper scoring
- ✅ 10th frame bonus throws
- ✅ Perfect game support (300 points)
- ✅ Up to 8 players

### User Experience
- ✅ Bull head logo with dartboard eyes
- ✅ Animated splash screen (iOS)
- ✅ Interactive pin selection
- ✅ Real-time scorecard
- ✅ Auto-save functionality
- ✅ Modern, clean interface

### Technical
- ✅ Cross-platform (macOS Python & Swift, iOS)
- ✅ Offline functionality
- ✅ No dependencies for end users
- ✅ Small file size (iOS: 5-10 MB)
- ✅ Fast performance

---

## 🆘 Support & Troubleshooting

### macOS Issues
- **"App is damaged"** → Right-click → Open (first time only)
- **App won't launch** → Check macOS 10.13+ required
- **Missing icon** → App works fine without it

### iOS Issues
- **Can't install** → Check iOS 15.0+ required
- **App expires after 7 days** → Use AltStore for auto-refresh
- **Want to publish** → Cannot due to personal use license

### Getting Help
1. Check troubleshooting in respective guides
2. Review full documentation
3. Check GitHub Issues
4. Open new issue if needed

---

## 📊 Project Statistics

- **Lines of Code:** ~5,000+ (Swift + Python)
- **Documentation Pages:** 7 comprehensive guides
- **Supported Platforms:** macOS, iOS
- **Build Scripts:** 5 automated scripts
- **License Model:** Personal Use Only
- **Cost to Users:** FREE (personal use)
- **Dependencies:** Minimal (PySide6 for Python version)

---

## 🎉 Success Criteria Met

- ✅ **Complete Apps:** Both macOS and iOS fully functional
- ✅ **Easy Installation:** One-click/drag-and-drop for users
- ✅ **Comprehensive Docs:** 7 detailed guides
- ✅ **Clear Licensing:** Personal use only, clearly stated
- ✅ **Build System:** Automated, one-command builds
- ✅ **No Signing Required:** Unsigned for personal distribution
- ✅ **No App Store:** Removed all commercial distribution references
- ✅ **Professional Quality:** Production-ready code and UX

---

## 🚀 Getting Started

**New Users (Want to Play):**
1. Go to [QUICK_DOWNLOAD.md](QUICK_DOWNLOAD.md)
2. Download the app for your platform
3. Follow simple installation steps
4. Start bowling!

**Developers (Want to Build):**
1. Clone this repository
2. Run `./build_standalone.sh all`
3. Find apps in `dist/` folder
4. Read full guides for details

**Sharing with Others:**
1. Build the apps using standalone script
2. Share .app (macOS) or .ipa (iOS) files
3. Include link to PERSONAL_USE_README.md
4. Remind recipients: Personal use only!

---

## 📜 Legal Compliance

This project complies with:
- ✅ **PySide6 LGPL License:** Using official Qt bindings properly
- ✅ **Apple Developer Agreement:** Personal use, no App Store
- ✅ **Open Source Best Practices:** Clear licensing, documentation
- ✅ **Personal Use Restrictions:** No commercial exploitation

Users must comply with:
- Personal use license terms (see LICENSE.txt)
- PySide6/Qt LGPL requirements
- Apple's terms for sideloading/development

---

## 🎯 Conclusion

This project provides a **complete, standalone app distribution system** for 
personal, non-commercial use. Everything needed to build, distribute, and use 
the apps is included and documented.

**Key Points:**
- Free for personal use
- No App Store publishing
- No commercial use
- Share with friends and family
- Professional quality
- Comprehensive documentation

**Ready to bowl? Start with [QUICK_DOWNLOAD.md](QUICK_DOWNLOAD.md)!**

---

**Happy Bowling! 🎳🐂**

*Strike & Score with Bulling!*

---

For questions or issues, please check the documentation or open a GitHub issue.
