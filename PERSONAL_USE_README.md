# 🎳 Bulling - Personal Use Distribution Guide

**IMPORTANT: This software is for PERSONAL USE ONLY**

---

## ⚠️ Personal Use Notice

This application is distributed for **personal, non-commercial use only**. 

### ✅ What You CAN Do:
- Install on your personal devices
- Use for personal entertainment and gaming
- Share with friends and family for personal use
- Modify the code for your own personal use
- Create personal backups

### ❌ What You CANNOT Do:
- Use for commercial purposes
- Sell or monetize the software
- Publish to app stores (Apple App Store, Google Play, etc.)
- Distribute for business or organizational use
- Remove this personal use restriction

---

## 📦 Quick Download & Install

### macOS App (Desktop)

**Option 1: Direct Download (Easiest)**
1. Download the standalone app (no installation required)
2. Unzip the file
3. Move `Bulling.app` to your Applications folder
4. Right-click → Open (first time only to bypass security warning)
5. Done! Start bowling! 🎳

**Option 2: Build from Source**
```bash
# Clone this repository
git clone https://github.com/sburdges-eng/Pentagon-core-100-things.git
cd Pentagon-core-100-things

# Build standalone app
./build_standalone.sh macos

# App will be in: dist/Bulling-macOS.app
```

### iOS App (iPhone/iPad)

**For iOS Simulator (Development/Testing)**
```bash
# Build for simulator
./build_standalone.sh ios

# Install to simulator
xcrun simctl install booted dist/Bulling-iOS-Simulator.app
```

**For Physical iOS Device (Personal Sideloading)**
```bash
# Build unsigned IPA
./build_standalone.sh ios-device

# Use a sideloading tool like:
# - AltStore (free, 7-day signing)
# - Sideloadly (free, 7-day signing)  
# - Your own Apple Developer certificate (annual fee, 1-year signing)
```

**Important:** You cannot publish this app to the App Store due to personal use restrictions.

---

## 🔒 Security & Privacy

- ✅ **No internet required** - Fully offline app
- ✅ **No data collection** - Your game data stays on your device
- ✅ **No tracking** - Completely private
- ✅ **No accounts** - Just download and play
- ✅ **Open source** - Inspect the code yourself

---

## 🚀 Building Standalone Apps

This repository includes a comprehensive build script for creating standalone,
unsigned applications for personal distribution:

```bash
# Build everything
./build_standalone.sh all

# Build macOS only
./build_standalone.sh macos

# Build iOS Simulator only
./build_standalone.sh ios

# Build iOS Device (unsigned) only
./build_standalone.sh ios-device
```

### What Gets Built:

**macOS:**
- `dist/Bulling-macOS.app` - Standalone macOS application
- `dist/Bulling-macOS-Standalone.zip` - Zipped for easy sharing

**iOS Simulator:**
- `dist/Bulling-iOS-Simulator.app` - For iOS Simulator testing
- `dist/Bulling-iOS-Simulator.zip` - Zipped package

**iOS Device:**
- `dist/Bulling-iOS-Device.app` - Unsigned iOS app
- `dist/Bulling-iOS-Unsigned.ipa` - For sideloading tools

---

## 🛠️ Technical Requirements

### To Use (Pre-built Apps)
- **macOS App**: macOS 10.13 or later
- **iOS App**: iOS 15.0 or later (requires sideloading for physical devices)

### To Build from Source
- **macOS**: Required for building
- **Xcode**: Version 14+ (includes command-line tools)
- **Git**: For cloning repository
- **Python 3.9+**: For Python-based builds (macOS app)

---

## 📱 iOS Personal Installation Methods

Since this is for personal use only, you have several options for iOS:

### 1. iOS Simulator (Free, macOS Required)
Best for: Testing and development on your Mac
```bash
./build_standalone.sh ios
xcrun simctl install booted dist/Bulling-iOS-Simulator.app
```

### 2. AltStore (Free, 7-Day Signing)
Best for: Personal devices without developer account
- Download AltStore: https://altstore.io/
- Install the `.ipa` file from `dist/Bulling-iOS-Unsigned.ipa`
- Re-sign every 7 days (automated if AltStore is running)

### 3. Sideloadly (Free, 7-Day Signing)
Best for: Simple drag-and-drop installation
- Download Sideloadly: https://sideloadly.io/
- Drag the `.ipa` file and install to your device
- Re-sign every 7 days

### 4. Personal Developer Certificate ($99/year)
Best for: No re-signing hassle
- Sign up for Apple Developer Program
- Sign the app with your certificate in Xcode
- Apps valid for 1 year
- Still for **personal use only** - cannot publish to App Store

---

## ⚙️ First-Time Setup (macOS)

When you first open the macOS app, you might see a security warning:

**"Bulling cannot be opened because it is from an unidentified developer"**

This is normal for unsigned apps. To fix:

1. **Right-click** (or Control-click) on `Bulling.app`
2. Select **"Open"** from the menu
3. Click **"Open"** in the dialog that appears
4. The app will open and won't ask again

**Alternative Method:**
1. Go to **System Settings** → **Privacy & Security**
2. Find the message about Bulling being blocked
3. Click **"Open Anyway"**

---

## 🎮 How to Play

1. **Launch the app**
2. **Add players** (1-8 players supported)
3. **Tap/Click pins** to knock them down (white = standing, red = knocked)
4. **Submit throw** to record your score
5. **View scorecard** to track progress
6. **Compete for 300** (perfect game)!

### Scoring Rules:
- **Strike (X)**: Knock all 10 pins on first throw
- **Spare (/)**: Knock all 10 pins in 2 throws
- **Open Frame**: Less than 10 pins total
- **10th Frame**: Bonus throws for strikes/spares

---

## 📂 What's Included

```
Bulling/
├── macOS/              # macOS native Swift app
├── iOS/                # iOS native Swift app
├── bulling_qt.py       # Python/Qt cross-platform version
├── build_standalone.sh # Standalone build script ⭐
├── LICENSE.txt         # Personal use license ⭐
├── README.md           # Main documentation
└── PERSONAL_USE_README.md  # This file ⭐
```

---

## 🆘 Troubleshooting

### macOS

**Q: "App is damaged and can't be opened"**
A: This is Gatekeeper blocking unsigned apps. Follow the "First-Time Setup" instructions above.

**Q: App won't launch after update**
A: Delete the old app completely and install the new version fresh.

**Q: Where is my game data saved?**
A: Game saves are in `~/Library/Application Support/Bulling/`

### iOS

**Q: Can I publish this to the App Store?**
A: No, this is for personal use only. Publishing would violate the license.

**Q: Why do I need to re-sign every 7 days?**
A: Free Apple IDs only allow 7-day certificates. Use a paid developer account for 1-year certificates.

**Q: App crashes on launch**
A: Make sure you're running iOS 15.0 or later. Older versions are not supported.

---

## 🤝 Sharing with Friends & Family

You can share the standalone apps with friends and family for their personal use:

### Sharing macOS App:
1. Share `Bulling-macOS-Standalone.zip` file
2. Tell them to unzip and move to Applications
3. Remind them to right-click → Open first time

### Sharing iOS App:
1. Share `Bulling-iOS-Unsigned.ipa` file
2. They'll need to sideload using AltStore or similar
3. Or they can build from source

**Important:** Remind recipients this is for personal use only!

---

## 📜 License

See [LICENSE.txt](LICENSE.txt) for the complete Personal Use License.

**Summary:**
- ✅ Free for personal use
- ❌ No commercial use
- ❌ No app store publishing
- ❌ No business use

---

## 🎯 Quick Links

- **Main README**: [README.md](README.md)
- **Build Guide**: [DISTRIBUTION_GUIDE.md](DISTRIBUTION_GUIDE.md)
- **macOS Guide**: [MACOS_APP_GUIDE.md](MACOS_APP_GUIDE.md)
- **iOS Guide**: [iOS_SETUP_GUIDE.md](iOS_SETUP_GUIDE.md)
- **Personal Use License**: [LICENSE.txt](LICENSE.txt)

---

## 🎉 Enjoy Bulling!

This app was created for personal entertainment and enjoyment. We hope you have 
fun bowling with friends and family!

**Remember:** This is free software for personal use. Please respect the license 
and don't use it commercially or publish it to app stores.

**Happy Bowling! 🎳🐂**

---

*For technical questions or issues, please check the repository's issue tracker.*
