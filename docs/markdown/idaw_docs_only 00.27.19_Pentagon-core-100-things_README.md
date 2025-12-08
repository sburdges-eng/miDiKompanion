# Bulling - Bowling Scoring Game

[![Build macOS App](../../actions/workflows/build-macos-app.yml/badge.svg)](../../actions/workflows/build-macos-app.yml)

**Strike & Score!** Traditional 10-pin bowling game with a unique bull-themed design.

<img src="bulling_icon.svg" width="200" alt="Bulling Logo - Bull head with dartboard eyes and bowling pin horns"/>

---

## ⚠️ **PERSONAL USE ONLY**

**This software is provided for personal, non-commercial use only.**
- ✅ Free for personal entertainment
- ❌ No commercial use or distribution
- ❌ No app store publishing
- ❌ No business/organizational use

📖 **See [PERSONAL_USE_README.md](PERSONAL_USE_README.md) for complete licensing terms and download instructions**

---

## 🎮 Features

- **🐂 Unique Bull Head Design**: Dartboard eyes & bowling pin horns
- **🎳 Traditional Bowling Rules**: Strikes, spares, and proper scoring
- **👥 Multi-Player**: Up to 8 players per game
- **📊 Real-Time Scorecard**: Track scores frame-by-frame
- **🎯 10th Frame Bonus**: Authentic bowling rules
- **💾 Auto-Save**: Resume your games anytime
- **📱 Cross-Platform**: Python version supports macOS, Linux & Windows (requires Qt6)
- **⚡️ Easy Install**: One command installation with `pip install .`
- **🖥️ Multiple Formats**: Run as CLI command, Python script, or macOS app bundle

---

## 📦 Available Versions

This repository contains **TWO complete versions** of Bulling:

### 1. 🖥️ macOS App (Native SwiftUI or Python/Qt6)
**Perfect for desktop use**

- Beautiful native macOS application
- **Native SwiftUI version** - Pure Swift, no dependencies
- **Python/Qt6 version** - Cross-platform alternative
- Double-click to run (no code required!)
- Standalone .app bundle
- No installation needed for users

📖 **[macOS Setup Guide](MACOS_APP_GUIDE.md)**

### 2. 📱 iOS App (Swift/SwiftUI)
**Perfect for mobile devices (Personal Use)**

- Native iOS & iPadOS app
- Animated splash screen
- Touch-optimized interface
- For personal sideloading only (not App Store)

📖 **[iOS Setup Guide](iOS_SETUP_GUIDE.md)**

---

## 🚀 Quick Start

📥 **[QUICK DOWNLOAD GUIDE](QUICK_DOWNLOAD.md)** - Fast track to download and install!

### For Users (Standalone Apps - Personal Use)

#### Build Standalone Apps for Personal Distribution

```bash
# Build unsigned, standalone apps for personal use
./build_standalone.sh all         # Build everything
./build_standalone.sh macos       # macOS only
./build_standalone.sh ios         # iOS simulator only
./build_standalone.sh ios-device  # iOS device (unsigned)
```

**Creates:**
- `dist/Bulling-macOS.app` - macOS standalone app
- `dist/Bulling-iOS-Simulator.app` - iOS simulator app
- `dist/Bulling-iOS-Unsigned.ipa` - iOS device app (for sideloading)

**Perfect for:**
- Personal use and testing
- Sharing with friends and family
- No signing or developer account required

### For Users (macOS App Bundle)

#### Download from GitHub Releases (Recommended)

1. Go to the [Releases page](../../releases)
2. Download `Bulling-macOS.dmg` or `Bulling-macOS.zip`
3. **DMG**: Open the disk image and drag Bulling to Applications
4. **ZIP**: Unzip and drag `Bulling.app` to Applications
5. **Double-click** to play!

#### Download from GitHub Actions (Latest Build)

1. Go to the [Actions tab](../../actions)
2. Click on the latest successful "Build macOS App" workflow run
3. Download one of the artifacts:
   - `Bulling-macOS-DMG` - Disk image (easiest installation)
   - `Bulling-macOS-ZIP` - Zipped app bundle
   - `Bulling-macOS-App` - Raw .app bundle
4. Extract and run!

#### First Launch Security Note (macOS)

Since the app is not signed with an Apple Developer certificate:
1. Right-click (or Control-click) on `Bulling.app`
2. Select "Open" from the context menu
3. Click "Open" in the security dialog
4. The app will now open normally in the future

### For Users (Install from Source)

```bash
# Clone or download the repository
git clone <repository-url>
cd Pentagon-core-100-things

# Install as executable command
pip3 install .

# Run the app
bulling
```

### For Developers (macOS Native SwiftUI)

```bash
# Clone the repository
git clone <repository-url>
cd Pentagon-core-100-things

# Option 1: Open the ready-to-build Xcode project (recommended)
open macOS/BullingMac.xcodeproj
# Select 'My Mac' and press Cmd+R to build and run!

# Option 2: Build from command line
./build_macos_native.sh debug     # Build Debug configuration
./build_macos_native.sh release   # Build Release configuration
./build_macos_native.sh archive   # Create archive for distribution
```

### For Developers (Python/Qt6 - Cross-Platform)

```bash
# Clone the repository
git clone <repository-url>
cd Pentagon-core-100-things

# Option 1: Install as executable (recommended)
pip3 install .
bulling  # Run the app

# Option 2: Install in development mode
pip3 install -e .
bulling  # Run the app

# Option 3: Run directly (requires dependencies)
pip3 install -r requirements.txt
python3 bulling_qt.py

# Option 4: Build standalone macOS app (Python)
./build_macos_app.sh
```

### For Developers (iOS)

```bash
# Option 1: Open the ready-to-build Xcode project
open iOS/BullingApp.xcodeproj
# Select a simulator and press Cmd+R to build and run!

# Option 2: Build from command line (macOS only)
./build_ios_app.sh simulator   # Build for iOS simulator
./build_ios_app.sh device      # Build for physical device
./build_ios_app.sh archive     # Create archive for distribution

# Option 3: Manual setup (alternative)
# Follow iOS_SETUP_GUIDE.md for complete instructions
```

---

## 🎨 App Icon & Branding

### Bull Head Logo Design

The Bulling logo features a creative bull head with:
- **🎯 Dartboard Eyes**: Concentric rings (black, white, green, red, bullseye)
- **🎳 Bowling Pin Horns**: White pins with red stripes
- **🟤 Brown Head**: Gradient brown circular head
- **✨ Animated**: Pulsing eyes and smooth entrance animation

### Generating the Icon

```bash
# Generate SVG icon
python3 generate_icon.py

# View the icon
open bulling_icon.svg

# Convert to PNG (macOS)
# Use Preview, Image2icon, or online tools

# Create .icns for macOS app
./create_icon.sh bulling_icon_1024.png
```

---

## 📖 How to Play

### Setup
1. **Launch Bulling** on macOS or iOS
2. **Add Players** (up to 8)
3. **Start Game**

### Gameplay
1. **Tap/Click** bowling pins to knock them down
   - White pins = Standing
   - Red pins = Knocked down
2. **Submit Throw** to record your throw
3. Game **automatically advances** to next player
4. **View scorecard** anytime

### Scoring
- **Strike (X)**: All 10 pins on first throw = 10 + next 2 throws
- **Spare (/)**: All 10 pins in 2 throws = 10 + next 1 throw  
- **Open Frame**: Count actual pins knocked
- **10th Frame**: Bonus throws for strikes/spares
- **Perfect Game**: 12 strikes = 300 points! 🏆

---

## 🛠️ Technology Stack

### macOS Version
- **Python 3.9+**
- **PySide6 (Qt6)**: Professional GUI framework
- **py2app**: macOS app bundling

### iOS Version
- **Swift 5.9+**
- **SwiftUI**: Modern declarative UI
- **iOS 15.0+**: Target deployment

---

## 📁 Project Structure

```
Pentagon-core-100-things/
├── bulling_qt.py              # macOS Python app (main)
├── setup.py                   # py2app build configuration
├── build_macos_app.sh         # Build script for macOS (Python)
├── build_macos_native.sh      # Build script for macOS (Native SwiftUI)
├── build_ios_app.sh           # Build script for iOS
├── create_icon.sh             # Icon creation helper
├── generate_icon.py           # Bull head icon generator
├── bulling_icon.svg           # App icon (SVG)
├── requirements.txt           # Python dependencies
├── .github/
│   └── workflows/
│       └── build-macos-app.yml # CI/CD for automated builds
├── iOS/
│   ├── BullingApp.xcodeproj/  # Xcode project (ready to build)
│   │   └── project.pbxproj
│   ├── BullingApp/            # iOS app source code
│   │   ├── BullingApp.swift
│   │   ├── GameModel.swift
│   │   ├── SplashScreen.swift
│   │   ├── ContentView.swift
│   │   ├── GameView.swift
│   │   ├── ScorecardView.swift
│   │   ├── Info.plist
│   │   └── Assets.xcassets/
│   └── Bulling/               # Legacy iOS Swift files
├── macOS/
│   ├── BullingMac.xcodeproj/  # macOS Xcode project (ready to build)
│   │   └── project.pbxproj
│   ├── BullingMac/            # macOS app source code
│   │   ├── BullingApp.swift
│   │   ├── GameModel.swift
│   │   ├── SplashScreen.swift
│   │   ├── ContentView.swift
│   │   ├── GameView.swift
│   │   ├── ScorecardView.swift
│   │   ├── Info.plist
│   │   ├── Bulling.entitlements
│   │   └── Assets.xcassets/
│   └── Bulling/               # Legacy macOS Swift files
├── MACOS_APP_GUIDE.md         # macOS detailed guide
├── iOS_SETUP_GUIDE.md         # iOS detailed guide
└── README.md                  # This file
```

---

## 🎯 Package Dependencies

### Python (macOS)
```
PySide6>=6.5.0          # Qt6 GUI framework (LGPL)
py2app>=0.28.0          # macOS app builder
```

**Why these packages?**
- ✅ **PySide6**: Official Qt bindings, professional UI, cross-platform
- ✅ **py2app**: Creates true macOS .app bundles, no user dependencies
- ✅ **Minimal**: Only 2 dependencies, small footprint
- ✅ **Stable**: Mature, well-maintained packages

### Swift (iOS)
```
SwiftUI (built-in)      # Native iOS UI framework
Foundation (built-in)   # Core functionality
Combine (built-in)      # Reactive programming
```

---

## 🔧 Building & Distribution

### Automated Builds (GitHub Actions)

Every push to the `main` branch or `claude/*` branches automatically triggers a build:

1. **macOS App** is built using py2app on GitHub's macOS runners
2. **Artifacts** are uploaded and available for download for 30 days
3. **Releases** are created automatically when you push a version tag (e.g., `v1.0.0`)

To create a release:
```bash
git tag v1.0.0
git push origin v1.0.0
```

This will:
- Build the macOS app
- Create a GitHub Release with `Bulling-macOS.dmg` and `Bulling-macOS.zip`
- Anyone can download from the Releases page

### Quick Distribution (Automated)

```bash
# Create both macOS and iOS distribution zips (macOS only)
./create_distribution_zips.sh

# Creates:
# - dist/Bulling-macOS.zip (macOS app bundle)
# - dist/Bulling-iOS.zip (iOS source files)
```

📖 **[Distribution Scripts Guide](DISTRIBUTION_SCRIPTS_README.md)**

### Manual Build

#### macOS App (Native SwiftUI - Recommended)

```bash
# Option 1: Open in Xcode (easiest)
open macOS/BullingMac.xcodeproj
# Then: Product → Run (Cmd+R) or Product → Archive

# Option 2: Build from command line
./build_macos_native.sh debug     # Build Debug configuration
./build_macos_native.sh release   # Build Release (creates dist/Bulling.app)
./build_macos_native.sh archive   # Create archive for distribution

# Result: dist/Bulling.app or build/macos/Bulling.xcarchive
```

#### macOS App (Python/Qt6 - Cross-Platform)

```bash
# Build the app
./build_macos_app.sh

# Result: dist/Bulling.app

# Create distribution zip
./create_macos_zip.sh
# Result: dist/Bulling-macOS.zip
```

#### iOS App

```bash
# Option 1: Open project in Xcode (recommended)
open iOS/BullingApp.xcodeproj
# Then: Product → Run (Cmd+R) or Product → Archive

# Option 2: Build from command line
./build_ios_app.sh simulator   # Build for simulator
./build_ios_app.sh device      # Build for device (unsigned)
./build_ios_app.sh archive     # Create archive

# Option 3: Create source files package
./create_ios_zip.sh
# Result: dist/Bulling-iOS.zip

# Personal Distribution (No App Store):
# - Share .ipa files with friends/family for sideloading
# - Use AltStore, Sideloadly, or personal developer certificate
# - ⚠️ NOT for App Store or TestFlight (personal use license)
```

---

## 🎨 Customization

### Change Colors

**macOS (Python)**: Edit color values in `bulling_qt.py`
```python
# Example: Change pin color
.setStyleSheet("background-color: #YOUR_COLOR;")
```

**iOS (Swift)**: Edit color values in Swift files
```swift
// Example: Change background
Color(red: 0.95, green: 0.97, blue: 1.0)
```

### Modify Bull Head Logo

Edit `SplashScreen.swift` (iOS) or `generate_icon.py` (icon) to customize:
- Eye colors
- Horn shapes
- Head color
- Animation effects

### Adjust Splash Screen Duration

In `BullingApp.swift`:
```swift
DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {  // Change 2.0 to your preferred seconds
    // ...
}
```

---

## 🐛 Troubleshooting

### macOS

**"App can't be opened" security warning**
- Go to System Settings → Privacy & Security
- Click "Open Anyway" next to Bulling

**Build fails**
```bash
# Update dependencies
pip3 install --upgrade -r requirements.txt

# Clean and rebuild
rm -rf build dist
./build_macos_app.sh
```

### iOS

**Build errors in Xcode**
- Ensure all .swift files are added to target
- Check deployment target is iOS 15.0+
- Clean build folder (⇧⌘K)

**Splash screen doesn't show**
- Verify SplashScreen.swift is in project
- Check `showSplash = true` in BullingApp.swift

---

## 📊 Version Comparison

| Feature | macOS | iOS |
|---------|-------|-----|
| **Platform** | macOS 10.13+ | iOS 15.0+ |
| **UI Framework** | Qt6/PySide6 | SwiftUI |
| **Distribution** | .app or .zip | Sideloading (.ipa) |
| **Setup Time** | 5 min | 10 min |
| **User Install** | Drag & drop | AltStore/Sideloadly |
| **Dev Environment** | Any IDE + Python | Xcode required |
| **Bull Logo** | In app | Splash + in app |
| **File Size** | ~100-150 MB | ~5-10 MB |
| **Use Case** | Personal desktop | Personal mobile |

---

## 🏆 Game Rules Reference

### Scoring Examples

**Strike (X)**: Pin 1st ball, then bowl 7 and 2
- Frame score: 10 + 7 + 2 = 19

**Spare (/)**: Bowl 7 then 3 (spare), then bowl 5
- Frame score: 10 + 5 = 15

**Open Frame**: Bowl 6 then 2
- Frame score: 6 + 2 = 8

**10th Frame**:
- Strike: Get 2 bonus balls (can score up to 30)
- Spare: Get 1 bonus ball
- Open: No bonus balls

**Perfect Game**: X X X X X X X X X X X X = 300

---

## 📄 License

**PERSONAL USE ONLY** - See [LICENSE.txt](LICENSE.txt) for complete terms.

**Summary:**
- ✅ Free for personal, non-commercial use
- ✅ Share with friends and family
- ❌ No commercial use or monetization
- ❌ No app store publishing (Apple App Store, Google Play, etc.)
- ❌ No business or organizational use

**Third-Party Components:**
- **PySide6** (Qt6): LGPL licensed - users must comply with LGPL terms
- **SwiftUI**: Part of Apple's SDK - subject to Apple's terms

📖 **Full personal use guide: [PERSONAL_USE_README.md](PERSONAL_USE_README.md)**

---

## 🎉 Features Highlights

- ✅ **No coding required** for users
- ✅ **Professional scoring system**
- ✅ **Beautiful, modern UI**
- ✅ **Unique bull head branding**
- ✅ **Cross-platform** (macOS & iOS)
- ✅ **Offline** - no internet needed
- ✅ **Auto-save** game progress
- ✅ **Responsive** - smooth 60 FPS
- ✅ **Intuitive** - easy to learn

---

## 🔗 Quick Links

- **[Quick Distribution Guide](QUICK_DISTRIBUTION_GUIDE.md)** - Fast track to creating distribution zips
- **[Distribution Scripts](DISTRIBUTION_SCRIPTS_README.md)** - Automated zip creation documentation
- **[Distribution Guide](DISTRIBUTION_GUIDE.md)** - Complete distribution and build guide
- **[macOS Guide](MACOS_APP_GUIDE.md)** - Detailed macOS setup and usage
- **[iOS Guide](iOS_SETUP_GUIDE.md)** - Complete iOS development guide
- **[Icon SVG](bulling_icon.svg)** - Bull head logo design

---

## 🎯 Perfect For

- 🏠 Home entertainment
- 🎉 Parties and gatherings
- 🍺 Bars and restaurants
- 🎳 Bowling alleys (virtual scoring)
- 📱 Personal mobile gaming
- 🖥️ Desktop casual gaming

---

## ✨ Coming Soon (Optional Future Features)

- [ ] Sound effects
- [ ] Game statistics and history
- [ ] Player profiles
- [ ] Dark mode
- [ ] Tournament mode
- [ ] Network multiplayer
- [ ] Additional themes

---

## 💪 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 📞 Support

For issues or questions:
1. Check the appropriate guide (macOS or iOS)
2. Review troubleshooting sections
3. Open an issue on GitHub

---

**Ready to bowl? 🎳🐂**

Download Bulling now and start striking!

---

*"Strike & Score with Bulling!"* 🎯
