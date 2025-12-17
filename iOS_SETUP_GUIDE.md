# 🐂 Bulling - iOS App Setup Guide

Complete guide for creating and building the Bulling iOS app in Xcode.

---

## ⚠️ **PERSONAL USE ONLY**

**This app is for personal, non-commercial use only.**
- ✅ Install on your personal devices
- ✅ Share with friends and family for personal use
- ❌ **Cannot** publish to App Store
- ❌ **Cannot** use for commercial purposes
- ❌ **Cannot** distribute via TestFlight for public/commercial use

See [LICENSE.txt](LICENSE.txt) for complete terms.

---

## 📋 Prerequisites

- **macOS** 11.0 (Big Sur) or later
- **Xcode** 14.0 or later
- **iOS device** or Simulator running iOS 15.0+
- **Personal Apple ID** (free) or Apple Developer account (optional)

---

## 🚀 Quick Setup (5 Minutes)

### Step 1: Create New Xcode Project

1. Open **Xcode**
2. Select **File** → **New** → **Project**
3. Choose **iOS** → **App**
4. Click **Next**

### Step 2: Configure Project

Fill in the following details:
- **Product Name**: `Bulling`
- **Team**: Select your team (or None for simulator only)
- **Organization Identifier**: `com.yourname` (or your identifier)
- **Interface**: **SwiftUI**
- **Language**: **Swift**
- **Use Core Data**: ❌ Unchecked
- **Include Tests**: ❌ Unchecked (optional)

Click **Next** and save the project.

---

## 📁 Step 3: Add Source Files

Copy all Swift files from the `iOS/Bulling/` folder to your Xcode project:

### Required Files:
1. **BullingApp.swift** - Main app entry point with splash screen
2. **GameModel.swift** - Core game logic and data models
3. **SplashScreen.swift** - Loading screen with bull head logo
4. **ContentView.swift** - Main menu and setup view
5. **GameView.swift** - Interactive bowling game interface
6. **ScorecardView.swift** - Score tracking and display

### How to Add Files:

1. In Xcode, **right-click** on the **Bulling** folder in the navigator
2. Select **Add Files to "Bulling"...**
3. Select all the Swift files from `iOS/Bulling/`
4. Make sure **"Copy items if needed"** is ✅ checked
5. Make sure **"Add to targets"** has **Bulling** ✅ checked
6. Click **Add**

---

## 🎨 App Icon (Optional)

To add a custom bull head icon:

1. Open **Assets.xcassets** in Xcode
2. Click on **AppIcon**
3. Drag and drop icon images for each size
   - 1024×1024 for App Store
   - 180×180 for iPhone
   - 120×120 for smaller devices
   - etc.

**Note**: The app works perfectly without a custom icon! The bull head logo shows in the app itself.

---

## ▶️ Step 4: Build and Run

### For Simulator:

1. Select an iOS Simulator from the device menu (e.g., "iPhone 14 Pro")
2. Press **⌘R** (Command + R) or click the **Play** button
3. Wait for build to complete
4. App launches with the bull head splash screen!

### For Physical Device:

1. Connect your iPhone/iPad via USB
2. Select your device from the device menu
3. You may need to:
   - Trust the computer on your device
   - Select a development team in **Signing & Capabilities**
4. Press **⌘R** to build and run

---

## 🎮 Testing the App

### Test Checklist:

- [ ] **Splash screen** appears with bull head logo (dartboard eyes, bowling pin horns)
- [ ] **Loading spinner** shows for 2 seconds
- [ ] **Main menu** displays with bull head logo
- [ ] **Add Player** button works
- [ ] **Start Game** button appears after adding players
- [ ] **Pin tapping** toggles pins (white ↔ red)
- [ ] **Submit Throw** records the throw
- [ ] **Pins reset** correctly between frames
- [ ] **Scorecard** displays properly
- [ ] **Scores** calculate correctly (strikes, spares, etc.)
- [ ] **Game over** shows winner

---

## 🔧 Troubleshooting

### Build Errors

**Error: "Cannot find 'BullHeadLogo' in scope"**
- **Solution**: Make sure all files are added to the target
- Check that SplashScreen.swift is included in the project

**Error: "No such module 'SwiftUI'"**
- **Solution**: Make sure deployment target is iOS 15.0+
- Check in **Project Settings** → **General** → **Deployment Info**

**Error: "Command CodeSign failed"**
- **Solution**: Go to **Signing & Capabilities** tab
- Select a development team or use "Sign to Run Locally"

### Runtime Issues

**App crashes on launch**
- Check the Console for error messages
- Make sure all Swift files are properly added
- Try **Clean Build Folder** (⇧⌘K) then rebuild

**Splash screen doesn't show**
- Check that `showSplash = true` in BullingApp.swift
- Verify SplashScreen.swift is in the project

**Pins don't tap**
- Make sure GameView.swift is included
- Check that pins are properly initialized in GameModel

---

## 📱 App Features

### Bull Head Logo Design
- **Head**: Brown circular gradient
- **Eyes**: Dartboard pattern (concentric rings)
- **Horns**: Bowling pin shapes
- **Animated**: Subtle pulsing eyes and entrance animation

### Gameplay Features
- ✅ Traditional 10-pin bowling rules
- ✅ Up to 8 players
- ✅ Strike and spare detection
- ✅ 10th frame bonus throws
- ✅ Real-time scoring
- ✅ Beautiful, intuitive interface

---

## 📦 Personal Distribution Methods

⚠️ **Remember:** This app is for personal use only. Do not publish to App Store or TestFlight for public distribution.

### iOS Simulator (For Testing)

1. Build and run in Xcode: **⌘R**
2. Select an iOS Simulator as the destination
3. Perfect for testing before installing on devices

### Personal Device Installation (Sideloading)

**Method 1: Direct Install via Xcode (7-day signing with free Apple ID)**
1. Connect your iPhone/iPad to your Mac
2. Select your device in Xcode
3. Trust computer on device
4. Build & Run: **⌘R**
5. Trust developer certificate on device (Settings → General → VPN & Device Management)

**Note:** Apps signed with free Apple ID expire after 7 days and must be re-installed.

**Method 2: AltStore (Recommended for Personal Use)**
1. Archive your app: **Product** → **Archive**
2. Export as **Development** or **Ad Hoc**
3. Install [AltStore](https://altstore.io/) on Mac and iPhone
4. Use AltStore to sideload the .ipa file
5. AltStore auto-refreshes apps before expiration (when running)

**Method 3: Sideloadly**
1. Archive and export .ipa
2. Download [Sideloadly](https://sideloadly.io/)
3. Connect device and drag .ipa into Sideloadly
4. Sign in with Apple ID and install

**Method 4: Personal Developer Account ($99/year)**
1. Sign up for [Apple Developer Program](https://developer.apple.com/programs/)
2. Sign app with your developer certificate
3. Apps valid for 1 year (no weekly re-signing)
4. Still for **personal use only** - cannot publish to public App Store due to license

### ⛔ What You CANNOT Do

- ❌ **Publish to App Store** - Violates personal use license
- ❌ **Public TestFlight distribution** - Personal use only
- ❌ **Commercial distribution** - Non-commercial license
- ❌ **Enterprise distribution for business** - Personal use restriction

### ✅ What You CAN Do

- ✅ Install on your personal devices
- ✅ Share .ipa with friends/family for their personal devices
- ✅ Use sideloading tools (AltStore, Sideloadly)
- ✅ Sign with your personal developer certificate
- ✅ Test in iOS Simulator

---

## 🎯 Quick Reference

| Action | Shortcut |
|--------|----------|
| Build & Run | ⌘R |
| Stop | ⌘. |
| Clean Build | ⇧⌘K |
| Build | ⌘B |
| Show Console | ⇧⌘C |

---

## 💡 Tips

1. **Use Live Preview**: SwiftUI views have preview support
   - Click **Resume** in the canvas to see live preview
   
2. **Simulator Testing**: 
   - Test on different screen sizes (iPhone SE, iPhone 14 Pro Max, iPad)
   
3. **Debug Logging**:
   - Add `print()` statements to track game state
   
4. **Performance**:
   - The app is highly optimized and runs at 60 FPS

---

## 🆘 Need Help?

### Common Questions

**Q: Can I change the colors?**
A: Yes! Edit the Color values in the Swift files.

**Q: Can I modify the bull head logo?**
A: Yes! Edit BullHeadLogo in SplashScreen.swift.

**Q: Does it work on iPad?**
A: Yes! The app is universal (iPhone + iPad).

**Q: Can I remove the splash screen?**
A: Yes! Set `showSplash = false` in BullingApp.swift or remove the conditional logic.

---

## 🎉 You're Ready!

Your Bulling iOS app is now set up and ready to play!

**Enjoy bowling! 🎳🐂**

---

For the macOS version, see **MACOS_APP_GUIDE.md**
