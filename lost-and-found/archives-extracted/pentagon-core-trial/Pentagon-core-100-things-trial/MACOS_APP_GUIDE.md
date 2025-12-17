# 🎯 Dart Strike - Easy Installation Guide for macOS

**No coding required! Just follow these simple steps.**

---

## 📥 For Users: Installing the App (Super Easy!)

### Option 1: Download Pre-Built App (Easiest) ⭐

1. **Download** the `Dart Strike.zip` file from the release
2. **Double-click** the zip file to unzip it
3. **Drag** `Dart Strike.app` to your Applications folder
4. **Double-click** Dart Strike in Applications to play!

**That's it!** 🎉

### First Launch Note
When you first open the app, macOS may show a security warning because the app isn't from the App Store.

**To allow it:**
1. Go to **System Settings** → **Privacy & Security**
2. Scroll down and click **"Open Anyway"** next to Dart Strike
3. Click **"Open"** in the confirmation dialog

You only need to do this once!

---

## 🎮 How to Play

### Starting a Game
1. **Launch Dart Strike** from Applications
2. **Click "Add Player"** to add each player (up to 8 players)
3. **Click "Start Game"** when ready

### Playing
1. **Click the bowling pins** you want to knock down
   - White pins = standing
   - Red pins = knocked down
2. **Click "Submit Throw"** to record your throw
3. The game automatically moves to the next player!

### Scoring
- **Strike (X)**: Knock down all 10 pins on first throw = 10 + next 2 throws
- **Spare (/)**: Knock down all 10 pins in 2 throws = 10 + next 1 throw
- **Perfect game**: 12 strikes = 300 points!

View your scores in the scorecard on the right side!

### Controls
- **Add Player**: Add new players before starting
- **Start Game**: Begin a new game
- **New Game**: Reset and start over
- **Submit Throw**: Record your current throw

---

## 🛠️ For Developers: Building the App Yourself

If you want to build the app from source code:

### Prerequisites
- macOS 10.13 or later
- Python 3.9 or later
- Terminal access

### Build Steps

1. **Open Terminal** (Applications → Utilities → Terminal)

2. **Navigate to the project folder**:
   ```bash
   cd /path/to/Pentagon-core-100-things
   ```

3. **Run the build script**:
   ```bash
   ./build_macos_app.sh
   ```

4. **Find your app**:
   - Located in: `dist/Dart Strike.app`
   - Copy to Applications folder
   - Double-click to run!

### Manual Build (Alternative)
```bash
# Install dependencies
pip3 install -r requirements.txt

# Build the app
python3 setup.py py2app

# Your app is in: dist/Dart Strike.app
```

---

## 📦 Package Information

### Current Dependencies
- **PySide6 (≥6.5.0)**: Modern Qt6 framework for Python
  - Provides professional cross-platform GUI
  - Active development and updates
  - Excellent performance and stability

### Build Tools (for developers)
- **py2app (≥0.28.0)**: Creates standalone macOS applications
  - Bundles Python and all dependencies
  - Creates double-clickable .app bundles
  - No Python installation needed for users

### Why These Packages?

**PySide6 Benefits:**
- ✅ Official Qt bindings for Python
- ✅ Cross-platform (could build for Windows/Linux too)
- ✅ Modern, native-looking interface
- ✅ Professional-grade GUI framework
- ✅ Active development and support

**py2app Benefits:**
- ✅ Creates true macOS applications
- ✅ No code/terminal required for users
- ✅ Bundles all dependencies
- ✅ Professional distribution

### Alternatives Considered

1. **Tkinter** (Built-in Python GUI)
   - ❌ Less modern appearance
   - ❌ Limited styling options
   - ✅ No dependencies

2. **PyQt6** (Qt6 alternative)
   - ❌ Different licensing (GPL/Commercial)
   - ✅ Similar features to PySide6
   - ⚠️ PySide6 preferred (official Qt bindings)

3. **Kivy** (Touch-focused GUI)
   - ❌ Overkill for desktop app
   - ❌ Non-native look and feel
   - ✅ Good for mobile

**Recommendation**: Current setup (PySide6 + py2app) is optimal for this use case.

---

## 🎨 Customization

### Changing the App Icon
1. Create or find a 1024×1024 PNG icon
2. Convert to .icns format:
   ```bash
   # Create iconset folder
   mkdir DartStrike.iconset
   
   # Add your PNG images in various sizes
   # (512x512, 256x256, 128x128, 64x64, 32x32, 16x16)
   # Named: icon_512x512.png, icon_256x256.png, etc.
   
   # Convert to .icns
   iconutil -c icns DartStrike.iconset -o app_icon.icns
   ```
3. Replace `app_icon.icns` in the project
4. Rebuild the app

### Modifying Colors/Styles
Edit `dart_strike_qt.py` and change the color values in the `setStyleSheet()` calls.

---

## 🔧 Troubleshooting

### "App can't be opened because it's from an unidentified developer"
**Solution**: Go to System Settings → Privacy & Security → Click "Open Anyway"

### App doesn't launch
**Solution**: 
1. Right-click the app → Show Package Contents
2. Open Terminal
3. Run: `./Contents/MacOS/Dart\ Strike`
4. Check error messages

### Build fails
**Solutions**:
- Update Python: `brew upgrade python3`
- Update pip: `pip3 install --upgrade pip`
- Reinstall dependencies: `pip3 install --upgrade -r requirements.txt`

### Missing icon
**Solution**: The app works fine without a custom icon. To add one, see "Changing the App Icon" above.

---

## 📤 Distributing the App

### For macOS Users

1. **Zip the app**:
   ```bash
   cd dist
   zip -r "Dart Strike.zip" "Dart Strike.app"
   ```

2. **Share** the zip file via:
   - Email
   - Cloud storage (Dropbox, Google Drive, etc.)
   - USB drive
   - Website download

3. **Recipients** simply:
   - Unzip the file
   - Copy to Applications
   - Double-click to run!

### Optional: Code Signing (Advanced)
For commercial distribution, you may want to sign the app with an Apple Developer certificate:

```bash
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" "Dart Strike.app"
```

This removes the security warning for users.

---

## 🆘 Support

### Common Questions

**Q: Do users need Python installed?**  
A: No! The .app bundle includes everything.

**Q: Will this work on Windows/Linux?**  
A: No, this builds a macOS-only app. For Windows, use PyInstaller. For Linux, use PyInstaller or native packaging.

**Q: Can I sell this app?**  
A: Check the repository license. PySide6 is LGPL licensed.

**Q: How big is the app?**  
A: Approximately 100-150 MB (includes Python runtime and Qt framework).

**Q: Does it need internet?**  
A: No! Fully offline, no internet required.

---

## 🎯 Quick Reference Card

| Task | Command |
|------|---------|
| Build app | `./build_macos_app.sh` |
| Clean builds | `rm -rf build dist` |
| Test app | Double-click in dist folder |
| Install dependencies | `pip3 install -r requirements.txt` |
| Manual build | `python3 setup.py py2app` |

---

## ✨ Features

- ✅ No coding required for users
- ✅ Double-click to run
- ✅ Professional bowling scoring
- ✅ Up to 8 players
- ✅ Traditional bowling rules (strikes, spares, 10th frame)
- ✅ Auto-save game progress
- ✅ Interactive pin selection
- ✅ Real-time scorecard
- ✅ Modern, clean interface

---

## 📝 Version History

**v1.0.0** (Current)
- Initial macOS app release
- Full bowling game implementation
- py2app build system
- User-friendly installation

---

**Ready to play? Double-click and enjoy! 🎳**

For more information, see the main README.md file.
