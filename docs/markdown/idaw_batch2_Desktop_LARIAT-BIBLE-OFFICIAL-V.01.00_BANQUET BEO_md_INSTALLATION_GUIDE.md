# Lariat Manager - Installation Guide
## Install Your Native macOS Application

> **Version 1.0** | Professional macOS Application

---

## 🎉 What You Have

A complete, native macOS application bundle:

**Lariat Manager.app**
- ✅ Custom icon (your logo)
- ✅ macOS-native appearance
- ✅ Appears in Applications folder
- ✅ Shows in Spotlight search
- ✅ Launchable from Dock
- ✅ Proper macOS integration

---

## 📦 Quick Install (Recommended)

### **Drag & Drop Installation**

1. **Open Finder**
2. **Navigate to:** `/Users/seanburdges/Desktop/BANQUET BEO/`
3. **Find:** `Lariat Manager.app` (should have your custom icon)
4. **Drag** `Lariat Manager.app` to your **Applications** folder
5. **Done!** ✅

**That's it!** The app is now installed.

---

## 🚀 Launch Methods

### **Method 1: Spotlight** (⌘+Space)

1. Press **⌘ + Space** (Command + Spacebar)
2. Type: **"Lariat"**
3. Press **Enter**

### **Method 2: Applications Folder**

1. Open **Finder**
2. Go to **Applications**
3. Find **Lariat Manager**
4. Double-click to launch

### **Method 3: Dock** (After first launch)

1. Launch the app once (using Method 1 or 2)
2. **Right-click** the Lariat icon in Dock
3. Choose: **Options → Keep in Dock**
4. Now you can launch it anytime from the Dock!

### **Method 4: Launchpad**

1. Open **Launchpad** (F4 or pinch gesture)
2. Find **Lariat Manager**
3. Click to launch

---

## 🎨 Application Features

### **Custom Icon**

Your application displays **your custom logo** from the photo in BANQUET BEO folder:
- Appears in Applications folder
- Shows in Dock
- Displays in window title bar
- Visible in Spotlight results

### **Professional Metadata**

```
Bundle Name:      Lariat Manager
Display Name:     Lariat Manager
Identifier:       com.lariat.banquetmanager
Version:          1.0
Copyright:        © 2024 The Lariat Restaurant
```

### **macOS Integration**

- ✅ Appears as native macOS app
- ✅ Spotlight searchable
- ✅ Proper window management
- ✅ macOS color scheme
- ✅ Retina display support
- ✅ Mission Control compatible

---

## 📁 Application Structure

```
Lariat Manager.app/
├── Contents/
│   ├── Info.plist              # App metadata
│   ├── MacOS/
│   │   └── lariat_launcher     # Launch script
│   └── Resources/
│       └── lariat_icon.icns    # Your custom icon (all sizes)
```

### **Icon Sizes Included**

Your logo converted to all required macOS icon sizes:
- 16x16, 32x32 (Finder list view)
- 128x128 (Finder icon view)
- 256x256 (Retina displays)
- 512x512 (Retina large icons)
- 1024x1024 (High-resolution displays)

---

## ⚙️ System Requirements

### **Minimum Requirements**

- **macOS:** 10.13 (High Sierra) or later
- **Python:** 3.8+ (already installed)
- **Disk Space:** ~100 MB (app + data)
- **Display:** Works on any resolution (Retina optimized)

### **Recommended**

- **macOS:** 12.0 (Monterey) or later
- **Python:** 3.13 (current)
- **RAM:** 4GB+ available
- **Display:** Retina display for best appearance

---

## 🔐 Security & Permissions

### **First Launch Security**

When you first open the app, macOS may show a security warning:

**"Lariat Manager.app" cannot be opened because it is from an unidentified developer.**

**Solution:**

1. **Right-click** (or Control+click) on `Lariat Manager.app`
2. Choose **"Open"**
3. Click **"Open"** in the dialog
4. macOS will remember this choice
5. Future launches will work normally

**Alternative:**

1. Go to **System Preferences → Security & Privacy**
2. Click **"Open Anyway"** next to the Lariat Manager message
3. Click **"Open"** in the confirmation dialog

### **Why This Happens**

The app is not code-signed with an Apple Developer certificate (requires paid Apple Developer account). This is normal for personal/internal apps.

**This is safe** - you built this app yourself with the code you can review.

---

## 📍 Installation Locations

### **Recommended:** Applications Folder

```
/Applications/Lariat Manager.app
```

**Benefits:**
- ✅ Shows in Spotlight
- ✅ Shows in Launchpad
- ✅ Standard macOS location
- ✅ Easy to find

### **Alternative:** User Applications

```
~/Applications/Lariat Manager.app
```

**Benefits:**
- ✅ Doesn't require admin access
- ✅ User-specific installation
- ✅ Still searchable in Spotlight

### **Current Location:**

```
/Users/seanburdges/Desktop/BANQUET BEO/Lariat Manager.app
```

**Use Case:**
- ⚠️ Works but not ideal
- ⚠️ Desktop gets cluttered
- ✅ Easy access during development

---

## 🔄 Updating the Application

### **To Update:**

1. **Quit** the running app
2. **Delete** old version from Applications
3. **Copy** new version to Applications
4. **Launch** the updated app

### **Check Version:**

1. Launch **Lariat Manager**
2. Click **"⚙️ Settings"** in sidebar
3. Check version under **"About"** section

Current version: **1.0.0**

---

## 🗑️ Uninstalling

### **To Remove:**

1. **Quit** Lariat Manager if running
2. Open **Applications** folder
3. **Drag** `Lariat Manager.app` to **Trash**
4. **Empty Trash**

### **Remove All Data (Optional):**

The application data remains in:
```
/Users/seanburdges/Desktop/BANQUET BEO/
```

To remove all data:
1. Delete the entire `BANQUET BEO` folder
2. This removes templates, database, invoices, reports

**⚠️ Warning:** This deletes all your banquet data!

---

## 🎯 Usage After Installation

### **Normal Workflow:**

```
1. Open Spotlight (⌘+Space)
   ↓
2. Type "Lariat"
   ↓
3. Press Enter
   ↓
4. Application opens with dashboard
   ↓
5. Use 5-view navigation:
   • Dashboard - Metrics
   • Import - Process invoices
   • Reports - Generate monthly reports
   • Database - View data in Excel
   • Settings - Configuration
```

### **Quick Access:**

1. **First time:** Use Spotlight or Applications folder
2. **Keep in Dock:** Right-click → Options → Keep in Dock
3. **Future:** Just click Dock icon

---

## 🔧 Troubleshooting

### **App Won't Launch**

**Issue:** Double-clicking does nothing

**Solutions:**
1. Try right-click → Open (bypasses security check)
2. Check System Preferences → Security & Privacy
3. Run from Terminal: `open "/Applications/Lariat Manager.app"`
4. Verify Python 3 is installed: `python3 --version`

### **Icon Not Showing**

**Issue:** Generic icon instead of custom logo

**Solutions:**
1. Delete app from Applications
2. Empty Trash
3. Reinstall app
4. Restart Finder: Option+Right-click Finder icon → Relaunch

### **Security Warning Every Time**

**Issue:** Security dialog appears on every launch

**Solution:**
1. Remove app from Applications
2. Right-click → Open (do this ONCE)
3. Move to Applications folder
4. macOS remembers the exception

### **Window Not Appearing**

**Issue:** App launches but no window visible

**Solutions:**
1. Press ⌘+Tab to find Lariat Manager
2. Check if window is on another desktop/space
3. Quit and relaunch
4. Check log: `/tmp/lariat_gui.log`

---

## 📊 Application Contents

### **What's Included in the App:**

**Launcher Script:**
- Locates BANQUET BEO folder
- Sets up Python environment
- Launches GUI application

**Icon File:**
- Your custom logo
- All required sizes (16px to 1024px)
- Retina-ready

**Metadata:**
- Application name and version
- Copyright information
- macOS compatibility settings

### **What's NOT Included:**

**Separate Data Files:**
- Templates (Invoice, Kitchen Prep, etc.)
- Database (Lariat_BEO_Database_Analysis.xlsx)
- Invoice files
- Reports

**Why:**
- Data remains in BANQUET BEO folder
- Easy to backup
- Easy to update app without affecting data
- Multiple computers can access same data

---

## 🚀 Advanced Installation

### **Install via Terminal**

```bash
# Copy app to Applications
cp -R "~/Desktop/BANQUET BEO/Lariat Manager.app" /Applications/

# Remove quarantine attribute (bypasses security warning)
xattr -cr "/Applications/Lariat Manager.app"

# Launch
open "/Applications/Lariat Manager.app"
```

### **Create Desktop Alias**

```bash
# Create symbolic link on Desktop
ln -s "/Applications/Lariat Manager.app" ~/Desktop/
```

### **Add to Dock Permanently**

```bash
# Add to Dock (appears at end)
defaults write com.apple.dock persistent-apps -array-add '<dict>
    <key>tile-data</key>
    <dict>
        <key>file-data</key>
        <dict>
            <key>_CFURLString</key>
            <string>/Applications/Lariat Manager.app</string>
            <key>_CFURLStringType</key>
            <integer>0</integer>
        </dict>
    </dict>
</dict>'

killall Dock
```

---

## 📚 Related Documentation

- **README.md** - Complete system overview
- **GUI_GUIDE.md** - Interface guide
- **AUTOMATION_GUIDE.md** - Automation details
- **QUICK_REFERENCE.txt** - Command reference

---

## ✅ Installation Checklist

Use this checklist to verify proper installation:

- [ ] Copied `Lariat Manager.app` to Applications folder
- [ ] Bypassed security warning (right-click → Open)
- [ ] Launched successfully
- [ ] Dashboard shows metrics
- [ ] Custom icon displays properly
- [ ] (Optional) Added to Dock for quick access
- [ ] (Optional) Tested Spotlight search

---

## 🎉 You're All Set!

Your Lariat Manager is now properly installed as a native macOS application!

**Launch anytime:**
- ⌘+Space → "Lariat" → Enter
- Applications folder → Lariat Manager
- Dock icon (if added)

**Features available:**
- Dashboard with business metrics
- Import invoices automatically
- Generate monthly reports
- Browse database
- Settings and configuration

---

## 📞 Support

**Issues with installation:**
1. Check this guide's Troubleshooting section
2. Run debug tool: `python3 debug_lariat.py`
3. Verify requirements in README.md

**Issues with app functionality:**
1. See GUI_GUIDE.md
2. Check /tmp/lariat_gui.log
3. Verify database file exists

---

*Installation Guide - Version 1.0*
*Last Updated: November 19, 2024*
*For: Lariat Manager v1.0*
