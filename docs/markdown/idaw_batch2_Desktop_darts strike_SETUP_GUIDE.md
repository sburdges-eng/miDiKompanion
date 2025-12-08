# 🚀 Quick Xcode Setup Guide

Follow these steps to get Dart Strike running in Xcode:

## Step 1: Create New Project (2 minutes)

1. **Open Xcode**
2. **File > New > Project** (or ⇧⌘N)
3. Select **iOS** tab
4. Choose **App** template
5. Click **Next**

## Step 2: Project Configuration

Fill in these details:
- **Product Name**: `DartStrike`
- **Team**: Select your Apple ID (or "None" for simulator only)
- **Organization Identifier**: `com.yourdomain` (use your own)
- **Interface**: **SwiftUI**
- **Language**: **Swift**
- **Storage**: Uncheck "Use Core Data"
- **Testing**: Uncheck both test options (optional)
- Click **Next**
- Choose save location
- Click **Create**

## Step 3: Add Project Files (5 minutes)

### Create Folder Structure

In Xcode's Project Navigator (left sidebar):

1. **Right-click on DartStrike folder**
2. Select **New Group**
3. Create these groups:
   - `Models`
   - `Views`
   - `Utilities`

### Add Files to Groups

**Drag and drop** or **copy files** into appropriate groups:

#### Models Group
- `GameModel.swift`

#### Views Group
- `GameView.swift`
- `ScorecardView.swift`
- `PlayerSetupView.swift`

#### Root (DartStrike folder)
- `DartStrikeApp.swift` (replace the existing one)

#### Utilities Group
- `PersistenceManager.swift`

### Delete Default Files (if present)
- Delete `ContentView.swift` (we have our own)
- Keep `Assets.xcassets`
- Keep `Preview Content` folder

## Step 4: Configure Build Settings

1. **Select your project** in the navigator (top item)
2. Click on **DartStrike** under TARGETS
3. Go to **General** tab
4. Set these values:

   **Deployment Info:**
   - **Minimum Deployments**: iOS 15.0
   - **iPhone Orientation**: Portrait
   - **iPad Orientation**: All (optional)
   
   **App Icons and Launch Screen:**
   - (Leave default for now - add custom icons later)

## Step 5: Build and Run

1. **Select Simulator**: 
   - Click device menu (top-left, next to "DartStrike")
   - Choose **iPhone 14** or **iPhone 15**

2. **Build the Project**:
   - Press **⌘B** or Product > Build
   - Wait for "Build Succeeded" message

3. **Run the App**:
   - Press **⌘R** or Product > Run
   - App should launch in simulator

## ✅ Verification Checklist

After the app launches, test these features:

- [ ] Player setup screen appears
- [ ] Can add multiple players
- [ ] "Start Game" button works
- [ ] Pin layout displays correctly (bowling pin formation)
- [ ] Tapping pins toggles their state (blue/red)
- [ ] "Submit Throw" advances the game
- [ ] Pins reset after 2 throws
- [ ] Pins reset when changing players
- [ ] Scorecard displays correctly
- [ ] Scores calculate properly

## 🐛 Common Issues

### Build Errors

**"Cannot find DartStrikeApp in scope"**
- Make sure `DartStrikeApp.swift` is in the project target
- Check file is not grayed out in navigator

**"Missing import"**
- Ensure all files have `import SwiftUI` at top
- Check `import Foundation` for model files

**"Duplicate symbol"**
- Remove the original `ContentView.swift`
- Make sure you're not including preview files

### Runtime Issues

**App crashes on launch**
- Check Console output (⌘⇧Y to show)
- Verify all `@StateObject` and `@ObservedObject` are used correctly

**Pins don't reset**
- This is already fixed in the code
- Make sure you're using the provided `GameModel.swift`

## 📱 Running on Your iPhone

1. **Connect iPhone** via USB
2. **Select your iPhone** from device menu
3. **Trust Developer**:
   - iPhone Settings > General > Device Management
   - Tap your Apple ID
   - Tap "Trust"
4. **Run** (⌘R)

**Note**: Free Apple ID allows 7-day installations. Developer account ($99/year) allows permanent installs.

## 🎨 Optional: Add App Icon

1. Open `Assets.xcassets`
2. Select `AppIcon`
3. Drag icon images (various sizes) into the wells
4. Or use a tool like https://appicon.co to generate all sizes

## 📝 Next Steps

Once running successfully:

1. **Test all features** using the test scenarios in README.md
2. **Play a full game** to verify scoring
3. **Test game saving** by closing and reopening app
4. **Customize** colors, sizes, or layout if desired

## 💡 Pro Tips

- **Hot Reload**: Use Live Preview (Canvas) for faster UI testing
- **Console Logs**: Look for "✅" and "❌" messages in debug output
- **Breakpoints**: Click line numbers to add breakpoints for debugging
- **Simulator**: Press ⌘K to toggle software keyboard

---

**Estimated Setup Time**: 10-15 minutes
**Difficulty**: Beginner-Friendly

Need help? Check README.md for detailed documentation! 🎯🎳
