# 🎯🎳 Dart Strike - JavaFX Desktop Application

**Dart Strike** is a hybrid darts-bowling game for desktop where players click pins to knock them down and score points using traditional bowling rules.

## 📱 Features

- **Interactive Pin Layout**: Click bowling pins to knock them down
- **Traditional Bowling Scoring**: Full implementation of strikes, spares, and open frames
- **Multi-Player Support**: Up to 8 players can compete
- **Auto Pin Reset**: Pins automatically reset after frames and turns ✅ **BUG FIXED**
- **Game State Saving**: Resume games after closing the app
- **Professional Scorecard**: View cumulative scores and frame-by-frame breakdown
- **10th Frame Rules**: Proper bonus throws for strikes and spares
- **Cross-Platform**: Runs on Windows, Mac, and Linux

## 🏗️ Project Structure

```
DartStrike-JavaFX/
├── pom.xml                              # Maven build configuration
├── src/
│   ├── main/
│   │   ├── java/com/dartstrike/
│   │   │   ├── DartStrikeApp.java       # Main application entry
│   │   │   ├── models/
│   │   │   │   ├── Pin.java             # Pin model
│   │   │   │   ├── Frame.java           # Frame model
│   │   │   │   ├── Player.java          # Player model
│   │   │   │   └── GameModel.java       # Core game logic ✅ PIN RESET FIXED
│   │   │   ├── controllers/
│   │   │   │   ├── PlayerSetupController.java
│   │   │   │   ├── GameViewController.java
│   │   │   │   └── ScorecardController.java
│   │   │   └── utils/
│   │   │       └── PersistenceManager.java
│   │   └── resources/
│   │       └── fxml/
│   │           ├── PlayerSetupView.fxml
│   │           ├── GameView.fxml
│   │           └── ScorecardView.fxml
```

## 🚀 Getting Started

### Prerequisites

- **Java JDK 11+** (Java 17 recommended)
- **Maven 3.6+**
- **JavaFX 17+** (handled by Maven)

### Installation Steps

#### Option 1: Using Maven (Recommended)

1. **Clone or Download Project**
   ```bash
   cd DartStrike-JavaFX
   ```

2. **Build with Maven**
   ```bash
   mvn clean package
   ```

3. **Run the Application**
   ```bash
   mvn javafx:run
   ```

#### Option 2: Using IDE (IntelliJ IDEA / Eclipse)

**IntelliJ IDEA:**
1. Open IntelliJ IDEA
2. File → Open → Select `DartStrike-JavaFX` folder
3. Wait for Maven to import dependencies
4. Right-click `DartStrikeApp.java` → Run

**Eclipse:**
1. Open Eclipse
2. File → Import → Maven → Existing Maven Projects
3. Select `DartStrike-JavaFX` folder
4. Right-click project → Run As → Java Application
5. Select `DartStrikeApp` as main class

#### Option 3: Command Line with Java

```bash
# Compile
javac --module-path $PATH_TO_FX --add-modules javafx.controls,javafx.fxml -d out src/main/java/com/dartstrike/**/*.java

# Run
java --module-path $PATH_TO_FX --add-modules javafx.controls,javafx.fxml -cp out com.dartstrike.DartStrikeApp
```

Replace `$PATH_TO_FX` with your JavaFX lib path.

## 📋 How to Play

### Setup
1. Launch the application
2. Add players (up to 8) using the player setup screen
3. Click "Start Game"

### Gameplay
1. **Select Pins**: Click the bowling pins you want to knock down (toggle red/blue)
2. **Submit Throw**: Press "Submit Throw" to lock in your selection
3. **Automatic Turn Rotation**: The game automatically switches players after each frame
4. **View Scores**: Click "Scorecard" to see detailed scoring

### Scoring Rules

#### Strike (X)
- Knock down all 10 pins on first throw
- Score: 10 + next 2 throws

#### Spare (/)
- Knock down all 10 pins using both throws
- Score: 10 + next 1 throw

#### Open Frame
- Less than 10 pins knocked down
- Score: Actual number of pins

#### 10th Frame
- Strike on 1st ball: Get 2 bonus throws
- Spare on 2nd ball: Get 1 bonus throw
- Maximum: 30 points in 10th frame

**Perfect Game**: 12 strikes = 300 points

## 🔧 Testing the Pin Reset Fix

The app includes the debugged pin reset functionality:

### Test Scenario 1: Basic Pin Reset
1. Start a 2-player game
2. Player 1: Select and knock down 5 pins, submit throw
3. Player 1: Select remaining 5 pins, submit second throw
4. **✅ Verify**: All pins reset to standing for Player 2

### Test Scenario 2: Strike Reset
1. Player knocks down all 10 pins (strike)
2. Submit throw
3. **✅ Verify**: Pins immediately reset for next frame

### Test Scenario 3: Frame Transition
1. Complete an entire frame (2 throws)
2. **✅ Verify**: Pins reset for next player's turn

### Test Scenario 4: 10th Frame
1. Reach the 10th frame
2. Get a strike on first throw
3. **✅ Verify**: Pins reset for bonus throws

## 💾 Game State Persistence

The app automatically saves your game progress:

- **Auto-Save**: When closing the application
- **Resume**: Prompted to resume when reopening
- **File Location**: `dartstrike_save.dat` in working directory
- **Manual Reset**: Use "New Game" button to start fresh

Saved data includes:
- All player scores and frames
- Current player turn
- Pin positions
- Game progress

## 🐛 Debugging Tips

### Build Issues

**Maven dependencies not downloading:**
```bash
mvn clean install -U
```

**JavaFX not found:**
- Ensure Java 11+ is installed
- Check Maven is using correct JDK
- Try: `export JAVA_HOME=/path/to/jdk11`

**Module errors:**
- Ensure `module-info.java` is not present (we're using classpath mode)
- Or add proper module configuration if needed

### Runtime Issues

**Pins not resetting:**
- This is already fixed in the code
- Make sure you're using the provided `GameModel.java`

**FXML load errors:**
- Verify FXML files are in `src/main/resources/fxml/`
- Check controller class names match

**Save file errors:**
- Check write permissions in working directory
- Try deleting `dartstrike_save.dat` if corrupted

## 🎨 Customization

### Change Pin Colors
In `GameViewController.java`:
```java
// Standing pin
"-fx-background-color: white; -fx-border-color: blue;"

// Knocked down pin
"-fx-background-color: #ffcccc; -fx-border-color: red;"
```

### Modify Layout
Edit FXML files in `src/main/resources/fxml/`:
- `PlayerSetupView.fxml` - Player setup screen
- `GameView.fxml` - Main game interface
- `ScorecardView.fxml` - Scorecard display

### Add Sound Effects (Future Enhancement)
```java
// In GameViewController.java
AudioClip sound = new AudioClip(getClass().getResource("/sounds/pin.wav").toString());
sound.play();
```

## 📦 Building Executable JAR

### Create Runnable JAR:
```bash
mvn clean package
```

The JAR will be in `target/DartStrike-JavaFX-1.0.0.jar`

### Run the JAR:
```bash
java -jar target/DartStrike-JavaFX-1.0.0.jar
```

## 🖥️ System Requirements

**Minimum:**
- Java 11+
- 2 GB RAM
- 100 MB disk space
- 1024x768 resolution

**Recommended:**
- Java 17+
- 4 GB RAM
- 200 MB disk space
- 1920x1080 resolution

## 🔄 Version History

**Version 1.0.0** (Current)
- ✅ Core bowling gameplay
- ✅ Pin reset functionality (debugged from Java/iOS)
- ✅ Traditional scoring system
- ✅ Multi-player support
- ✅ Game state persistence
- ✅ Professional scorecard
- ✅ JavaFX modern UI

## 🎯 Future Enhancements (Optional)

- [ ] Sound effects for pin knockdowns
- [ ] Game statistics and history
- [ ] Player profiles and avatars
- [ ] Handicap scoring system
- [ ] Dark mode theme
- [ ] Tournament mode
- [ ] Network multiplayer
- [ ] Database persistence (instead of file)

## 📞 Support

For issues or questions:
1. Check the debugging tips section
2. Review Maven logs for build errors
3. Test with the provided test scenarios
4. Check JavaFX documentation: https://openjfx.io/

## 📄 Technologies Used

- **Java 11+** - Programming language
- **JavaFX 17** - UI framework
- **Maven** - Build tool
- **FXML** - UI markup language
- **Java Serialization** - Game state persistence

## 📝 Comparison: JavaFX vs iOS

| Feature | iOS (SwiftUI) | JavaFX (Desktop) |
|---------|---------------|------------------|
| Platform | iOS Native | Windows/Mac/Linux |
| UI Framework | SwiftUI | JavaFX + FXML |
| State Management | @Published/@State | ObservableList/Property |
| Persistence | UserDefaults | File Serialization |
| Pin Reset Bug | ✅ Fixed | ✅ Fixed |
| Performance | Excellent | Excellent |
| Distribution | App Store | JAR file |

## 🎉 Quick Start Checklist

- [ ] Java 11+ installed
- [ ] Maven installed
- [ ] Project downloaded/cloned
- [ ] `mvn clean package` successful
- [ ] `mvn javafx:run` launches app
- [ ] Can add players and start game
- [ ] Pins reset correctly between turns
- [ ] Scoring calculates properly
- [ ] Game saves and loads correctly

**Ready to bowl? 🎯🎳**

---

## 📧 Contact

For questions or feedback about Dart Strike JavaFX, please reach out through GitHub or project documentation.
