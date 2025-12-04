# 🎯 Dart Strike - Python/Qt Version with Dartboard Values

**Dart Strike** is a unique bowling scoring game where pins are labeled with dartboard black section values instead of traditional pin numbers.

## 🎯 Key Feature: Dartboard Pin Values

Instead of traditional bowling pin numbers (1-10), this version displays **dartboard black section values**:

### Pin Layout with Dartboard Values
```
       19  12  18   9      (Back row - Pins 7, 8, 9, 10)
          8  16   7        (Middle row - Pins 4, 5, 6)
            3  11          (Front row - Pins 2, 3)
             20            (Front pin - Pin 1)
```

### Black Section Mapping
Each pin displays a value from the black sections of a standard dartboard:
- **Pin 1** → **20** (top/center, most valuable)
- **Pin 2** → **3**
- **Pin 3** → **11**
- **Pin 4** → **8**
- **Pin 5** → **16**
- **Pin 6** → **7**
- **Pin 7** → **19**
- **Pin 8** → **12**
- **Pin 9** → **18**
- **Pin 10** → **9**

## 📱 Features

- **Dartboard Values Display**: Pins show dartboard black section values (not 1-10)
- **Interactive Pin Layout**: Click pins to knock them down
- **Traditional Bowling Scoring**: Full implementation of strikes, spares, and open frames
- **Multi-Player Support**: Up to 8 players can compete
- **Auto Pin Reset**: Pins automatically reset after frames and turns
- **Game State Management**: Track all players and frames
- **Professional Scorecard**: View cumulative scores and frame-by-frame breakdown
- **10th Frame Rules**: Proper bonus throws for strikes and spares
- **Cross-Platform**: Runs on Windows, Mac, and Linux

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PySide6 (Qt for Python)

### Installation

1. **Install PySide6**
   ```bash
   pip install PySide6
   ```

2. **Run the application**
   ```bash
   python3 dart_strike_qt.py
   ```

## 📋 How to Play

### Setup
1. Click "Add Player" to add players (enter name)
2. Add up to 8 players
3. Click "Start Game"

### Gameplay
1. **Click dartboard values** to select which pins to knock down
   - White pins = standing
   - Red pins = knocked down
2. **Submit Throw** to record your selection
3. **Automatic Turn Rotation**: Game switches players after each frame
4. **View Scorecard**: Real-time scoring on the right side

### Scoring Rules

The game uses traditional bowling scoring (based on number of pins knocked, not their values):

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

## 🎨 Visual Design

- **Standing pins**: White with black border, showing dartboard value
- **Knocked pins**: Red with darker border
- **Pin counter**: Shows "Pins Down: X" at bottom
- **Current player**: Highlighted in blue on scorecard
- **Status bar**: Shows current player, frame, and throw

## 🔍 Understanding the Dartboard Values

The values shown on the pins come from a standard dartboard's black sections. On a dartboard:
- Black and white/cream sections alternate
- Numbers go from 1-20 around the board
- The 10 black sections used are: **20, 3, 11, 8, 16, 7, 19, 12, 18, 9**

**Note**: While the pins display dartboard values, the scoring is still traditional bowling (counting number of pins knocked, not their dartboard value).

## 📁 Files

- **dart_strike_qt.py** - Main application file
- **TODO.md** - Task tracking and implementation notes
- **README_DART_VALUES.md** - This file

## 🧪 Testing

The application has been tested with:
- Pin value mapping verification
- Display correctness
- Game logic integrity
- Player management
- Scoring calculations

All tests pass successfully!

## 🎯 Why Dartboard Values?

This unique twist combines the visual appeal of darts with bowling mechanics:
- Makes the game more visually interesting
- Creates a hybrid darts/bowling theme
- Each pin has a unique, meaningful value
- Still maintains traditional bowling scoring

## 💡 Technical Details

- **Framework**: PySide6 (Qt for Python)
- **Language**: Python 3.9+
- **Architecture**: Single-file application with clear class separation
- **Pin Class**: Enhanced with dartboard value mapping
- **Scoring**: Traditional bowling algorithm (unchanged)

## 🔄 Future Enhancements

Potential additions:
- [ ] Sound effects for pin knockdowns
- [ ] Save/load game state
- [ ] Player statistics
- [ ] Dark mode theme
- [ ] Alternative scoring (using dartboard values)
- [ ] Animations for knocked pins

## 📞 Support

For issues or questions, refer to:
- The TODO.md file for implementation details
- The main project documentation
- Python/PySide6 documentation

---

**Enjoy bowling with dartboard values! 🎯🎳**
