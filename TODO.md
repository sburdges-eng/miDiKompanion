# Dart Strike Game - TODO List

## Overview
This TODO tracks the conversion of bowling pin values to dartboard black section values.

## Task: Change Pin Values to Dartboard Black Sections

### Requirements
- Each black section on a dartboard corresponds to one pin
- Standard dartboard has alternating black and white/cream sections
- 10 pins need to map to 10 black dartboard values

### Black Sections on a Standard Dartboard (clockwise from top)
The black sections in the single number ring are:
1. 20 (top)
2. 3
3. 11
4. 8
5. 16
6. 7
7. 19
8. 12
9. 18
10. 9

### Mapping Plan
- Pin 1 (front) → 20 (most valuable, top center)
- Pin 2 → 3
- Pin 3 → 11
- Pin 4 → 8
- Pin 5 → 16
- Pin 6 → 7
- Pin 7 → 19
- Pin 8 → 12
- Pin 9 → 18
- Pin 10 → 9

### Implementation Checklist
- [x] Create TODO.md file
- [x] Update Pin class to display dartboard values instead of pin numbers
- [x] Update pin layout documentation
- [x] Test the application
- [x] Verify all functionality works with new values

## Completed Successfully ✓

All tasks have been completed. The dart game now displays dartboard black section values instead of pin numbers:
- Pin 1 → 20
- Pin 2 → 3
- Pin 3 → 11
- Pin 4 → 8
- Pin 5 → 16
- Pin 6 → 7
- Pin 7 → 19
- Pin 8 → 12
- Pin 9 → 18
- Pin 10 → 9

The bowling game logic remains unchanged - only the display values have been updated.

## Notes
- Keep the same 10-pin bowling triangle layout
- Only change the displayed numbers on the pins
- Maintain all bowling scoring logic
- The pin IDs internally can stay 1-10 for logic purposes
