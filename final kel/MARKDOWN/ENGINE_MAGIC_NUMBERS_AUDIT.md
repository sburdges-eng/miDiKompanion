# Engine Magic Numbers Audit Report

## Summary

This audit identifies magic numbers in engine files that should be replaced with named constants from `MusicConstants.h`.

## Findings

### FillEngine.cpp

**Velocity Boost Values** (lines 115-118):

- `-10` (Subtle intensity boost)
- `0` (Moderate intensity boost)
- `15` (Intense intensity boost)
- `30` (Explosive intensity boost)
- **Recommendation**: Add constants `FILL_VELOCITY_BOOST_SUBTLE`, `FILL_VELOCITY_BOOST_MODERATE`, `FILL_VELOCITY_BOOST_INTENSE`, `FILL_VELOCITY_BOOST_EXPLOSIVE`

**Velocity Variation Ranges**:

- `(-10, 10)` - Line 129, 194, 333
- `(-8, 8)` - Line 162, 268
- `(-5, 5)` - Line 297
- **Recommendation**: Add constants `VELOCITY_VARIATION_WIDE`, `VELOCITY_VARIATION_MEDIUM`, `VELOCITY_VARIATION_NARROW`

**Base Velocity Values**:

- `80` - Lines 131, 270, 335
- `70` - Line 164
- `85` - Line 196
- `60` - Line 230
- `90` - Line 300
- **Recommendation**: These are emotion-specific and context-specific, so they're acceptable as-is. However, they could reference `MIDI_VELOCITY_MEDIUM` as a base.

**Timing Offsets**:

- `-10` - Lines 138, 148, 180, 207, 213, 277, 283, 319
- `-5` - Lines 174, 250, 304
- `-2` - Line 347
- `15` - Line 337 (flamGap)
- **Recommendation**: Add constants `NOTE_DURATION_OFFSET_STANDARD`, `NOTE_DURATION_OFFSET_SMALL`, `NOTE_DURATION_OFFSET_TINY`, `FLAM_GRACE_NOTE_GAP_TICKS`

**Velocity Adjustments**:

- `-25` - Line 348 (flam grace note reduction)
- **Recommendation**: Add constant `FLAM_GRACE_NOTE_VELOCITY_REDUCTION`

**Velocity Ranges in Profiles** (lines 25, 33, 41, 49, 57, 66, 74, 82, 90, 98):

- These are emotion-specific configuration values (e.g., {30, 65}, {35, 70})
- **Recommendation**: These are acceptable as-is since they're emotion-specific profiles

### MelodyEngine.cpp

**Velocity Ranges in Profiles** (lines 56, 68, 80, 92, 104, 117, 129, 141, etc.):

- These are emotion-specific configuration values
- **Recommendation**: Acceptable as-is

**Pitch Ranges in Profiles** (lines 60, 72, 84, 96, 108, 121, 133, 145, etc.):

- These are emotion-specific configuration values
- **Recommendation**: Acceptable as-is

### BassEngine.cpp

**Velocity Ranges in Profiles** (lines 51, 61, 71, 81, 91, 102, 112, 122, 132, 142):

- These are emotion-specific configuration values
- **Recommendation**: Acceptable as-is

### DynamicsEngine.cpp

**Velocity Ranges in Profiles** (lines 24, 32, 40, 48, 56, 65, 73, 81, 89, 97):

- These are emotion-specific configuration values
- **Recommendation**: Acceptable as-is

## Priority Fixes

### High Priority (Should be fixed)

1. **FillEngine velocity boost values** - Used in switch statement, should be constants
2. **FillEngine flamGap (15 ticks)** - Used for timing calculation, should be constant
3. **FillEngine flam grace note velocity reduction (-25)** - Should be constant

### Medium Priority (Nice to have)

1. **Velocity variation ranges** - Could be constants but are context-specific
2. **Note duration offsets** - Small timing adjustments, could be constants

### Low Priority (Acceptable as-is)

1. **Emotion-specific velocity ranges** - These are configuration values, not magic numbers
2. **Emotion-specific pitch ranges** - These are configuration values, not magic numbers
3. **Base velocity values** - Context-specific, acceptable as-is

## Implementation Plan

1. Add new constants to `MusicConstants.h`:
   - Fill engine velocity boost constants
   - Flam timing constant
   - Flam velocity reduction constant
   - Note duration offset constants (optional)

2. Replace magic numbers in `FillEngine.cpp`:
   - Replace velocity boost values in `intensityToVelocityBoost()`
   - Replace `flamGap = 15` with constant
   - Replace `velocity - 25` with constant

3. Verify no regressions in tests

## Notes

- Most "magic numbers" in engine files are actually emotion-specific configuration values
- The real magic numbers are the hardcoded offsets, gaps, and boost values used in calculations
- Profile velocity/pitch ranges are intentional configuration, not magic numbers
