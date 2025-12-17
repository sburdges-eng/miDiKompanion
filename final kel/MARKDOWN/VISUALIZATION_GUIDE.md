# Emotion & Musical Variable Visualization Guide

This guide describes the 3D visualizations available for exploring emotions and their musical mappings.

## Available Visualizations

### 1. Emotion Wheel (3D)
**File:** `visualize_3d_emotion_wheel.py`  
**Output:** `emotion_wheel_3d.html`

Visualizes emotions in 3D space using:
- **X-axis:** Valence (-1 = Negative, +1 = Positive)
- **Y-axis:** Arousal (0 = Calm, 1 = Excited)
- **Z-axis:** Intensity (0 = Subtle, 1 = Extreme)

**Usage:**
```bash
python3 visualize_3d_emotion_wheel.py
```

### 2. Musical Variables (3D)
**File:** `visualize_musical_variables_3d.py`  
**Outputs:**
- `musical_variables_tempo_harmony_articulation.html`
- `musical_variables_dynamics_rhythm_space.html`
- `musical_variables_dissonance_mode_register.html`

Creates three 3D visualizations showing musical parameters:

#### Visualization 1: Tempo × Harmony × Articulation
- **X-axis:** Tempo Modifier (0.7x - 2.0x)
- **Y-axis:** Harmonic Complexity (0.0 - 1.0)
- **Z-axis:** Articulation (0 = Legato, 1 = Staccato)

#### Visualization 2: Dynamics × Rhythm × Space
- **X-axis:** Average Velocity (30 - 127)
- **Y-axis:** Rhythm Regularity (0.0 = Free, 1.0 = Strict)
- **Z-axis:** Space Density (0.0 = Dense, 1.0 = Sparse)

#### Visualization 3: Dissonance × Mode × Register
- **X-axis:** Dissonance Tolerance (0.0 = None, 1.0 = High)
- **Y-axis:** Mode (0.0 = Locrian, 1.0 = Major)
- **Z-axis:** Register (0.0 = Low, 1.0 = High)

**Usage:**
```bash
python3 visualize_musical_variables_3d.py
```

### 3. Comprehensive Overlap Analysis
**File:** `visualize_comprehensive_overlap.py`  
**Outputs:**
- `comprehensive_overlap.html`
- `parallel_coordinates.html`

Creates comprehensive visualizations showing overlaps and commonalities:

#### Comprehensive Overlap (4-panel view)
1. **Emotion Space 3D:** Valence × Arousal × Intensity
2. **Musical Space 3D:** Tempo × Harmony × Articulation
3. **Emotion → Musical Mapping:** 2D scatter with density contours showing how valence maps to tempo
4. **Musical Clusters:** Shows emotions that cluster together in musical parameter space

#### Parallel Coordinates
Shows all variables simultaneously:
- Valence, Arousal, Intensity
- Tempo Modifier, Harmonic Complexity, Dissonance Tolerance
- Rhythm Regularity, Articulation, Space Density

**Usage:**
```bash
python3 visualize_comprehensive_overlap.py
```

## Interactive Features

All visualizations are interactive HTML files that can be opened in any web browser:

- **Rotate:** Click and drag
- **Zoom:** Scroll wheel
- **Pan:** Shift + Click and drag
- **Hover:** See detailed information about each point
- **Legend:** Click to show/hide categories

## Color Coding

Emotions are color-coded by category:
- **Joy:** Gold (#FFD700)
- **Sadness:** Royal Blue (#4169E1)
- **Anger:** Crimson (#DC143C)
- **Fear:** Dark Magenta (#8B008B)
- **Surprise:** Dark Orange (#FF8C00)
- **Disgust:** Forest Green (#228B22)

## Requirements

Install required packages:
```bash
pip3 install plotly scipy numpy
```

## Understanding the Visualizations

### Overlap Analysis
The comprehensive overlap visualization helps identify:
- **Emotional Similarities:** Emotions that cluster together in emotion space
- **Musical Similarities:** Emotions that share similar musical parameters
- **Mapping Patterns:** How emotional states translate to musical characteristics
- **Commonalities:** Regions where multiple emotions overlap in both spaces

### Clustering
The musical cluster view identifies emotions that:
- Share similar tempo, harmony, and articulation patterns
- May be musically interchangeable
- Form natural groupings in musical parameter space

### Parallel Coordinates
The parallel coordinates plot allows you to:
- See all variables at once
- Identify patterns across multiple dimensions
- Compare emotions across all parameters simultaneously
- Trace relationships between emotional and musical variables

## Tips for Analysis

1. **Look for clusters:** Dense regions indicate common patterns
2. **Compare spaces:** See how emotion space maps to musical space
3. **Identify outliers:** Points far from clusters may represent unique cases
4. **Trace relationships:** Use parallel coordinates to see how variables relate
5. **Explore overlaps:** Areas where multiple categories overlap show shared characteristics
