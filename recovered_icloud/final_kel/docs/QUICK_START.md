# miDiKompanion - Quick Start Guide

## Welcome to miDiKompanion

miDiKompanion is a therapeutic MIDI generation tool that transforms emotional states into musical patterns using a 216-node emotion thesaurus.

## Getting Started

### 1. Launch the Plugin

- **Plugin Mode**: Load miDiKompanion in your DAW (Logic Pro, Ableton Live, etc.)
- **Standalone Mode**: Launch miDiKompanion as a standalone application

### 2. Describe Your Emotion

In the "What's on your heart?" text field, describe how you're feeling:

- Examples:
  - "I'm feeling anxious about an upcoming presentation"
  - "Grateful for my family's support"
  - "Sad about losing a friend"

The system will analyze your text and map it to emotions in the 216-node thesaurus.

### 3. Select an Emotion (Alternative)

Instead of typing, you can:
- Click on the **Emotion Wheel** to select from 216 emotions
- Watch the **Emotion Radar** update to show Valence, Arousal, and Intensity

### 4. Adjust Parameters (Optional)

Fine-tune the musical output using the parameter sliders:

- **Valence**: Negative (sad/angry) to Positive (happy/joyful)
- **Arousal**: Calm to Excited
- **Intensity**: Subtle to Extreme
- **Complexity**: Simple to Complex
- **Humanize**: Mechanical to Human-like
- **Feel**: Tight to Loose
- **Dynamics**: Quiet to Loud
- **Bars**: Length of generated music (4-32 bars)

### 5. Generate MIDI

Click the **Generate** button to create MIDI based on your emotional input.

The system will:
- Process your emotion through the intent pipeline
- Generate MIDI patterns (melody, bass, chords, drums)
- Display the result in the Piano Roll Preview

### 6. Preview and Export

- **Preview**: Click **Preview** to hear the generated MIDI
- **Export to DAW**:
  - **Plugin Mode**: MIDI automatically flows to your DAW
  - **Standalone Mode**: Click **Export to DAW** to save a MIDI file (.mid)

## Project Management

### Saving Your Work

1. Click the **Project** menu button
2. Select **Save Project As...**
3. Choose a location and name your project
4. Project files are saved with `.midikompanion` extension

### Opening a Project

1. Click the **Project** menu button
2. Select **Open Project...**
3. Select a `.midikompanion` file
4. Your project state will be restored:
   - Plugin parameters
   - Emotion selections
   - Project metadata

**Note**: Generated MIDI metadata is saved, but you may need to regenerate MIDI after loading (this will be enhanced in v1.1).

### New Project

1. Click the **Project** menu button
2. Select **New Project**
3. This clears the current state so you can start fresh

## Understanding the Emotion Wheel

The Emotion Wheel displays 216 emotion nodes organized in a 6×6×6 structure:

- **6 Base Emotions**: Happy, Sad, Angry, Fear, Surprise, Disgust
- **36 Sub-Emotions**: 6 variations per base emotion
- **216 Total Nodes**: Each with unique Valence, Arousal, Dominance, and Intensity values

Clicking an emotion on the wheel:
- Updates the Emotion Radar visualization
- Adjusts parameter sliders to match the emotion
- Updates the Music Theory Panel (key, mode, tempo)

## Export Options

### MIDI Export

When exporting MIDI files, you can choose:

- **Format**:
  - SMF Type 0 (single track - all layers merged)
  - SMF Type 1 (multi-track - separate track per layer)
- **Include Vocals**: Export vocal notes if available
- **Include Lyrics**: Export lyrics as MIDI text events
- **Include Expression**: Export CC events for dynamics

### Project Export

Project files (`.midikompanion`) save:
- Complete plugin state (all parameters)
- Generated MIDI metadata
- Vocal notes and lyrics
- Emotion selections (216-node thesaurus)
- Project metadata (name, dates, version)

## Tips for Best Results

1. **Be Specific**: Detailed emotion descriptions yield better results
   - Good: "I feel overwhelmed by work deadlines and anxious about meeting expectations"
   - Less effective: "I feel bad"

2. **Use the Emotion Wheel**: If you know the emotion name, clicking it is faster than typing

3. **Experiment with Parameters**: After generation, adjust sliders and regenerate to explore variations

4. **Save Frequently**: Use the Project menu to save your work regularly

5. **Combine Methods**: Type a description AND select from the wheel for nuanced results

## Troubleshooting

### "No MIDI to Export"
- Make sure you've clicked **Generate** before exporting
- Check that at least one track (melody, bass, or chords) has notes

### Project Won't Load
- Ensure the file has `.midikompanion` extension
- Check that the file isn't corrupted
- Try opening in a text editor to verify it's valid JSON

### MIDI Sounds Wrong
- Adjust the **Humanize** slider for more natural timing
- Try different **Complexity** levels
- Experiment with **Feel** parameter for groove variations

## Next Steps

- See `EMOTION_WHEEL_GUIDE.md` for detailed emotion wheel usage
- See `EXPORT_WORKFLOW.md` for advanced export options
- Check `TROUBLESHOOTING.md` for common issues and solutions

## Support

For issues or questions, please refer to the troubleshooting guide or contact support.

---

**Version**: 1.0
**Last Updated**: 2025-01-XX
