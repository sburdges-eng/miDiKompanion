# iDAW User Guide

## Introduction

iDAW (Intelligent Digital Audio Workstation) is an emotion-to-music AI system that converts your emotional input into musical MIDI output.

**Core Philosophy**: "Interrogate Before Generate" - We make you braver, not lazier.

## Installation

### macOS

1. Download the plugin installer
2. Run the installer package
3. Plugins will be installed to:
   - AU: `/Library/Audio/Plug-Ins/Components/`
   - VST3: `/Library/Audio/Plug-Ins/VST3/`

### Windows

1. Download the plugin installer
2. Run the installer executable
3. Plugins will be installed to:
   - VST3: `C:\Program Files\Common Files\VST3\`

### Linux

1. Extract the plugin archive
2. Copy to: `~/.vst3/`
3. Rescan plugins in your DAW

## Quick Start

### Basic Workflow

1. **Load Plugin**: Add iDAW plugin to a track in your DAW
2. **Input Emotion**: Enter emotional text or provide audio input
3. **Generate**: Click generate to create MIDI matching your emotion
4. **Refine**: Adjust parameters and regenerate as needed

### Emotion Input

**Text Input**:
- Describe your emotional state: "I feel serene and peaceful"
- Describe your creative goal: "I want to express longing"
- Use emotion words: "serenity", "longing", "wonder", "grief"

**Audio Input**:
- Provide reference audio with emotional content
- System extracts emotion from audio features

**Biometric Input** (future):
- Heart rate, skin conductance, etc.

## Plugin Parameters

### Emotion Controls

- **Valence**: Positive/negative emotion (-1.0 to 1.0)
- **Arousal**: Energy level (0.0 to 1.0)
- **Dominance**: Control/power (0.0 to 1.0)
- **Tension**: Musical tension (0.0 to 1.0)

### Generation Controls

- **Creativity**: Rule-breaking level (0.0 = strict, 1.0 = maximum)
- **Tempo**: Generated tempo (BPM)
- **Key**: Musical key
- **Mode**: Major/minor/other modes

### Output Controls

- **MIDI Output**: Route to MIDI track
- **Preview**: Preview generated MIDI
- **Export**: Export MIDI file

## Workflow Guide

### Expressing Emotions

1. **Identify Emotion**: What are you feeling?
2. **Input Emotion**: Type or speak your emotion
3. **Review Questions**: System may ask clarifying questions
4. **Generate**: Create music matching your emotion
5. **Refine**: Adjust and regenerate as needed

### Rule-Breaking

The system can intentionally break musical rules for emotional expression:

- **Harmony**: "Wrong" chords for tension
- **Rhythm**: Displaced beats for groove
- **Arrangement**: Unusual mixes for emotion

Adjust the "Creativity" parameter to control rule-breaking.

## Tips

1. **Be Specific**: More specific emotions produce better results
2. **Iterate**: Generate multiple versions and refine
3. **Experiment**: Try different rule-breaking levels
4. **Reference**: Provide reference audio for style matching

## Troubleshooting

### Plugin Not Loading

- Check plugin format matches your DAW
- Verify models are in Resources/models/
- Check DAW plugin scan

### Poor Quality Output

- Be more specific with emotion input
- Adjust creativity parameter
- Try different emotional inputs

### Performance Issues

- Reduce buffer size
- Close other plugins
- Check CPU usage

## FAQ

**Q: How does the emotion-to-music conversion work?**  
A: The system uses neural networks trained on emotion-labeled music data to convert your emotional input into musical patterns.

**Q: Can I use my own MIDI files?**  
A: Yes, you can use MIDI files as reference or starting points.

**Q: Does it work in real-time?**  
A: Yes, the system is optimized for real-time audio processing with <10ms latency.

**Q: Can I customize the models?**  
A: Advanced users can retrain models with custom datasets.

## Support

For issues and questions:
- Check documentation in `docs/`
- Review troubleshooting section
- Check GitHub issues

---

**Version**: 1.0  
**Last Updated**: 2025-12-18
