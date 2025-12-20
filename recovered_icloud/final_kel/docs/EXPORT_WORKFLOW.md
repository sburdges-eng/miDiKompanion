# Export Workflow Guide

## Overview

miDiKompanion supports multiple export formats for different workflows. This guide covers MIDI export and project file management.

## MIDI Export

### When to Export MIDI

Export MIDI when you want to:
- Import into your DAW for further editing
- Share MIDI files with collaborators
- Archive musical ideas
- Use in other music software

### Export Process

1. **Generate MIDI First**
   - Click the **Generate** button
   - Wait for generation to complete
   - Preview if desired

2. **Choose Export Method**
   - **Plugin Mode**: MIDI flows automatically to your DAW
   - **Standalone Mode**: Use the export dialog

3. **Select Export Options** (Standalone Mode)
   - **Format**:
     - SMF Type 0: Single track (all layers merged)
     - SMF Type 1: Multi-track (separate track per layer) - **Recommended**
   - **Include Vocals**: Export vocal notes if available
   - **Include Lyrics**: Export lyrics as MIDI text events
   - **Include Expression**: Export CC events for dynamics

4. **Save File**
   - Choose location
   - Name your file
   - Click Save

### MIDI File Contents

Exported MIDI files include:

- **Track 0**: Tempo and time signature (meta track)
- **Track 1**: Chords
- **Track 2**: Melody
- **Track 3**: Bass
- **Track 4**: Counter-melody (if present)
- **Track 5**: Pad (if present)
- **Track 6**: Strings (if present)
- **Track 7**: Fills (if present)
- **Track 8**: Rhythm (if present)
- **Track 9**: Drum Groove (if present)
- **Track 10+**: Vocals (if included)

### MIDI Format Details

#### SMF Type 0 (Single Track)
- All layers merged into one track
- Smaller file size
- Easier to import into simple sequencers
- **Use when**: You want a simple, single-track file

#### SMF Type 1 (Multi-Track) - **Recommended**
- Separate track per layer
- Better for DAW editing
- Preserves layer separation
- **Use when**: You want to edit individual layers in your DAW

### MIDI Channels

Default channel assignments:
- **Chords**: Channel 1
- **Melody**: Channel 2
- **Bass**: Channel 3
- **Counter-melody**: Channel 4
- **Pad**: Channel 5
- **Strings**: Channel 6
- **Fills**: Channel 7
- **Rhythm**: Channel 8
- **Drums**: Channel 10 (standard MIDI drum channel)

### Lyric Events

When "Include Lyrics" is enabled:
- Lyrics are exported as MIDI text events (0xFF 05)
- Distributed across the MIDI timeline
- Compatible with most DAWs and notation software

### Expression Events

When "Include Expression" is enabled:
- CC 11 (Expression) events are added
- Provides dynamics information
- Can be edited in your DAW

## Project File Export

### What Gets Saved

Project files (`.midikompanion`) save:

1. **Plugin State**
   - All parameter values (Valence, Arousal, Intensity, etc.)
   - Wound description text
   - Selected emotion IDs
   - Cassette state (Side A/B if applicable)

2. **Generated MIDI Metadata**
   - Tempo, key, mode, time signature
   - Track note counts
   - Length in beats
   - **Note**: Individual notes are not saved in v1.0 (user regenerates after loading)

3. **Vocal Data**
   - Vocal notes (if any)
   - Lyrics (if any)

4. **Emotion Selections**
   - Selected emotion node IDs (216-node thesaurus)
   - Primary emotion ID used for generation

5. **Project Metadata**
   - Project name
   - Created/modified dates
   - Version information

### Saving a Project

1. Click **Project** menu button
2. Select **Save Project As...**
3. Choose location
4. Enter project name
5. Click Save

**File Format**: `.midikompanion` (JSON-based)

### Opening a Project

1. Click **Project** menu button
2. Select **Open Project...**
3. Navigate to project file
4. Select `.midikompanion` file
5. Click Open

**After Loading**:
- All parameters are restored
- Emotion selections are restored
- You may need to regenerate MIDI (v1.0 limitation - will be improved in v1.1)

### Project File Compatibility

- **Version 1.0**: Current format
- **Future Versions**: Migration support will be added for backward compatibility

## Workflow Examples

### Example 1: Quick MIDI Export

1. Type emotion description
2. Click Generate
3. Click Export to DAW (standalone) or let it flow (plugin)
4. Import MIDI into your DAW
5. Continue editing in your DAW

### Example 2: Save Work in Progress

1. Generate MIDI
2. Adjust parameters
3. Click **Project** → **Save Project As...**
4. Name it "Anxiety Processing Session 1"
5. Later: Open project, regenerate if needed, continue work

### Example 3: Multiple Export Formats

1. Generate MIDI
2. Export as SMF Type 1 (multi-track) for DAW editing
3. Also export as SMF Type 0 (single track) for simple sequencers
4. Save project file for future reference

### Example 4: Collaborative Workflow

1. Generate MIDI in miDiKompanion
2. Export MIDI file
3. Share MIDI file with collaborator
4. Collaborator imports into their DAW
5. Both can work on the same musical foundation

## Best Practices

1. **Save Projects Regularly**: Don't lose your emotional state and parameter settings
2. **Use Descriptive Names**: "Grief Processing - Session 3" is better than "Project 1"
3. **Export After Generation**: Export MIDI immediately after generating to preserve the exact output
4. **Choose Right Format**: SMF Type 1 for DAW editing, Type 0 for simple use cases
5. **Include Lyrics**: Enable lyrics export if you plan to use them in your DAW

## Troubleshooting Export

### "No MIDI to Export" Error
- **Solution**: Click Generate first before exporting

### MIDI File Won't Open in DAW
- **Solution**: Try SMF Type 0 format (some older DAWs prefer single-track)
- **Solution**: Ensure file has `.mid` extension

### Project File Won't Load
- **Solution**: Check file extension is `.midikompanion`
- **Solution**: Verify file isn't corrupted (should be valid JSON)
- **Solution**: Check version compatibility

### Missing Tracks in DAW
- **Solution**: Use SMF Type 1 format (multi-track)
- **Solution**: Check that tracks have notes (empty tracks may not import)

## Advanced Export (v1.1+)

Future enhancements:
- **Stem Export**: Export individual tracks as audio files
- **Batch Export**: Export multiple projects at once
- **Export Templates**: Save export settings as presets
- **Full MIDI Restoration**: Restore all notes in project files (not just metadata)

---

**Version**: 1.0
**Last Updated**: 2025-01-XX
