# Troubleshooting Guide

## Common Issues and Solutions

### MIDI Generation

#### "No MIDI Generated" or Empty Output

**Symptoms**: Generate button completes but no MIDI appears in preview

**Solutions**:
1. Check that you have selected an emotion or entered wound text
2. Verify parameter sliders are not all at extremes
3. Try a different emotion selection
4. Check console/logs for error messages
5. Restart the plugin/application

#### MIDI Sounds Wrong or Unmusical

**Symptoms**: Generated MIDI doesn't match expected emotion

**Solutions**:
1. **Adjust Humanize**: Increase humanize slider for more natural timing
2. **Try Different Complexity**: Lower complexity for simpler patterns
3. **Check Music Theory Panel**: Verify key and mode settings
4. **Regenerate**: Sometimes regeneration produces better results
5. **Fine-tune Parameters**: Small adjustments can make big differences

### Project Management

#### Project Won't Save

**Symptoms**: Save dialog appears but file isn't created

**Solutions**:
1. Check file permissions on target directory
2. Ensure sufficient disk space
3. Verify file extension is `.midikompanion`
4. Try a different save location
5. Check error message in alert dialog

#### Project Won't Load

**Symptoms**: "Load Failed" error when opening project

**Solutions**:
1. **Verify File Format**: Ensure file has `.midikompanion` extension
2. **Check File Integrity**: Open in text editor - should be valid JSON
3. **Version Compatibility**: v1.0 projects may not work with future versions (migration coming in v1.1)
4. **Try Opening in Text Editor**: Verify JSON structure is valid
5. **Check Error Message**: Alert dialog shows specific error

#### Project Loads But MIDI is Missing

**Symptoms**: Project opens but no MIDI in preview

**Solutions**:
1. **This is Expected in v1.0**: Project files save metadata, not individual notes
2. **Regenerate MIDI**: Click Generate after loading project
3. **Check Metadata**: Verify tempo, key, mode are correct
4. **Future Enhancement**: v1.1 will restore full MIDI data

### MIDI Export

#### "No MIDI to Export" Error

**Symptoms**: Export button shows error message

**Solutions**:
1. **Generate First**: Click Generate before exporting
2. **Check Tracks**: Ensure at least one track (melody, bass, or chords) has notes
3. **Verify Generation**: Check Piano Roll Preview shows notes

#### MIDI File Won't Open in DAW

**Symptoms**: DAW rejects imported MIDI file

**Solutions**:
1. **Try SMF Type 0**: Some DAWs prefer single-track format
2. **Check File Extension**: Ensure `.mid` extension
3. **Verify File Size**: Empty or corrupted files won't open
4. **Try Different DAW**: Test in multiple DAWs to isolate issue
5. **Re-export**: Export again with different options

#### Missing Tracks in DAW

**Symptoms**: Only some tracks appear after import

**Solutions**:
1. **Use SMF Type 1**: Multi-track format preserves all tracks
2. **Check Track Count**: Verify all tracks have notes (empty tracks may not import)
3. **DAW Settings**: Some DAWs hide empty tracks - check DAW preferences

### UI Issues

#### Emotion Wheel Not Responding

**Symptoms**: Clicking emotions doesn't update parameters

**Solutions**:
1. **Check Connection**: Verify EmotionWorkstation is properly initialized
2. **Restart Plugin**: Reload plugin in DAW or restart standalone
3. **Check Logs**: Look for error messages in console

#### Parameter Sliders Don't Update

**Symptoms**: Sliders don't reflect emotion selection

**Solutions**:
1. **Wait for Update**: Updates happen asynchronously
2. **Manual Adjustment**: You can manually adjust sliders
3. **Check APVTS**: Verify AudioProcessorValueTreeState is connected

### Performance

#### Slow Generation

**Symptoms**: Generate takes a long time

**Solutions**:
1. **Reduce Bars**: Lower bar count generates faster
2. **Lower Complexity**: Simpler patterns are faster
3. **Check System Resources**: Close other applications
4. **Future**: ML models may add latency (optional feature)

#### High CPU Usage

**Symptoms**: Plugin uses excessive CPU

**Solutions**:
1. **Disable Preview Generation**: Turn off real-time preview if enabled
2. **Reduce Update Rate**: Lower timer callback frequency
3. **Check for Loops**: Ensure no infinite update loops
4. **Profile**: Use performance profiler to identify bottlenecks

### Audio Issues

#### No Audio Output (Standalone)

**Symptoms**: Can't hear generated MIDI

**Solutions**:
1. **Check Audio Settings**: Verify audio device is selected
2. **Check Volume**: Ensure volume is not muted
3. **Test with DAW**: Try plugin mode to isolate issue
4. **Check MIDI Routing**: Verify MIDI is being sent to synthesizer

#### Audio Glitches

**Symptoms**: Clicks, pops, or dropouts

**Solutions**:
1. **Increase Buffer Size**: Larger buffers reduce glitches
2. **Check Sample Rate**: Match DAW sample rate
3. **Disable Other Plugins**: Isolate miDiKompanion
4. **Check System Load**: Close other applications

## Getting Help

### Before Reporting Issues

1. **Check Version**: Ensure you're using v1.0 or later
2. **Reproduce**: Can you consistently reproduce the issue?
3. **Check Logs**: Look for error messages
4. **Try Workarounds**: Test solutions above

### Information to Provide

When reporting issues, include:

- **Version**: miDiKompanion version number
- **Platform**: macOS/Windows, DAW name and version
- **Steps to Reproduce**: Exact steps that cause the issue
- **Expected vs Actual**: What should happen vs what actually happens
- **Error Messages**: Copy exact error text
- **System Info**: OS version, DAW version, hardware specs

### Known Limitations (v1.0)

These are not bugs, but planned enhancements:

1. **MIDI Restoration**: Project files don't restore individual notes (v1.1)
2. **ML Models**: ML enhancement is optional and may not be available (v1.1+)
3. **Stem Export**: Audio stem export coming in v1.1
4. **Batch Operations**: Multiple project operations coming in v1.1

## Quick Fixes

### Reset to Defaults

1. Click **Project** → **New Project**
2. This clears all state and resets to defaults

### Clear Cache (if issues persist)

1. Close miDiKompanion
2. Delete cache files (location varies by platform)
3. Restart application

### Reinstall (Last Resort)

1. Uninstall miDiKompanion
2. Delete preferences/cache
3. Reinstall from installer
4. Test with default settings

---

**Version**: 1.0
**Last Updated**: 2025-01-XX
