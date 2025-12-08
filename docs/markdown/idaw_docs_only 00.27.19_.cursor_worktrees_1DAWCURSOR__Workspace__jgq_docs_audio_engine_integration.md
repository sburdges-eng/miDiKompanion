# Audio Engine Integration (TASK 8)

## Objective

Integrate Rust audio engine with CPAL for low-latency MIDI playback in the Tauri application.

## Implementation Plan

### 1. Add Dependencies to Cargo.toml

```toml
[dependencies]
cpal = "0.15"
tokio = { version = "1", features = ["full"] }
midir = "0.9"  # For MIDI input/output
```

### 2. Create Audio Module Structure

```
src-tauri/src/
  audio/
    mod.rs          # Audio engine initialization
    midi_player.rs  # MIDI playback logic
    synth.rs        # Simple synthesizer
```

### 3. Audio Engine Initialization (audio/mod.rs)

```rust
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, Stream, StreamConfig};

pub struct AudioEngine {
    host: Host,
    device: Device,
    config: StreamConfig,
    stream: Option<Stream>,
}

impl AudioEngine {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let host = cpal::default_host();
        let device = host.default_output_device()
            .ok_or("No output device available")?;
        let config = device.default_output_config()?.into();

        Ok(Self {
            host,
            device,
            config,
            stream: None,
        })
    }

    pub fn start_stream<F>(&mut self, callback: F) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(&mut [f32]) + Send + 'static,
    {
        let stream = self.device.build_output_stream(
            &self.config,
            callback,
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )?;

        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }
}
```

### 4. MIDI Player (audio/midi_player.rs)

```rust
use midir::{MidiInput, MidiOutput};
use std::sync::Arc;
use std::sync::Mutex;

pub struct MidiPlayer {
    // MIDI file data
    // Synthesizer state
    // Playback position
}

impl MidiPlayer {
    pub fn load_midi_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Load MIDI file
        // Parse events
        // Initialize synthesizer
        Ok(Self {})
    }

    pub fn play(&mut self) {
        // Start playback
        // Generate audio samples from MIDI events
    }

    pub fn stop(&mut self) {
        // Stop playback
    }
}
```

### 5. Tauri Commands (commands.rs)

```rust
use tauri::command;

#[command]
pub async fn play_midi(file_path: String) -> Result<(), String> {
    // Load MIDI file
    // Initialize audio engine
    // Start playback
    Ok(())
}

#[command]
pub async fn stop_playback() -> Result<(), String> {
    // Stop audio stream
    Ok(())
}

#[command]
pub async fn get_audio_devices() -> Result<Vec<String>, String> {
    // Enumerate audio devices
    Ok(vec![])
}
```

### 6. Frontend Integration

```typescript
// In useMusicBrain.ts or new hook
import { invoke } from '@tauri-apps/api/core';

export const playMidiFile = async (filePath: string) => {
  await invoke('play_midi', { filePath });
};

export const stopPlayback = async () => {
  await invoke('stop_playback');
};
```

## Testing Steps

1. **Compile Rust code**

   ```bash
   cd src-tauri
   cargo build
   ```

2. **Test audio device detection**
   - Verify devices are listed
   - Test default device selection

3. **Test MIDI playback**
   - Load a test MIDI file
   - Verify audio plays through system speakers
   - Check latency (<10ms target)

4. **Integration test**
   - Generate MIDI from emotion
   - Play through Rust audio engine
   - Verify synchronization

## Challenges

1. **MIDI to Audio Synthesis**
   - Need simple synthesizer (sine waves, basic waveforms)
   - Or use existing synth library

2. **Timing and Synchronization**
   - MIDI events need precise timing
   - Audio buffer management

3. **Cross-platform Compatibility**
   - CPAL should handle macOS/Windows/Linux
   - Test on all platforms

## Next Steps

1. Set up basic CPAL audio stream
2. Implement simple synthesizer (sine wave)
3. Parse MIDI file and convert to audio
4. Add Tauri commands
5. Integrate with frontend

## Status

**Pending** - Requires Rust development and audio synthesis implementation.
