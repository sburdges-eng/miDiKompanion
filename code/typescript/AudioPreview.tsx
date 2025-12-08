import React, { useState, useRef, useEffect } from 'react';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';

interface AudioPreviewProps {
  midiData: string | null;
  onError?: (error: string) => void;
}

export const AudioPreview: React.FC<AudioPreviewProps> = ({
  midiData,
  onError,
}) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [duration, setDuration] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const synthRef = useRef<Tone.PolySynth | null>(null);
  const scheduledNotesRef = useRef<number[]>([]);
  const animationFrameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      stopPlayback();
    };
  }, []);

  const stopPlayback = () => {
    if (synthRef.current) {
      synthRef.current.releaseAll();
    }
    scheduledNotesRef.current.forEach((id) => {
      Tone.Transport.clear(id);
    });
    scheduledNotesRef.current = [];
    Tone.Transport.stop();
    Tone.Transport.cancel();
    setIsPlaying(false);
    setCurrentTime(0);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    startTimeRef.current = null;
  };

  const updateTime = () => {
    if (startTimeRef.current && duration) {
      const elapsed = (Tone.now() - startTimeRef.current) * 1000; // Convert to ms
      setCurrentTime(Math.min(elapsed, duration));
      if (elapsed < duration) {
        animationFrameRef.current = requestAnimationFrame(updateTime);
      } else {
        setIsPlaying(false);
        setCurrentTime(0);
      }
    }
  };

  const playMidi = async () => {
    if (!midiData) {
      onError?.('No MIDI data available');
      return;
    }

    if (isPlaying) {
      stopPlayback();
      return;
    }

    try {
      setIsLoading(true);

      // Initialize Tone.js if not already started
      if (Tone.context.state !== 'running') {
        await Tone.start();
      }

      // Decode base64 MIDI data
      const binaryString = atob(midiData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Parse MIDI file
      const midi = new Midi(bytes);
      const totalDuration = midi.duration * 1000; // Convert to ms
      setDuration(totalDuration);

      // Create synthesizer
      if (!synthRef.current) {
        synthRef.current = new Tone.PolySynth(Tone.Synth, {
          oscillator: {
            type: 'sawtooth',
          },
          envelope: {
            attack: 0.02,
            decay: 0.1,
            sustain: 0.3,
            release: 0.3,
          },
        }).toDestination();
      }

      const synth = synthRef.current;

      // Schedule all notes
      midi.tracks.forEach((track: { notes: Array<{ time: number; duration: number; name: string; velocity?: number }> }) => {
        track.notes.forEach((note: { time: number; duration: number; name: string; velocity?: number }) => {
          const noteStartTime = note.time;
          const velocity = note.velocity || 0.8;

          const id = Tone.Transport.schedule((time: number) => {
            synth.triggerAttackRelease(
              note.name,
              note.duration,
              time,
              velocity
            );
          }, noteStartTime);

          scheduledNotesRef.current.push(id);
        });
      });

      // Start playback
      Tone.Transport.start();
      startTimeRef.current = Tone.now();
      setIsPlaying(true);
      setIsLoading(false);

      // Start time update loop
      updateTime();

      // Auto-stop when done
      Tone.Transport.schedule(() => {
        stopPlayback();
      }, midi.duration);
    } catch (error) {
      console.error('Error playing MIDI:', error);
      onError?.(`Failed to play MIDI: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setIsLoading(false);
      setIsPlaying(false);
    }
  };

  const formatTime = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  if (!midiData) {
    return null;
  }

  return (
    <div
      style={{
        marginTop: '15px',
        padding: '15px',
        backgroundColor: 'rgba(33, 150, 243, 0.1)',
        borderRadius: '4px',
        border: '1px solid #2196f3',
      }}
    >
      <div style={{ marginBottom: '10px' }}>
        <strong>🎵 Audio Preview</strong>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <button
          onClick={playMidi}
          disabled={isLoading}
          style={{
            padding: '10px 20px',
            backgroundColor: isPlaying ? '#f44336' : '#2196f3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            fontSize: '0.9em',
            fontWeight: 'bold',
            opacity: isLoading ? 0.6 : 1,
          }}
        >
          {isLoading
            ? 'Loading...'
            : isPlaying
            ? '⏹ Stop'
            : '▶ Play'}
        </button>
      </div>

      {duration !== null && (
        <div style={{ fontSize: '0.85em', color: '#666' }}>
          <div>
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
          <div
            style={{
              width: '100%',
              height: '4px',
              backgroundColor: '#ddd',
              borderRadius: '2px',
              marginTop: '5px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${duration ? (currentTime / duration) * 100 : 0}%`,
                height: '100%',
                backgroundColor: '#2196f3',
                transition: 'width 0.1s linear',
              }}
            />
          </div>
        </div>
      )}

      <div
        style={{
          marginTop: '10px',
          fontSize: '0.8em',
          color: '#666',
          fontStyle: 'italic',
        }}
      >
        Note: Preview uses basic synthesis. For best quality, download and open
        in your DAW.
      </div>
    </div>
  );
};
