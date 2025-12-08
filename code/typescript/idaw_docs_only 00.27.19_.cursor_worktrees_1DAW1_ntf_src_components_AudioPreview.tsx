import React, { useState, useEffect, useRef } from 'react';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';

interface AudioPreviewProps {
  midiData: string | null; // Base64 encoded MIDI data
  onPlayingChange?: (isPlaying: boolean) => void;
}

export const AudioPreview: React.FC<AudioPreviewProps> = ({ midiData, onPlayingChange }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const synthRef = useRef<Tone.PolySynth | null>(null);
  const midiRef = useRef<Midi | null>(null);
  const playbackRef = useRef<{ start: number; interval: NodeJS.Timeout | null } | null>(null);

  // Initialize Tone.js on component mount
  useEffect(() => {
    // Create a PolySynth for MIDI playback
    synthRef.current = new Tone.PolySynth(Tone.Synth).toDestination();

    return () => {
      // Cleanup on unmount
      if (synthRef.current) {
        synthRef.current.dispose();
      }
      if (playbackRef.current?.interval) {
        clearInterval(playbackRef.current.interval);
      }
    };
  }, []);

  // Load MIDI data when it changes
  useEffect(() => {
    if (!midiData) {
      setIsLoaded(false);
      setError(null);
      return;
    }

    try {
      // Decode base64 MIDI data
      const binaryString = atob(midiData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Parse MIDI file
      const midi = new Midi(bytes);
      midiRef.current = midi;

      // Calculate duration (in seconds)
      const totalDuration = midi.duration;
      setDuration(totalDuration);
      setIsLoaded(true);
      setError(null);
    } catch (err) {
      console.error('Error loading MIDI:', err);
      setError('Failed to load MIDI data');
      setIsLoaded(false);
    }
  }, [midiData]);

  // Update current time during playback
  useEffect(() => {
    if (isPlaying && playbackRef.current) {
      playbackRef.current.interval = setInterval(() => {
        if (playbackRef.current) {
          const elapsed = Tone.now() - playbackRef.current.start;
          setCurrentTime(Math.min(elapsed, duration));

          // Stop when duration reached
          if (elapsed >= duration) {
            handleStop();
          }
        }
      }, 100); // Update every 100ms
    } else {
      if (playbackRef.current?.interval) {
        clearInterval(playbackRef.current.interval);
        playbackRef.current.interval = null;
      }
    }

    return () => {
      if (playbackRef.current?.interval) {
        clearInterval(playbackRef.current.interval);
      }
    };
  }, [isPlaying, duration]);

  const handlePlay = async () => {
    if (!midiRef.current || !synthRef.current || !isLoaded) {
      setError('MIDI not loaded');
      return;
    }

    try {
      // Start Tone.js context if not already started
      if (Tone.context.state !== 'running') {
        await Tone.start();
      }

      const midi = midiRef.current;
      const synth = synthRef.current;
      const now = Tone.now();

      // Stop any existing playback
      synth.releaseAll();

      // Schedule all MIDI notes
      midi.tracks.forEach((track) => {
        track.notes.forEach((note) => {
          synth.triggerAttackRelease(
            note.name,
            note.duration,
            now + note.time,
            note.velocity / 127 // Normalize velocity to 0-1
          );
        });
      });

      // Track playback start time
      playbackRef.current = {
        start: Tone.now(),
        interval: null
      };

      setIsPlaying(true);
      setError(null);
      onPlayingChange?.(true);
    } catch (err) {
      console.error('Error playing MIDI:', err);
      setError('Failed to play MIDI');
      setIsPlaying(false);
      onPlayingChange?.(false);
    }
  };

  const handleStop = () => {
    if (synthRef.current) {
      synthRef.current.releaseAll();
    }
    if (playbackRef.current?.interval) {
      clearInterval(playbackRef.current.interval);
      playbackRef.current.interval = null;
    }
    setIsPlaying(false);
    setCurrentTime(0);
    onPlayingChange?.(false);
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!midiData) {
    return null;
  }

  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div style={{
      marginTop: '15px',
      padding: '15px',
      backgroundColor: 'rgba(0, 0, 0, 0.1)',
      borderRadius: '8px',
      border: '1px solid rgba(255, 255, 255, 0.1)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
        <button
          onClick={isPlaying ? handleStop : handlePlay}
          disabled={!isLoaded}
          style={{
            padding: '8px 16px',
            backgroundColor: isPlaying ? '#f44336' : '#4caf50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoaded ? 'pointer' : 'not-allowed',
            fontSize: '0.9em',
            fontWeight: 'bold',
            opacity: isLoaded ? 1 : 0.5
          }}
        >
          {isPlaying ? '⏹ Stop' : '▶ Play'}
        </button>

        <div style={{ flex: 1 }}>
          <div style={{
            width: '100%',
            height: '4px',
            backgroundColor: 'rgba(255, 255, 255, 0.2)',
            borderRadius: '2px',
            overflow: 'hidden'
          }}>
            <div style={{
              width: `${progressPercent}%`,
              height: '100%',
              backgroundColor: '#4caf50',
              transition: 'width 0.1s linear'
            }} />
          </div>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '0.75em',
            color: 'rgba(255, 255, 255, 0.6)',
            marginTop: '4px'
          }}>
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>
      </div>

      {error && (
        <div style={{
          padding: '8px',
          backgroundColor: 'rgba(244, 67, 54, 0.2)',
          color: '#f44336',
          borderRadius: '4px',
          fontSize: '0.85em',
          marginTop: '8px'
        }}>
          {error}
        </div>
      )}

      {!isLoaded && !error && (
        <div style={{
          padding: '8px',
          color: 'rgba(255, 255, 255, 0.6)',
          fontSize: '0.85em'
        }}>
          Loading MIDI...
        </div>
      )}
    </div>
  );
};
