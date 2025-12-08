import { useState, useEffect, useRef } from 'react';
import * as Tone from 'tone';
import { Midi } from '@tonejs/midi';

interface AudioPreviewProps {
  midiData: string | null;
}

export const AudioPreview: React.FC<AudioPreviewProps> = ({ midiData }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const synthRef = useRef<Tone.PolySynth | null>(null);
  const scheduledNotesRef = useRef<Array<{ time: number; note: any }>>([]);
  const startTimeRef = useRef<number | null>(null);

  useEffect(() => {
    // Initialize Tone.js synth
    if (!synthRef.current) {
      synthRef.current = new Tone.PolySynth(Tone.Synth).toDestination();
    }

    // Cleanup on unmount
    return () => {
      if (synthRef.current) {
        synthRef.current.dispose();
      }
    };
  }, []);

  const playMidi = async () => {
    if (!midiData || isPlaying) return;

    try {
      setIsLoading(true);

      // Decode base64 MIDI data
      const binaryString = atob(midiData);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      // Parse MIDI file
      const midi = new Midi(bytes);

      // Calculate total duration
      let maxTime = 0;
      midi.tracks.forEach(track => {
        track.notes.forEach(note => {
          maxTime = Math.max(maxTime, note.time + note.duration);
        });
      });
      setDuration(maxTime);

      // Start Tone.js context if not already started
      if (Tone.context.state !== 'running') {
        await Tone.start();
      }

      // Clear any previously scheduled notes
      scheduledNotesRef.current = [];

      // Schedule all notes
      const now = Tone.now();
      startTimeRef.current = now;

      midi.tracks.forEach(track => {
        track.notes.forEach(note => {
          const noteTime = now + note.time;
          const noteOffTime = noteTime + note.duration;

          // Schedule note on
          synthRef.current!.triggerAttackRelease(
            note.name,
            note.duration,
            noteTime,
            note.velocity / 127
          );
        });
      });

      setIsPlaying(true);
      setIsLoading(false);

      // Update progress
      const updateProgress = () => {
        if (startTimeRef.current !== null) {
          const elapsed = Tone.now() - startTimeRef.current;
          setProgress(Math.min(elapsed / maxTime, 1));

          if (elapsed < maxTime && isPlaying) {
            requestAnimationFrame(updateProgress);
          } else {
            setIsPlaying(false);
            setProgress(0);
          }
        }
      };
      updateProgress();

    } catch (error) {
      console.error('Error playing MIDI:', error);
      setIsLoading(false);
      setIsPlaying(false);
    }
  };

  const stopMidi = () => {
    if (synthRef.current) {
      synthRef.current.releaseAll();
    }
    setIsPlaying(false);
    setProgress(0);
    startTimeRef.current = null;
  };

  if (!midiData) {
    return null;
  }

  return (
    <div style={{
      marginTop: '15px',
      padding: '15px',
      backgroundColor: 'rgba(0, 0, 0, 0.05)',
      borderRadius: '8px',
      border: '1px solid rgba(0, 0, 0, 0.1)'
    }}>
      <h4 style={{ margin: '0 0 10px 0', fontSize: '1em', fontWeight: 'bold' }}>
        🎵 Audio Preview
      </h4>

      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '10px' }}>
        <button
          onClick={isPlaying ? stopMidi : playMidi}
          disabled={isLoading}
          style={{
            padding: '8px 16px',
            backgroundColor: isPlaying ? '#f44336' : '#4caf50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '0.9em',
            fontWeight: 'bold',
            minWidth: '80px'
          }}
        >
          {isLoading ? 'Loading...' : isPlaying ? '⏹ Stop' : '▶ Play'}
        </button>

        {duration > 0 && (
          <span style={{ fontSize: '0.85em', color: '#666' }}>
            {Math.floor(progress * duration)}s / {Math.floor(duration)}s
          </span>
        )}
      </div>

      {duration > 0 && (
        <div style={{
          width: '100%',
          height: '6px',
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
          borderRadius: '3px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${progress * 100}%`,
            height: '100%',
            backgroundColor: '#4caf50',
            transition: 'width 0.1s linear'
          }} />
        </div>
      )}
    </div>
  );
};
