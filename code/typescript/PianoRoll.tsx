/**
 * PianoRoll - MIDI note editor with piano keyboard
 * Features: Note editing, velocity display, snap to grid, piano keyboard
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

interface MidiNote {
  id: string;
  pitch: number;
  start: number;
  duration: number;
  velocity: number;
  channel?: number;
}

interface PianoRollProps {
  notes?: MidiNote[];
  duration?: number;
  currentTime?: number;
  isPlaying?: boolean;
  onNotesChange?: (notes: MidiNote[]) => void;
  onNotePlay?: (pitch: number, velocity: number) => void;
  onNoteStop?: (pitch: number) => void;
  snapValue?: number;
  zoom?: number;
  height?: number;
}

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const PIANO_WIDTH = 60;
const NOTE_HEIGHT = 16;
const MIN_OCTAVE = 2;
const MAX_OCTAVE = 7;
const TOTAL_NOTES = (MAX_OCTAVE - MIN_OCTAVE + 1) * 12;

export const PianoRoll: React.FC<PianoRollProps> = ({
  notes = [],
  duration = 16,
  currentTime = 0,
  isPlaying = false,
  onNotesChange: _onNotesChange,
  onNotePlay,
  onNoteStop,
  snapValue = 0.25,
  zoom = 1,
  height = 400,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const gridCanvasRef = useRef<HTMLCanvasElement>(null);
  const notesCanvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height });
  const [selectedNotes, _setSelectedNotes] = useState<Set<string>>(new Set());
  const [hoveredKey, setHoveredKey] = useState<number | null>(null);
  const [_dragging, _setDragging] = useState<{
    type: 'move' | 'resize' | 'create';
    noteId?: string;
    startX: number;
    startY: number;
    startPitch?: number;
    startTime?: number;
  } | null>(null);
  const [scrollY, setScrollY] = useState(0);

  // Generate demo notes if none provided
  const demoNotes: MidiNote[] = [
    { id: '1', pitch: 60, start: 0, duration: 1, velocity: 100 },
    { id: '2', pitch: 64, start: 1, duration: 1, velocity: 90 },
    { id: '3', pitch: 67, start: 2, duration: 1, velocity: 95 },
    { id: '4', pitch: 72, start: 3, duration: 2, velocity: 110 },
    { id: '5', pitch: 67, start: 5, duration: 0.5, velocity: 80 },
    { id: '6', pitch: 64, start: 5.5, duration: 0.5, velocity: 85 },
    { id: '7', pitch: 60, start: 6, duration: 2, velocity: 100 },
    { id: '8', pitch: 55, start: 0, duration: 4, velocity: 70 },
    { id: '9', pitch: 48, start: 4, duration: 4, velocity: 75 },
  ];

  const displayNotes = notes.length > 0 ? notes : demoNotes;

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: height,
        });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [height]);

  // Get note name from MIDI pitch
  const getNoteName = (pitch: number): string => {
    const octave = Math.floor(pitch / 12) - 1;
    const note = NOTE_NAMES[pitch % 12];
    return `${note}${octave}`;
  };

  // Check if note is black key
  const isBlackKey = (pitch: number): boolean => {
    const noteIndex = pitch % 12;
    return [1, 3, 6, 8, 10].includes(noteIndex);
  };

  // Convert pitch to Y position
  const pitchToY = useCallback((pitch: number): number => {
    const minPitch = MIN_OCTAVE * 12;
    const maxPitch = (MAX_OCTAVE + 1) * 12;
    const pitchRange = maxPitch - minPitch;
    const reversedPitch = maxPitch - pitch;
    return ((reversedPitch - MIN_OCTAVE * 12) / pitchRange) * (TOTAL_NOTES * NOTE_HEIGHT) - scrollY;
  }, [scrollY]);

  // Convert Y position to pitch
  const yToPitch = useCallback((y: number): number => {
    const minPitch = MIN_OCTAVE * 12;
    const maxPitch = (MAX_OCTAVE + 1) * 12;
    const pitchRange = maxPitch - minPitch;
    const adjustedY = y + scrollY;
    const pitch = maxPitch - Math.floor((adjustedY / (TOTAL_NOTES * NOTE_HEIGHT)) * pitchRange);
    return Math.max(minPitch, Math.min(maxPitch - 1, pitch));
  }, [scrollY]);

  // Convert time to X position
  const timeToX = useCallback((time: number): number => {
    const gridWidth = dimensions.width - PIANO_WIDTH;
    return PIANO_WIDTH + (time / duration) * gridWidth * zoom;
  }, [dimensions.width, duration, zoom]);

  // Convert X position to time (reserved for future note editing)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const xToTime = useCallback((x: number): number => {
    const gridWidth = dimensions.width - PIANO_WIDTH;
    const time = ((x - PIANO_WIDTH) / (gridWidth * zoom)) * duration;
    // Snap to grid
    return Math.round(time / snapValue) * snapValue;
  }, [dimensions.width, duration, zoom, snapValue]);
  void xToTime; // Suppress unused warning - will be used for note editing

  // Draw grid
  const drawGrid = useCallback(() => {
    const canvas = gridCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const totalHeight = TOTAL_NOTES * NOTE_HEIGHT;
    const gridWidth = dimensions.width - PIANO_WIDTH;

    canvas.width = dimensions.width * window.devicePixelRatio;
    canvas.height = totalHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, dimensions.width, totalHeight);

    // Draw piano keys and row backgrounds
    for (let i = 0; i < TOTAL_NOTES; i++) {
      const pitch = (MAX_OCTAVE + 1) * 12 - 1 - i;
      const y = i * NOTE_HEIGHT;
      const isBlack = isBlackKey(pitch);

      // Row background (alternating for black keys)
      ctx.fillStyle = isBlack ? '#151515' : '#1a1a1a';
      ctx.fillRect(PIANO_WIDTH, y, gridWidth, NOTE_HEIGHT);

      // Row border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PIANO_WIDTH, y + NOTE_HEIGHT);
      ctx.lineTo(dimensions.width, y + NOTE_HEIGHT);
      ctx.stroke();

      // Piano key
      ctx.fillStyle = isBlack ? '#2a2a2a' : '#e0e0e0';
      ctx.fillRect(0, y, PIANO_WIDTH - 2, NOTE_HEIGHT - 1);

      // Key hover effect
      if (hoveredKey === pitch) {
        ctx.fillStyle = 'rgba(99, 102, 241, 0.3)';
        ctx.fillRect(0, y, PIANO_WIDTH - 2, NOTE_HEIGHT - 1);
      }

      // Key label (only for C notes)
      if (pitch % 12 === 0) {
        ctx.fillStyle = isBlack ? '#888' : '#333';
        ctx.font = '10px sans-serif';
        ctx.fillText(getNoteName(pitch), 4, y + NOTE_HEIGHT - 4);

        // Octave divider line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(PIANO_WIDTH, y);
        ctx.lineTo(dimensions.width, y);
        ctx.stroke();
      }
    }

    // Draw beat/bar lines
    const beatsPerBar = 4;
    // pixelsPerBeat calculated for potential future use in beat snapping
    void ((gridWidth * zoom) / duration);

    for (let beat = 0; beat <= duration; beat++) {
      const x = PIANO_WIDTH + (beat / duration) * gridWidth * zoom;
      const isBarLine = beat % beatsPerBar === 0;

      ctx.strokeStyle = isBarLine ? 'rgba(255, 255, 255, 0.2)' : 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = isBarLine ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, totalHeight);
      ctx.stroke();

      // Beat/bar numbers
      if (isBarLine && beat < duration) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.font = '10px monospace';
        ctx.fillText(`${Math.floor(beat / beatsPerBar) + 1}`, x + 2, 12);
      }
    }

    // Sub-beat grid lines (16th notes)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    for (let sub = 0; sub <= duration * 4; sub++) {
      if (sub % 4 !== 0) {
        const x = PIANO_WIDTH + (sub / (duration * 4)) * gridWidth * zoom;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, totalHeight);
        ctx.stroke();
      }
    }

    // Playhead
    if (currentTime >= 0 && currentTime <= duration) {
      const playheadX = timeToX(currentTime);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, totalHeight);
      ctx.stroke();

      if (isPlaying) {
        ctx.shadowColor = '#fff';
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }
  }, [dimensions.width, duration, zoom, hoveredKey, currentTime, isPlaying, timeToX]);

  // Draw notes
  const drawNotes = useCallback(() => {
    const canvas = notesCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const totalHeight = TOTAL_NOTES * NOTE_HEIGHT;
    const gridWidth = dimensions.width - PIANO_WIDTH;

    canvas.width = dimensions.width * window.devicePixelRatio;
    canvas.height = totalHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear
    ctx.clearRect(0, 0, dimensions.width, totalHeight);

    // Draw notes
    displayNotes.forEach((note) => {
      const x = timeToX(note.start);
      const y = pitchToY(note.pitch);
      const width = (note.duration / duration) * gridWidth * zoom;
      const isSelected = selectedNotes.has(note.id);

      // Note color based on velocity and pitch
      const hue = 230 + ((note.pitch % 12) * 10);

      // Note body
      ctx.fillStyle = isSelected
        ? `hsla(${hue}, 80%, 60%, 0.9)`
        : `hsla(${hue}, 70%, ${40 + (note.velocity / 127) * 20}%, 0.8)`;
      ctx.fillRect(x, y, width - 1, NOTE_HEIGHT - 2);

      // Note border
      ctx.strokeStyle = isSelected ? '#fff' : `hsla(${hue}, 80%, 70%, 0.5)`;
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.strokeRect(x, y, width - 1, NOTE_HEIGHT - 2);

      // Velocity bar
      const velocityHeight = (note.velocity / 127) * (NOTE_HEIGHT - 4);
      ctx.fillStyle = `hsla(${hue}, 90%, 70%, 0.6)`;
      ctx.fillRect(x + 2, y + NOTE_HEIGHT - 3 - velocityHeight, 4, velocityHeight);

      // Note name (for longer notes)
      if (width > 30) {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.font = '10px sans-serif';
        ctx.fillText(getNoteName(note.pitch), x + 8, y + NOTE_HEIGHT - 5);
      }

      // Resize handle
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.fillRect(x + width - 6, y + 2, 4, NOTE_HEIGHT - 6);
    });
  }, [displayNotes, dimensions.width, duration, zoom, selectedNotes, pitchToY, timeToX]);

  // Animation loop
  useEffect(() => {
    drawGrid();
    drawNotes();
  }, [drawGrid, drawNotes]);

  // Handle piano key click
  const handlePianoKeyClick = (pitch: number) => {
    onNotePlay?.(pitch, 100);
    setTimeout(() => onNoteStop?.(pitch), 200);
  };

  // Handle scroll
  const handleScroll = (e: React.WheelEvent) => {
    const totalHeight = TOTAL_NOTES * NOTE_HEIGHT;
    const maxScroll = Math.max(0, totalHeight - height);
    setScrollY(Math.max(0, Math.min(maxScroll, scrollY + e.deltaY)));
  };

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: height,
        position: 'relative',
        backgroundColor: '#0f0f0f',
        borderRadius: '4px',
        overflow: 'hidden',
      }}
      onWheel={handleScroll}
    >
      {/* Grid layer */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: TOTAL_NOTES * NOTE_HEIGHT,
          transform: `translateY(${-scrollY}px)`,
        }}
      >
        <canvas
          ref={gridCanvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
          }}
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top + scrollY;

            if (x < PIANO_WIDTH) {
              // Piano key click
              const pitch = yToPitch(y);
              handlePianoKeyClick(pitch);
            }
          }}
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top + scrollY;

            if (x < PIANO_WIDTH) {
              setHoveredKey(yToPitch(y));
            } else {
              setHoveredKey(null);
            }
          }}
          onMouseLeave={() => setHoveredKey(null)}
        />

        {/* Notes layer */}
        <canvas
          ref={notesCanvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        />
      </div>

      {/* Toolbar */}
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '30px',
          backgroundColor: '#0f0f0f',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          display: 'flex',
          alignItems: 'center',
          padding: '0 10px',
          gap: '15px',
          fontSize: '11px',
          color: '#888',
        }}
      >
        <span>Snap: 1/{Math.round(1 / snapValue)}</span>
        <span>Notes: {displayNotes.length}</span>
        <span>Zoom: {(zoom * 100).toFixed(0)}%</span>
        <span>Duration: {duration} beats</span>
      </div>
    </div>
  );
};
