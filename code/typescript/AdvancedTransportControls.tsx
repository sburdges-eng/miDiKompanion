import React, { useState, useEffect } from 'react';
import { PlaybackEngine, PlaybackState } from './PlaybackEngine';

interface AdvancedTransportControlsProps {
  engine: PlaybackEngine;
  tempo?: number;
  timeSignature?: [number, number];
  onStateChange?: (state: PlaybackState) => void;
}

export const AdvancedTransportControls: React.FC<AdvancedTransportControlsProps> = ({
  engine,
  tempo = 120,
  timeSignature = [4, 4],
  onStateChange,
}) => {
  const [state, setState] = useState<PlaybackState>(engine.getState());
  const [showSpeedControls, setShowSpeedControls] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

  useEffect(() => {
    const interval = setInterval(() => {
      const newState = engine.getState();
      setState(newState);
      setPlaybackSpeed(newState.playbackSpeed);
      onStateChange?.(newState);
    }, 100);
    return () => clearInterval(interval);
  }, [engine, onStateChange]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  const formatBars = (bars: number): string => {
    const wholeBars = Math.floor(bars);
    const beats = Math.floor((bars - wholeBars) * timeSignature[0]);
    return `${wholeBars}:${beats.toString().padStart(2, '0')}`;
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        padding: '15px',
        backgroundColor: '#1a1a1a',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      {/* Main Transport */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        {/* Return to Zero */}
        <button
          onClick={() => engine.returnToZero()}
          style={{
            padding: '8px 12px',
            backgroundColor: '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.85em',
          }}
          title="Return to Zero"
        >
          ⏮
        </button>

        {/* Return to Start Marker */}
        <button
          onClick={() => engine.returnToStartMarker()}
          style={{
            padding: '8px 12px',
            backgroundColor: '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.85em',
          }}
          title="Return to Start Marker"
        >
          ⏪
        </button>

        {/* Stop */}
        <button
          onClick={() => engine.stop()}
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            backgroundColor: '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          title="Stop"
        >
          ⏹
        </button>

        {/* Play/Pause */}
        <button
          onClick={() => engine.togglePlayPause()}
          style={{
            width: '50px',
            height: '50px',
            borderRadius: '50%',
            backgroundColor: state.isPlaying ? '#4caf50' : '#6366f1',
            border: 'none',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'bold',
          }}
          title={state.isPlaying ? 'Pause' : 'Play'}
        >
          {state.isPlaying ? '⏸' : '▶'}
        </button>

        {/* Record */}
        <button
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            backgroundColor: '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          title="Record"
        >
          ⏺
        </button>
      </div>

      {/* Playback Mode Buttons */}
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        <button
          onClick={() => engine.playFromCursor()}
          style={{
            padding: '6px 12px',
            backgroundColor: '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.8em',
          }}
          title="Play from Cursor"
        >
          ▶ Cursor
        </button>
        <button
          onClick={() => engine.playFromSelection()}
          style={{
            padding: '6px 12px',
            backgroundColor: '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.8em',
          }}
          title="Play from Selection"
        >
          ▶ Selection
        </button>
        <button
          onClick={() => engine.playSelectionOnly()}
          style={{
            padding: '6px 12px',
            backgroundColor: '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.8em',
          }}
          title="Play Selection Only"
        >
          🔁 Selection
        </button>
        <button
          onClick={() => engine.setLooping(!state.isLooping)}
          style={{
            padding: '6px 12px',
            backgroundColor: state.isLooping ? '#6366f1' : '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.8em',
          }}
          title="Toggle Loop"
        >
          🔁 Loop {state.isLooping ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* Speed Controls */}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={() => setShowSpeedControls(!showSpeedControls)}
          style={{
            padding: '6px 12px',
            backgroundColor: '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.8em',
          }}
        >
          ⚡ Speed: {playbackSpeed.toFixed(2)}x
        </button>

        {showSpeedControls && (
          <div style={{ display: 'flex', gap: '4px', alignItems: 'center' }}>
            <button
              onClick={() => engine.setHalfSpeed()}
              style={{
                padding: '4px 8px',
                backgroundColor: '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.75em',
              }}
              title="Half Speed"
            >
              0.5x
            </button>
            <button
              onClick={() => engine.setPlaybackSpeed(1.0)}
              style={{
                padding: '4px 8px',
                backgroundColor: '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.75em',
              }}
              title="Normal Speed"
            >
              1.0x
            </button>
            <button
              onClick={() => engine.setDoubleSpeed()}
              style={{
                padding: '4px 8px',
                backgroundColor: '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.75em',
              }}
              title="Double Speed"
            >
              2.0x
            </button>
            <button
              onClick={() => engine.setReverse(!state.isReversed)}
              style={{
                padding: '4px 8px',
                backgroundColor: state.isReversed ? '#f44336' : '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.75em',
              }}
              title="Reverse"
            >
              ⏪ Reverse
            </button>
            <input
              type="range"
              min="0.25"
              max="2.0"
              step="0.05"
              value={playbackSpeed}
              onChange={(e) => {
                const speed = parseFloat(e.target.value);
                engine.setPlaybackSpeed(speed);
              }}
              style={{ width: '100px' }}
            />
            <button
              onClick={() => engine.advanceFrame()}
              style={{
                padding: '4px 8px',
                backgroundColor: '#333',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.75em',
              }}
              title="Frame Advance"
            >
              ⏩ Frame
            </button>
          </div>
        )}
      </div>

      {/* Time Display */}
      <div style={{ display: 'flex', gap: '15px', alignItems: 'center', fontSize: '0.9em' }}>
        <div style={{ color: '#aaa' }}>
          Time: {formatTime(state.currentTime)}
        </div>
        <div style={{ color: '#aaa' }}>
          Position: {formatBars(state.playheadPosition)}
        </div>
        <div style={{ color: '#aaa' }}>
          Cursor: {formatBars(state.cursorPosition)}
        </div>
        {state.selectionStart !== null && state.selectionEnd !== null && (
          <div style={{ color: '#888', fontSize: '0.85em' }}>
            Selection: {formatBars(state.selectionStart)} - {formatBars(state.selectionEnd)}
          </div>
        )}
      </div>

      {/* Tempo and Time Signature */}
      <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
        <div style={{ color: '#aaa', fontSize: '0.85em' }}>
          BPM: {tempo}
        </div>
        <div style={{ color: '#aaa', fontSize: '0.85em' }}>
          {timeSignature[0]}/{timeSignature[1]}
        </div>
      </div>
    </div>
  );
};
