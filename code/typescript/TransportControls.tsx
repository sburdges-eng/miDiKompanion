// TransportControls - Features 58-73: Advanced Transport Controls

import React, { useState, useEffect } from 'react';
import { TransportEngine, TimePosition } from './TransportEngine';

interface TransportControlsProps {
  engine: TransportEngine;
  onStateChange?: () => void;
}

export const TransportControls: React.FC<TransportControlsProps> = ({
  engine,
  onStateChange,
}) => {
  const [state, setState] = useState(engine.getState());
  const [shuttleValue, setShuttleValue] = useState(0);
  const [speedValue, setSpeedValue] = useState(1.0);

  useEffect(() => {
    const interval = setInterval(() => {
      setState(engine.getState());
      onStateChange?.();
    }, 100);
    return () => clearInterval(interval);
  }, [engine, onStateChange]);

  const formatPosition = (pos: TimePosition): string => {
    return engine.getTimeFormatString(pos, state.displayFormat);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '15px',
        padding: '20px',
        backgroundColor: '#1a1a1a',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <h3 style={{ color: '#fff', margin: '0 0 15px 0' }}>Transport Controls (Features 58-73)</h3>

      {/* Main Transport Buttons */}
      <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
        {/* Feature 58: Play */}
        <button
          onClick={() => engine.play()}
          disabled={state.isPlaying}
          style={{
            padding: '12px 20px',
            backgroundColor: state.isPlaying ? '#4caf50' : '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: state.isPlaying ? 'not-allowed' : 'pointer',
            fontSize: '1em',
            fontWeight: 'bold',
          }}
        >
          ▶ Play
        </button>

        {/* Feature 59: Stop */}
        <button
          onClick={() => engine.stop()}
          style={{
            padding: '12px 20px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '1em',
            fontWeight: 'bold',
          }}
        >
          ⏹ Stop
        </button>

        {/* Feature 60: Record */}
        <button
          onClick={() => engine.record()}
          style={{
            padding: '12px 20px',
            backgroundColor: state.isRecording ? '#f44336' : '#2a2a2a',
            border: `1px solid ${state.isRecording ? '#f44336' : 'rgba(255, 255, 255, 0.2)'}`,
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '1em',
            fontWeight: 'bold',
          }}
        >
          ⏺ Record
        </button>

        {/* Feature 61: Return to Zero */}
        <button
          onClick={() => engine.returnToZero()}
          style={{
            padding: '12px 20px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.9em',
          }}
        >
          ⏮ RZ
        </button>

        {/* Feature 62: Return to Start Marker */}
        <button
          onClick={() => engine.returnToStartMarker()}
          style={{
            padding: '12px 20px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.9em',
          }}
        >
          ⏮ Start
        </button>
      </div>

      {/* Position Display */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '5px' }}>Position</div>
        <div style={{ color: '#fff', fontSize: '1.5em', fontFamily: 'monospace', fontWeight: 'bold' }}>
          {formatPosition(state.position)}
        </div>
        {state.secondaryTimeFormat && (
          <div style={{ color: '#888', fontSize: '0.9em', marginTop: '5px' }}>
            {engine.getTimeFormatString(state.position, state.secondaryTimeFormat)}
          </div>
        )}
      </div>

      {/* Speed Controls */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
        {/* Feature 66: Shuttle */}
        <div>
          <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
            Shuttle Speed (Feature 66)
          </label>
          <input
            type="range"
            min="-1"
            max="1"
            step="0.1"
            value={shuttleValue}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              setShuttleValue(val);
              engine.setShuttleSpeed(val);
            }}
            style={{ width: '100%' }}
          />
          <div style={{ color: '#888', fontSize: '0.8em', textAlign: 'center', marginTop: '5px' }}>
            {shuttleValue.toFixed(1)}x
          </div>
        </div>

        {/* Feature 68: Variable Speed */}
        <div>
          <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
            Variable Speed (Feature 68)
          </label>
          <input
            type="range"
            min="0.25"
            max="4"
            step="0.05"
            value={speedValue}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              setSpeedValue(val);
              engine.setVariableSpeed(val);
            }}
            style={{ width: '100%' }}
          />
          <div style={{ color: '#888', fontSize: '0.8em', textAlign: 'center', marginTop: '5px' }}>
            {speedValue.toFixed(2)}x
          </div>
        </div>
      </div>

      {/* Speed Presets */}
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
        {/* Feature 69: Half Speed */}
        <button
          onClick={() => {
            engine.setHalfSpeed(!state.halfSpeed);
            if (!state.halfSpeed) setSpeedValue(0.5);
          }}
          style={{
            padding: '8px 16px',
            backgroundColor: state.halfSpeed ? '#6366f1' : '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.9em',
          }}
        >
          ½ Speed
        </button>

        {/* Feature 70: Double Speed */}
        <button
          onClick={() => {
            engine.setDoubleSpeed(!state.doubleSpeed);
            if (!state.doubleSpeed) setSpeedValue(2.0);
          }}
          style={{
            padding: '8px 16px',
            backgroundColor: state.doubleSpeed ? '#6366f1' : '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.9em',
          }}
        >
          2x Speed
        </button>

        {/* Feature 71: Reverse */}
        <button
          onClick={() => engine.setReverse(!state.reverse)}
          style={{
            padding: '8px 16px',
            backgroundColor: state.reverse ? '#6366f1' : '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.9em',
          }}
        >
          ↻ Reverse
        </button>

        {/* Feature 72: Frame-by-Frame */}
        <button
          onClick={() => engine.setFrameByFrame(!state.frameByFrame)}
          style={{
            padding: '8px 16px',
            backgroundColor: state.frameByFrame ? '#6366f1' : '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.9em',
          }}
        >
          Frame-by-Frame
        </button>
        {state.frameByFrame && (
          <button
            onClick={() => engine.advanceFrame()}
            style={{
              padding: '8px 16px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.9em',
            }}
          >
            Advance Frame
          </button>
        )}
      </div>

      {/* Loop Controls */}
      <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
        {/* Feature 65: Loop Toggle */}
        <button
          onClick={() => engine.toggleLoop()}
          style={{
            padding: '8px 16px',
            backgroundColor: state.cycleEnabled ? '#6366f1' : '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.9em',
          }}
        >
          🔁 Loop
        </button>
        {state.cycleEnabled && state.loopStart && state.loopEnd && (
          <div style={{ color: '#aaa', fontSize: '0.85em' }}>
            Loop: {formatPosition(state.loopStart)} - {formatPosition(state.loopEnd)}
          </div>
        )}
      </div>

      {/* Feature 67: Scrub */}
      <div>
        <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
          Scrub (Feature 67)
        </label>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            onClick={() => engine.enableScrub(!state.scrubEnabled)}
            style={{
              padding: '8px 16px',
              backgroundColor: state.scrubEnabled ? '#6366f1' : '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.9em',
            }}
          >
            {state.scrubEnabled ? 'Disable Scrub' : 'Enable Scrub'}
          </button>
          {state.scrubEnabled && (
            <div style={{ color: '#888', fontSize: '0.85em' }}>
              Drag timeline to scrub
            </div>
          )}
        </div>
      </div>

      {/* Feature 73: Sync Source */}
      <div>
        <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
          Sync Source (Feature 73)
        </label>
        <select
          value={state.syncSource}
          onChange={(e) => engine.setSyncSource(e.target.value as any)}
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            fontSize: '0.9em',
          }}
        >
          <option value="internal">Internal</option>
          <option value="midi-clock">MIDI Clock</option>
          <option value="word-clock">Word Clock</option>
          <option value="ltc">LTC (Linear Time Code)</option>
          <option value="video">Video</option>
          <option value="none">None</option>
        </select>
        {state.syncEnabled && (
          <div style={{ color: '#4caf50', fontSize: '0.85em', marginTop: '5px' }}>
            ✓ Sync Active
          </div>
        )}
      </div>

      {/* Status Indicators */}
      <div
        style={{
          display: 'flex',
          gap: '10px',
          padding: '10px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          fontSize: '0.85em',
        }}
      >
        <div style={{ color: state.isPlaying ? '#4caf50' : '#888' }}>
          {state.isPlaying ? '▶ Playing' : '⏸ Stopped'}
        </div>
        <div style={{ color: state.isRecording ? '#f44336' : '#888' }}>
          {state.isRecording ? '⏺ Recording' : '○ Not Recording'}
        </div>
        <div style={{ color: state.cycleEnabled ? '#6366f1' : '#888' }}>
          {state.cycleEnabled ? '🔁 Looping' : '○ No Loop'}
        </div>
      </div>
    </div>
  );
};
