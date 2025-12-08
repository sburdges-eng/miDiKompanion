// TimeFormatControls - Features 74-80: Time Format Display

import React, { useState, useEffect } from 'react';
import { TransportEngine, TimeFormat } from './TransportEngine';

interface TimeFormatControlsProps {
  engine: TransportEngine;
  onFormatChange?: () => void;
}

export const TimeFormatControls: React.FC<TimeFormatControlsProps> = ({
  engine,
  onFormatChange,
}) => {
  const [state, setState] = useState(engine.getState());

  useEffect(() => {
    const interval = setInterval(() => {
      setState(engine.getState());
    }, 100);
    return () => clearInterval(interval);
  }, [engine]);

  const timeFormats: { value: TimeFormat; label: string; description: string }[] = [
    { value: 'bars-beats', label: 'Bars:Beats', description: 'Feature 74: Bars:Beats display' },
    { value: 'time', label: 'Time Code', description: 'Feature 75: Time display (HH:MM:SS:FF)' },
    { value: 'samples', label: 'Samples', description: 'Feature 76: Samples display' },
    { value: 'feet-frames', label: 'Feet+Frames', description: 'Feature 77: Feet+Frames display' },
    { value: 'seconds', label: 'Seconds', description: 'Feature 78: Seconds display' },
    { value: 'minutes-seconds', label: 'Min:Sec', description: 'Feature 79: Minutes:Seconds' },
    { value: 'custom', label: 'Custom', description: 'Feature 80: Custom time format' },
  ];

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
      <h3 style={{ color: '#fff', margin: '0 0 15px 0' }}>Time Formats (Features 74-80)</h3>

      {/* Primary Time Format */}
      <div>
        <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
          Primary Time Format (Feature 74)
        </label>
        <select
          value={state.primaryTimeFormat}
          onChange={(e) => {
            engine.setPrimaryTimeFormat(e.target.value as TimeFormat);
            onFormatChange?.();
          }}
          style={{
            width: '100%',
            padding: '10px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            fontSize: '1em',
          }}
        >
          {timeFormats.map((format) => (
            <option key={format.value} value={format.value}>
              {format.label} - {format.description}
            </option>
          ))}
        </select>
      </div>

      {/* Secondary Time Format */}
      <div>
        <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
          Secondary Time Format (Feature 78)
        </label>
        <select
          value={state.secondaryTimeFormat || 'none'}
          onChange={(e) => {
            const val = e.target.value === 'none' ? null : (e.target.value as TimeFormat);
            engine.setSecondaryTimeFormat(val);
            onFormatChange?.();
          }}
          style={{
            width: '100%',
            padding: '10px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            fontSize: '1em',
          }}
        >
          <option value="none">None</option>
          {timeFormats.map((format) => (
            <option key={format.value} value={format.value}>
              {format.label}
            </option>
          ))}
        </select>
      </div>

      {/* Display Format Preview */}
      <div
        style={{
          padding: '20px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          textAlign: 'center',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Current Position Display
        </div>
        <div
          style={{
            color: '#fff',
            fontSize: '2em',
            fontFamily: 'monospace',
            fontWeight: 'bold',
            marginBottom: '10px',
          }}
        >
          {engine.getTimeFormatString(state.position, state.displayFormat)}
        </div>
        {state.secondaryTimeFormat && (
          <>
            <div style={{ color: '#666', fontSize: '0.85em', marginBottom: '5px' }}>Secondary:</div>
            <div
              style={{
                color: '#888',
                fontSize: '1.2em',
                fontFamily: 'monospace',
              }}
            >
              {engine.getTimeFormatString(state.position, state.secondaryTimeFormat)}
            </div>
          </>
        )}
      </div>

      {/* Format Descriptions */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          fontSize: '0.85em',
        }}
      >
        <div style={{ color: '#aaa', marginBottom: '10px', fontWeight: 'bold' }}>
          Available Formats:
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {timeFormats.map((format) => (
            <div
              key={format.value}
              style={{
                padding: '8px',
                backgroundColor: state.displayFormat === format.value ? '#6366f120' : 'transparent',
                borderRadius: '4px',
                border: state.displayFormat === format.value ? '1px solid #6366f1' : 'none',
              }}
            >
              <div style={{ color: '#fff', fontWeight: 'bold' }}>{format.label}</div>
              <div style={{ color: '#888', fontSize: '0.9em' }}>{format.description}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Custom Format Input (Feature 80) */}
      {state.displayFormat === 'custom' && (
        <div>
          <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
            Custom Format String (Feature 80)
          </label>
          <input
            type="text"
            placeholder="e.g., %B:%b:%t"
            style={{
              width: '100%',
              padding: '10px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '1em',
            }}
            onBlur={(e) => {
              if (e.target.value) {
                engine.setCustomTimeFormat(e.target.value);
                onFormatChange?.();
              }
            }}
          />
          <div style={{ color: '#888', fontSize: '0.8em', marginTop: '5px' }}>
            Format codes: %B=bars, %b=beats, %t=ticks, %s=seconds, %S=samples
          </div>
        </div>
      )}

      {/* Sample Rate Info */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          padding: '10px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          fontSize: '0.85em',
          color: '#888',
        }}
      >
        <div>Sample Rate: {state.sampleRate / 1000}kHz</div>
        <div>Ticks/Quarter: {state.ticksPerQuarter}</div>
      </div>
    </div>
  );
};
