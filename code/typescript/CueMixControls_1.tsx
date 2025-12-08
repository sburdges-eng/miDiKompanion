import React, { useState } from 'react';
import { PlaybackEngine } from './PlaybackEngine';
import { AdvancedSlider } from './AdvancedSlider';

interface CueMixControlsProps {
  engine: PlaybackEngine;
  trackId: string;
  trackName: string;
  onStateChange?: () => void;
}

export const CueMixControls: React.FC<CueMixControlsProps> = ({
  engine,
  trackId,
  trackName,
  onStateChange,
}) => {
  const [cueSend, setCueSend] = useState(engine.getCueSend(trackId));
  const [listenBusMode, setListenBusMode] = useState<'AFL' | 'PFL' | 'off'>('off');

  const handleCueSendChange = (value: number) => {
    engine.setCueSend(trackId, value);
    setCueSend(value);
    onStateChange?.();
  };

  const handleListenBus = (mode: 'AFL' | 'PFL' | 'off') => {
    engine.setListenBus(mode, trackId);
    setListenBusMode(mode);
    onStateChange?.();
  };

  return (
    <div
      style={{
        padding: '10px',
        backgroundColor: '#2a2a2a',
        borderRadius: '4px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <div style={{ marginBottom: '8px', color: '#fff', fontSize: '0.9em', fontWeight: 'bold' }}>
        {trackName} - Cue Mix
      </div>

      {/* Cue Send Level */}
      <div style={{ marginBottom: '10px' }}>
        <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '4px' }}>
          Cue Send Level
        </label>
        <AdvancedSlider
          value={cueSend}
          min={0}
          max={1}
          step={0.01}
          orientation="horizontal"
          width={200}
          height={30}
          onChange={handleCueSendChange}
          showValue={true}
          unit="%"
          color="#6366f1"
        />
      </div>

      {/* Listen Bus Controls */}
      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          onClick={() => handleListenBus('AFL')}
          style={{
            flex: 1,
            padding: '6px 12px',
            backgroundColor: listenBusMode === 'AFL' ? '#6366f1' : '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.8em',
          }}
          title="After Fader Listen"
        >
          AFL
        </button>
        <button
          onClick={() => handleListenBus('PFL')}
          style={{
            flex: 1,
            padding: '6px 12px',
            backgroundColor: listenBusMode === 'PFL' ? '#6366f1' : '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.8em',
          }}
          title="Pre Fader Listen"
        >
          PFL
        </button>
        <button
          onClick={() => handleListenBus('off')}
          style={{
            flex: 1,
            padding: '6px 12px',
            backgroundColor: listenBusMode === 'off' ? '#666' : '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.8em',
          }}
          title="Off"
        >
          Off
        </button>
      </div>

      {/* Pre-listen Button */}
      <button
        onClick={() => engine.preListen(trackId, 2.0)}
        style={{
          width: '100%',
          marginTop: '8px',
          padding: '8px',
          backgroundColor: '#4caf50',
          border: 'none',
          borderRadius: '4px',
          color: '#fff',
          cursor: 'pointer',
          fontSize: '0.85em',
          fontWeight: 'bold',
        }}
        title="Pre-listen/Audition (2 seconds)"
      >
        🎧 Pre-listen
      </button>
    </div>
  );
};
