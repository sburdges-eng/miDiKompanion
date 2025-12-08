import React, { useState } from 'react';
import { PlaybackEngine } from './PlaybackEngine';

interface SoloMuteControlsProps {
  engine: PlaybackEngine;
  trackId: string;
  trackName: string;
  onStateChange?: () => void;
}

export const SoloMuteControls: React.FC<SoloMuteControlsProps> = ({
  engine,
  trackId,
  trackName,
  onStateChange,
}) => {
  const [isSoloed, setIsSoloed] = useState(engine.isTrackSoloed(trackId));
  const [isMuted, setIsMuted] = useState(engine.isTrackMuted(trackId));
  const [isSoloSafe, setIsSoloSafe] = useState(engine.isTrackSoloSafe(trackId));
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSolo = (mode: 'normal' | 'exclusive' | 'xor') => {
    switch (mode) {
      case 'normal':
        engine.soloInPlace(trackId, !isSoloed);
        setIsSoloed(!isSoloed);
        break;
      case 'exclusive':
        engine.setExclusiveSolo(trackId);
        setIsSoloed(true);
        break;
      case 'xor':
        engine.setXORSolo(trackId);
        setIsSoloed(!isSoloed);
        break;
    }
    onStateChange?.();
  };

  const handleMute = () => {
    engine.mute(trackId, !isMuted);
    setIsMuted(!isMuted);
    onStateChange?.();
  };

  const handleSoloSafe = () => {
    engine.setSoloSafe(trackId, !isSoloSafe);
    setIsSoloSafe(!isSoloSafe);
    onStateChange?.();
  };

  const handleSoloDefeat = () => {
    engine.soloDefeat();
    setIsSoloed(false);
    onStateChange?.();
  };

  return (
    <div
      style={{
        display: 'flex',
        gap: '8px',
        alignItems: 'center',
        padding: '8px',
        backgroundColor: '#2a2a2a',
        borderRadius: '4px',
      }}
    >
      {/* Mute Button */}
      <button
        onClick={handleMute}
        style={{
          padding: '6px 12px',
          backgroundColor: isMuted ? '#f44336' : '#333',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '4px',
          color: '#fff',
          cursor: 'pointer',
          fontSize: '0.85em',
          fontWeight: 'bold',
        }}
        title="Mute"
      >
        M
      </button>

      {/* Solo Button */}
      <button
        onClick={() => handleSolo('normal')}
        style={{
          padding: '6px 12px',
          backgroundColor: isSoloed ? '#ffeb3b' : '#333',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '4px',
          color: isSoloed ? '#000' : '#fff',
          cursor: 'pointer',
          fontSize: '0.85em',
          fontWeight: 'bold',
        }}
        title="Solo"
      >
        S
      </button>

      {/* Advanced Solo Options */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        style={{
          padding: '6px 8px',
          backgroundColor: '#333',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '4px',
          color: '#fff',
          cursor: 'pointer',
          fontSize: '0.75em',
        }}
        title="Advanced Solo Options"
      >
        ⚙
      </button>

      {showAdvanced && (
        <div
          style={{
            position: 'absolute',
            marginTop: '40px',
            padding: '10px',
            backgroundColor: '#1a1a1a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
            zIndex: 1000,
          }}
        >
          <button
            onClick={() => {
              handleSolo('exclusive');
              setShowAdvanced(false);
            }}
            style={{
              padding: '6px 12px',
              backgroundColor: '#333',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.8em',
              textAlign: 'left',
            }}
            title="Exclusive Solo (mute all others)"
          >
            Exclusive Solo
          </button>
          <button
            onClick={() => {
              handleSolo('xor');
              setShowAdvanced(false);
            }}
            style={{
              padding: '6px 12px',
              backgroundColor: '#333',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.8em',
              textAlign: 'left',
            }}
            title="X-OR Solo (toggle)"
          >
            X-OR Solo
          </button>
          <button
            onClick={() => {
              handleSoloSafe();
              setShowAdvanced(false);
            }}
            style={{
              padding: '6px 12px',
              backgroundColor: isSoloSafe ? '#6366f1' : '#333',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.8em',
              textAlign: 'left',
            }}
            title="Solo Safe (immune to solo)"
          >
            Solo Safe {isSoloSafe ? '✓' : ''}
          </button>
          <button
            onClick={() => {
              handleSoloDefeat();
              setShowAdvanced(false);
            }}
            style={{
              padding: '6px 12px',
              backgroundColor: '#333',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.8em',
              textAlign: 'left',
            }}
            title="Solo Defeat (clear all solos)"
          >
            Solo Defeat
          </button>
        </div>
      )}

      {/* Track Name */}
      <span style={{ color: '#aaa', fontSize: '0.85em', marginLeft: '8px' }}>
        {trackName}
      </span>
    </div>
  );
};
