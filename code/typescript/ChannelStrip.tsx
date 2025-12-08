import { useState } from 'react';
import { VUMeter } from './VUMeter';

interface ChannelStripProps {
  channel: number;
  name: string;
  volume?: number;
  pan?: number;
  muted?: boolean;
  solo?: boolean;
  onVolumeChange?: (volume: number) => void;
  onPanChange?: (pan: number) => void;
  onMute?: () => void;
  onSolo?: () => void;
}

export const ChannelStrip: React.FC<ChannelStripProps> = ({
  channel,
  name,
  volume = 0.75,
  pan = 0,
  muted = false,
  solo = false,
  onVolumeChange,
  onPanChange,
  onMute,
  onSolo,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [vuLevel, setVuLevel] = useState(0.3 + Math.random() * 0.4);

  const handleVolumeDrag = (e: React.MouseEvent) => {
    if (e.buttons !== 1) return;
    setIsDragging(true);
    const rect = e.currentTarget.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const newVolume = 1 - (y / rect.height);
    onVolumeChange?.(Math.max(0, Math.min(1, newVolume)));
  };

  const handlePanDrag = (e: React.MouseEvent) => {
    if (e.buttons !== 1) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const newPan = (x / rect.width) * 2 - 1; // -1 to 1
    onPanChange?.(Math.max(-1, Math.min(1, newPan)));
  };

  // Simulate VU meter animation
  useState(() => {
    const interval = setInterval(() => {
      setVuLevel(0.2 + Math.random() * 0.6);
    }, 100);
    return () => clearInterval(interval);
  });

  const volumePercent = Math.round(volume * 100);
  const panPercent = Math.round((pan + 1) * 50);

  return (
    <div style={{
      width: '60px',
      height: '100%',
      backgroundColor: '#1a1a1a',
      borderRight: '1px solid rgba(255, 255, 255, 0.1)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '8px 4px',
      gap: '8px'
    }}>
      {/* Channel name */}
      <div style={{
        fontSize: '0.75em',
        color: '#fff',
        textAlign: 'center',
        fontWeight: 'bold',
        width: '100%',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap'
      }}>
        {name}
      </div>

      {/* VU Meter */}
      <VUMeter level={vuLevel} label={channel.toString()} />

      {/* Volume fader */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '4px',
        width: '100%'
      }}>
        <div
          style={{
            width: '20px',
            height: '200px',
            backgroundColor: '#0f0f0f',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '2px',
            position: 'relative',
            cursor: 'pointer'
          }}
          onMouseDown={handleVolumeDrag}
          onMouseMove={isDragging ? handleVolumeDrag : undefined}
          onMouseUp={() => setIsDragging(false)}
          onMouseLeave={() => setIsDragging(false)}
        >
          {/* Fader position */}
          <div style={{
            position: 'absolute',
            bottom: `${volume * 100}%`,
            left: '50%',
            transform: 'translateX(-50%)',
            width: '24px',
            height: '8px',
            backgroundColor: '#6366f1',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '2px',
            cursor: 'grab',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)'
          }} />

          {/* Volume scale */}
          <div style={{
            position: 'absolute',
            right: '-20px',
            top: 0,
            bottom: 0,
            width: '16px',
            fontSize: '0.65em',
            color: '#888',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            padding: '4px 0'
          }}>
            <span>0</span>
            <span>-6</span>
            <span>-12</span>
            <span>-18</span>
            <span>-∞</span>
          </div>
        </div>

        {/* Volume value */}
        <div style={{
          fontSize: '0.7em',
          color: '#888',
          minHeight: '16px'
        }}>
          {volumePercent}%
        </div>
      </div>

      {/* Pan control */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '4px',
        width: '100%'
      }}>
        <div
          style={{
            width: '100%',
            height: '40px',
            backgroundColor: '#0f0f0f',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            position: 'relative',
            cursor: 'pointer'
          }}
          onMouseDown={handlePanDrag}
          onMouseMove={isDragging ? handlePanDrag : undefined}
          onMouseUp={() => setIsDragging(false)}
        >
          {/* Center line */}
          <div style={{
            position: 'absolute',
            left: '50%',
            top: 0,
            bottom: 0,
            width: '1px',
            backgroundColor: 'rgba(255, 255, 255, 0.3)'
          }} />

          {/* Pan position */}
          <div style={{
            position: 'absolute',
            left: `${panPercent}%`,
            top: '50%',
            transform: 'translate(-50%, -50%)',
            width: '8px',
            height: '8px',
            backgroundColor: '#6366f1',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '50%',
            cursor: 'grab'
          }} />
        </div>
        <div style={{
          fontSize: '0.65em',
          color: '#888',
          display: 'flex',
          justifyContent: 'space-between',
          width: '100%',
          padding: '0 4px'
        }}>
          <span>L</span>
          <span>C</span>
          <span>R</span>
        </div>
      </div>

      {/* Mute/Solo buttons */}
      <div style={{
        display: 'flex',
        gap: '4px',
        width: '100%'
      }}>
        <button
          onClick={onMute}
          style={{
            flex: 1,
            padding: '4px',
            backgroundColor: muted ? '#f44336' : '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.7em',
            fontWeight: 'bold'
          }}
          title="Mute"
        >
          M
        </button>
        <button
          onClick={onSolo}
          style={{
            flex: 1,
            padding: '4px',
            backgroundColor: solo ? '#ffeb3b' : '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: solo ? '#000' : '#fff',
            cursor: 'pointer',
            fontSize: '0.7em',
            fontWeight: 'bold'
          }}
          title="Solo"
        >
          S
        </button>
      </div>
    </div>
  );
};
