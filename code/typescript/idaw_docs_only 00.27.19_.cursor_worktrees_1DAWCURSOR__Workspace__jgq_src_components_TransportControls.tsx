import { useState } from 'react';

interface TransportControlsProps {
  tempo?: number;
  timeSignature?: [number, number];
  onPlay?: () => void;
  onPause?: () => void;
  onStop?: () => void;
  onRecord?: () => void;
  isPlaying?: boolean;
  isRecording?: boolean;
}

export const TransportControls: React.FC<TransportControlsProps> = ({
  tempo = 120,
  timeSignature = [4, 4],
  onPlay,
  onPause,
  onStop,
  onRecord,
  isPlaying = false,
  isRecording = false,
}) => {
  const [currentTempo, setCurrentTempo] = useState(tempo);
  const [isLooping, setIsLooping] = useState(false);

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '15px',
      padding: '12px 20px',
      backgroundColor: '#1a1a1a',
      borderTop: '1px solid rgba(255, 255, 255, 0.1)',
      borderRadius: '0 0 8px 8px'
    }}>
      {/* Transport buttons */}
      <div style={{ display: 'flex', gap: '8px' }}>
        <button
          onClick={onStop}
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
            justifyContent: 'center'
          }}
          title="Stop"
        >
          ⏹
        </button>

        <button
          onClick={isPlaying ? onPause : onPlay}
          style={{
            width: '50px',
            height: '50px',
            borderRadius: '50%',
            backgroundColor: isPlaying ? '#4caf50' : '#6366f1',
            border: 'none',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '20px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'bold'
          }}
          title={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>

        <button
          onClick={onRecord}
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            backgroundColor: isRecording ? '#f44336' : '#333',
            border: isRecording ? '2px solid #f44336' : '1px solid rgba(255, 255, 255, 0.2)',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          title="Record"
        >
          ⏺
        </button>
      </div>

      {/* Tempo display */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '6px 12px',
        backgroundColor: '#2a2a2a',
        borderRadius: '4px',
        border: '1px solid rgba(255, 255, 255, 0.1)'
      }}>
        <label style={{ fontSize: '0.85em', color: '#aaa' }}>BPM</label>
        <input
          type="number"
          value={currentTempo}
          onChange={(e) => setCurrentTempo(Number(e.target.value))}
          min="60"
          max="200"
          style={{
            width: '60px',
            padding: '4px 8px',
            backgroundColor: '#1a1a1a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            fontSize: '0.9em',
            textAlign: 'center'
          }}
        />
      </div>

      {/* Time signature */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '4px',
        padding: '6px 12px',
        backgroundColor: '#2a2a2a',
        borderRadius: '4px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        fontSize: '0.9em',
        color: '#fff'
      }}>
        <span>{timeSignature[0]}</span>
        <span>/</span>
        <span>{timeSignature[1]}</span>
      </div>

      {/* Loop toggle */}
      <button
        onClick={() => setIsLooping(!isLooping)}
        style={{
          padding: '6px 12px',
          backgroundColor: isLooping ? '#6366f1' : '#333',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '4px',
          color: '#fff',
          cursor: 'pointer',
          fontSize: '0.85em'
        }}
        title="Toggle Loop"
      >
        🔁 {isLooping ? 'ON' : 'OFF'}
      </button>

      {/* Time display */}
      <div style={{
        marginLeft: 'auto',
        padding: '6px 12px',
        backgroundColor: '#2a2a2a',
        borderRadius: '4px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        fontSize: '0.9em',
        color: '#fff',
        fontFamily: 'monospace'
      }}>
        00:00:00
      </div>
    </div>
  );
};
