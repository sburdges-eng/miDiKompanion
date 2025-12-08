import React, { useState } from 'react';
import { Track } from './RecordingEngine';

interface TakeLanesProps {
  track: Track;
  onTakeSelect: (takeId: string) => void;
  onTakeDelete: (takeId: string) => void;
  onCompCreate?: (regions: Array<{ start: number; end: number; takeId: string }>) => void;
}

export const TakeLanes: React.FC<TakeLanesProps> = ({
  track,
  onTakeSelect,
  onTakeDelete,
  onCompCreate,
}) => {
  const [selectedTakes, setSelectedTakes] = useState<Set<string>>(new Set());
  const [compRegions, setCompRegions] = useState<Array<{ start: number; end: number; takeId: string }>>([]);

  const toggleTakeSelection = (takeId: string) => {
    const newSelected = new Set(selectedTakes);
    if (newSelected.has(takeId)) {
      newSelected.delete(takeId);
    } else {
      newSelected.add(takeId);
    }
    setSelectedTakes(newSelected);
  };

  const createComp = () => {
    if (compRegions.length > 0 && onCompCreate) {
      onCompCreate(compRegions);
      setCompRegions([]);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  return (
    <div
      style={{
        padding: '15px',
        backgroundColor: '#1a1a1a',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h3 style={{ margin: 0, color: '#fff' }}>Take Lanes: {track.name}</h3>
        {selectedTakes.size > 0 && (
          <button
            onClick={createComp}
            style={{
              padding: '6px 12px',
              backgroundColor: '#6366f1',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9em',
            }}
          >
            Create Comp ({selectedTakes.size} takes)
          </button>
        )}
      </div>

      {track.takes.length === 0 ? (
        <div style={{ padding: '20px', textAlign: 'center', color: '#888' }}>
          No takes recorded yet
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {track.takes.map((take, index) => (
            <div
              key={take.id}
              style={{
                padding: '12px',
                backgroundColor: take.selected ? '#6366f120' : '#2a2a2a',
                border: `2px solid ${take.selected ? '#6366f1' : 'rgba(255, 255, 255, 0.1)'}`,
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onClick={() => {
                onTakeSelect(take.id);
                toggleTakeSelection(take.id);
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ color: '#fff', fontWeight: 'bold' }}>Take {index + 1}</span>
                  {take.selected && <span style={{ color: '#6366f1' }}>✓ Selected</span>}
                  {selectedTakes.has(take.id) && <span style={{ color: '#4caf50' }}>✓ In Comp</span>}
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <span style={{ color: '#888', fontSize: '0.85em' }}>
                    {formatTime(take.startTime)} - {formatTime(take.endTime)}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onTakeDelete(take.id);
                    }}
                    style={{
                      padding: '4px 8px',
                      backgroundColor: '#f44336',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '0.8em',
                    }}
                  >
                    Delete
                  </button>
                </div>
              </div>

              {take.audioBuffer && (
                <div style={{ marginTop: '8px', padding: '8px', backgroundColor: '#1a1a1a', borderRadius: '4px' }}>
                  <div style={{ color: '#aaa', fontSize: '0.85em' }}>
                    Duration: {formatTime(take.audioBuffer.duration)}
                    {' | '}
                    Sample Rate: {take.audioBuffer.sampleRate} Hz
                    {' | '}
                    Channels: {take.audioBuffer.numberOfChannels}
                  </div>
                  {/* Waveform visualization could go here */}
                  <div
                    style={{
                      width: '100%',
                      height: '40px',
                      backgroundColor: '#0a0a0a',
                      borderRadius: '2px',
                      marginTop: '8px',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                    }}
                  >
                    {/* Placeholder for waveform */}
                    <div style={{ color: '#666', fontSize: '0.75em', padding: '10px', textAlign: 'center' }}>
                      Waveform Preview
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {track.takes.length > 0 && (
        <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px', fontSize: '0.85em', color: '#aaa' }}>
          <div>Current Take: {track.currentTake + 1} of {track.takes.length}</div>
          <div style={{ marginTop: '4px' }}>
            Selected for Comp: {selectedTakes.size} take{selectedTakes.size !== 1 ? 's' : ''}
          </div>
        </div>
      )}
    </div>
  );
};
