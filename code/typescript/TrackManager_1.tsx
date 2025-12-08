import React, { useState } from 'react';
import { Track, RecordingEngine } from './RecordingEngine';

interface TrackManagerProps {
  engine: RecordingEngine;
  tracks: Track[];
  onTracksChange: (tracks: Track[]) => void;
  onTrackSelect: (trackId: string) => void;
}

export const TrackManager: React.FC<TrackManagerProps> = ({
  engine,
  tracks,
  onTracksChange,
  onTrackSelect,
}) => {
  const [showAddTrack, setShowAddTrack] = useState(false);
  const [newTrackName, setNewTrackName] = useState('');

  const addTrack = () => {
    const newTrack: Track = {
      id: `track-${Date.now()}`,
      name: newTrackName || `Track ${tracks.length + 1}`,
      armed: false,
      recordSafe: false,
      inputMonitoring: 'off',
      takes: [],
      currentTake: -1,
      inputChannel: tracks.length,
      pan: 0,
      volume: 1,
    };

    const updatedTracks = [...tracks, newTrack];
    onTracksChange(updatedTracks);
    setNewTrackName('');
    setShowAddTrack(false);
  };

  const removeTrack = (trackId: string) => {
    const updatedTracks = tracks.filter((t) => t.id !== trackId);
    onTracksChange(updatedTracks);
  };

  const toggleArm = (trackId: string) => {
    const track = tracks.find((t) => t.id === trackId);
    if (track) {
      engine.setTrackArmed(trackId, !track.armed);
      onTracksChange(engine.getTracks());
    }
  };

  const toggleRecordSafe = (trackId: string) => {
    const track = tracks.find((t) => t.id === trackId);
    if (track) {
      engine.setRecordSafe(trackId, !track.recordSafe);
      onTracksChange(engine.getTracks());
    }
  };

  const setMonitoring = (trackId: string, mode: 'off' | 'software' | 'hardware' | 'direct') => {
    engine.setInputMonitoring(trackId, mode);
    onTracksChange(engine.getTracks());
  };

  const armGroup = (trackIds: string[]) => {
    trackIds.forEach((id) => {
      engine.setTrackArmed(id, true);
    });
    onTracksChange(engine.getTracks());
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
        <h3 style={{ margin: 0, color: '#fff' }}>Tracks</h3>
        <button
          onClick={() => setShowAddTrack(!showAddTrack)}
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
          + Add Track
        </button>
      </div>

      {showAddTrack && (
        <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
          <input
            type="text"
            value={newTrackName}
            onChange={(e) => setNewTrackName(e.target.value)}
            placeholder="Track name"
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: '#1a1a1a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              marginBottom: '8px',
            }}
          />
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={addTrack}
              style={{
                flex: 1,
                padding: '8px',
                backgroundColor: '#4caf50',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Add
            </button>
            <button
              onClick={() => setShowAddTrack(false)}
              style={{
                flex: 1,
                padding: '8px',
                backgroundColor: '#666',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {tracks.map((track) => (
          <div
            key={track.id}
            onClick={() => onTrackSelect(track.id)}
            style={{
              padding: '12px',
              backgroundColor: track.armed ? '#4caf5020' : '#2a2a2a',
              border: `2px solid ${track.armed ? '#4caf50' : 'rgba(255, 255, 255, 0.1)'}`,
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <span style={{ color: '#fff', fontWeight: 'bold' }}>{track.name}</span>
                {track.armed && <span style={{ color: '#4caf50' }}>⚡</span>}
                {track.recordSafe && <span style={{ color: '#f44336' }}>🔒</span>}
                <span style={{ color: '#888', fontSize: '0.85em' }}>
                  {track.takes.length} take{track.takes.length !== 1 ? 's' : ''}
                </span>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removeTrack(track.id);
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
                ×
              </button>
            </div>

            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleArm(track.id);
                }}
                style={{
                  padding: '6px 12px',
                  backgroundColor: track.armed ? '#4caf50' : '#333',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.85em',
                }}
              >
                {track.armed ? '⚡ Armed' : 'Arm'}
              </button>

              <button
                onClick={(e) => {
                  e.stopPropagation();
                  toggleRecordSafe(track.id);
                }}
                style={{
                  padding: '6px 12px',
                  backgroundColor: track.recordSafe ? '#f44336' : '#333',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.85em',
                }}
              >
                {track.recordSafe ? '🔒 Safe' : 'Record Safe'}
              </button>

              <select
                value={track.inputMonitoring}
                onChange={(e) => {
                  e.stopPropagation();
                  setMonitoring(track.id, e.target.value as any);
                }}
                onClick={(e) => e.stopPropagation()}
                style={{
                  padding: '6px 12px',
                  backgroundColor: '#333',
                  color: 'white',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.85em',
                }}
              >
                <option value="off">Monitoring: Off</option>
                <option value="software">Software</option>
                <option value="hardware">Hardware</option>
                <option value="direct">Direct</option>
              </select>
            </div>
          </div>
        ))}
      </div>

      {tracks.length > 1 && (
        <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
          <button
            onClick={() => armGroup(tracks.map((t) => t.id))}
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: '#6366f1',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9em',
            }}
          >
            Arm All Tracks
          </button>
        </div>
      )}
    </div>
  );
};
