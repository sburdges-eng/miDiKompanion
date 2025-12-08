// MIDIEditingPanel - UI Component for Features 183-200

import React, { useState, useEffect } from 'react';
import { MIDIEngine } from './MIDIEngine';

interface MIDIEditingPanelProps {
  engine: MIDIEngine;
  onTracksChange?: () => void;
}

export const MIDIEditingPanel: React.FC<MIDIEditingPanelProps> = ({
  engine,
  onTracksChange,
}) => {
  const [state, setState] = useState(engine.getState());
  const [newTrackName, setNewTrackName] = useState('');
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  const [quantizeGrid, setQuantizeGrid] = useState(state.quantizeGrid);
  const [quantizeStrength, setQuantizeStrength] = useState(state.quantizeStrength);

  useEffect(() => {
    const interval = setInterval(() => {
      setState(engine.getState());
    }, 100);
    return () => clearInterval(interval);
  }, [engine]);

  const handleCreateTrack = () => {
    if (newTrackName.trim()) {
      engine.createTrack(newTrackName);
      setNewTrackName('');
      onTracksChange?.();
    }
  };

  const handleDeleteTrack = (trackId: string) => {
    engine.deleteTrack(trackId);
    onTracksChange?.();
  };

  const handleQuantize = () => {
    const selectedNotes = engine.getSelectedNotes();
    if (selectedNotes.length > 0) {
      engine.quantizeNotes(selectedNotes);
      onTracksChange?.();
    }
  };

  const handleTranspose = (semitones: number) => {
    const selectedNotes = engine.getSelectedNotes();
    if (selectedNotes.length > 0) {
      engine.transposeNotes(selectedNotes, semitones);
      onTracksChange?.();
    }
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
      <h3 style={{ color: '#fff', margin: '0 0 15px 0' }}>MIDI Editing (Features 183-200)</h3>

      {/* Create Track (183) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Create MIDI Track (Feature 183)
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <input
            type="text"
            value={newTrackName}
            onChange={(e) => setNewTrackName(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleCreateTrack();
              }
            }}
            placeholder="Track name"
            style={{
              flex: 1,
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          />
          <button
            onClick={handleCreateTrack}
            style={{
              padding: '8px 16px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Create
          </button>
        </div>
      </div>

      {/* MIDI Tracks List */}
      <div>
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          MIDI Tracks ({state.tracks.length})
        </div>
        <div
          style={{
            maxHeight: '300px',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
          }}
        >
          {state.tracks.length === 0 ? (
            <div style={{ color: '#888', fontSize: '0.85em', textAlign: 'center', padding: '20px' }}>
              No MIDI tracks created yet
            </div>
          ) : (
            state.tracks.map((track) => (
              <div
                key={track.id}
                style={{
                  padding: '12px',
                  backgroundColor: selectedTrackId === track.id ? '#6366f120' : '#2a2a2a',
                  borderRadius: '4px',
                  border: `2px solid ${selectedTrackId === track.id ? '#6366f1' : 'rgba(255, 255, 255, 0.1)'}`,
                  cursor: 'pointer',
                }}
                onClick={() => setSelectedTrackId(track.id)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ color: '#fff', fontWeight: 'bold', fontSize: '0.95em' }}>
                      {track.name}
                    </div>
                    <div style={{ color: '#888', fontSize: '0.85em' }}>
                      {track.clips.length} clips • {track.clips.reduce((sum, c) => sum + c.notes.length, 0)} notes
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteTrack(track.id);
                    }}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: '#f44336',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      cursor: 'pointer',
                      fontSize: '0.85em',
                    }}
                  >
                    Delete (184)
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Quantize (198) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>Quantize (Feature 198)</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <label style={{ color: '#aaa', fontSize: '0.85em', minWidth: '100px' }}>Grid:</label>
            <select
              value={quantizeGrid}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setQuantizeGrid(val);
                engine.setQuantizeGrid(val);
              }}
              style={{
                flex: 1,
                padding: '8px',
                backgroundColor: '#2a2a2a',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
              }}
            >
              <option value={1}>Whole Note</option>
              <option value={1 / 2}>Half Note</option>
              <option value={1 / 4}>Quarter Note</option>
              <option value={1 / 8}>Eighth Note</option>
              <option value={1 / 16}>16th Note</option>
              <option value={1 / 32}>32nd Note</option>
            </select>
          </div>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <label style={{ color: '#aaa', fontSize: '0.85em', minWidth: '100px' }}>Strength:</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={quantizeStrength}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setQuantizeStrength(val);
                engine.setQuantizeStrength(val);
              }}
              style={{ flex: 1 }}
            />
            <span style={{ color: '#aaa', fontSize: '0.85em', minWidth: '40px' }}>
              {(quantizeStrength * 100).toFixed(0)}%
            </span>
          </div>
          <button
            onClick={handleQuantize}
            disabled={engine.getSelectedNotes().length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: engine.getSelectedNotes().length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: engine.getSelectedNotes().length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            Quantize Selected Notes
          </button>
        </div>
      </div>

      {/* Transpose (199) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>Transpose (Feature 199)</div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={() => handleTranspose(-12)}
            disabled={engine.getSelectedNotes().length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: engine.getSelectedNotes().length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: engine.getSelectedNotes().length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            -12 semitones
          </button>
          <button
            onClick={() => handleTranspose(-1)}
            disabled={engine.getSelectedNotes().length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: engine.getSelectedNotes().length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: engine.getSelectedNotes().length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            -1 semitone
          </button>
          <button
            onClick={() => handleTranspose(1)}
            disabled={engine.getSelectedNotes().length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: engine.getSelectedNotes().length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: engine.getSelectedNotes().length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            +1 semitone
          </button>
          <button
            onClick={() => handleTranspose(12)}
            disabled={engine.getSelectedNotes().length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: engine.getSelectedNotes().length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: engine.getSelectedNotes().length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            +12 semitones
          </button>
        </div>
      </div>

      {/* Note Selection (189-192) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Note Selection (Features 189-192)
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => engine.selectAllNotes()}
            style={{
              padding: '8px 16px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Select All (191)
          </button>
          <button
            onClick={() => engine.deselectAllNotes()}
            style={{
              padding: '8px 16px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Deselect All (192)
          </button>
        </div>
      </div>

      {/* Copy/Paste Notes (200-201) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Copy/Paste Notes (Features 200-201)
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => {
              const selected = engine.getSelectedNotes();
              if (selected.length > 0) {
                engine.copyNotes(selected);
              }
            }}
            disabled={engine.getSelectedNotes().length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: engine.getSelectedNotes().length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: engine.getSelectedNotes().length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            Copy Notes (200)
          </button>
          <button
            onClick={() => {
              if (selectedTrackId && state.clipboard.notes.length > 0) {
                const track = state.tracks.find(t => t.id === selectedTrackId);
                if (track && track.clips.length > 0) {
                  engine.pasteNotes(track.clips[0].id, 0);
                  onTracksChange?.();
                }
              }
            }}
            disabled={!selectedTrackId || state.clipboard.notes.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedTrackId && state.clipboard.notes.length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedTrackId && state.clipboard.notes.length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            Paste Notes (201)
          </button>
        </div>
      </div>

      {/* Status */}
      <div
        style={{
          padding: '10px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          fontSize: '0.85em',
          color: '#888',
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <div>Tracks: {state.tracks.length}</div>
        <div>Selected Notes: {engine.getSelectedNotes().length}</div>
        <div>Clipboard: {state.clipboard.notes.length} notes</div>
      </div>
    </div>
  );
};
