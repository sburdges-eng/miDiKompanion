// AutomationPanel - UI Component for Features 228-247

import React, { useState, useEffect } from 'react';
import { AutomationEngine } from './AutomationEngine';

interface AutomationPanelProps {
  engine: AutomationEngine;
  onLanesChange?: () => void;
}

export const AutomationPanel: React.FC<AutomationPanelProps> = ({
  engine,
  onLanesChange,
}) => {
  const [state, setState] = useState(engine.getState());
  const [selectedTrackId] = useState<string>('track-1');
  const [parameter, setParameter] = useState<string>('volume');
  const [recordingMode, setRecordingMode] = useState<'touch' | 'latch' | 'write'>('touch');

  useEffect(() => {
    const interval = setInterval(() => {
      setState(engine.getState());
    }, 100);
    return () => clearInterval(interval);
  }, [engine]);

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
      <h3 style={{ color: '#fff', margin: '0 0 15px 0' }}>Automation (Features 228-247)</h3>

      {/* Create Automation Lane (228) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Create Automation Lane (Feature 228)
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <select
            value={parameter}
            onChange={(e) => setParameter(e.target.value)}
            style={{
              flex: 1,
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          >
            <option value="volume">Volume</option>
            <option value="pan">Pan</option>
            <option value="filter-cutoff">Filter Cutoff</option>
            <option value="filter-resonance">Filter Resonance</option>
            <option value="reverb-send">Reverb Send</option>
            <option value="delay-send">Delay Send</option>
          </select>
          <button
            onClick={() => {
              engine.createLane(selectedTrackId, parameter);
              onLanesChange?.();
            }}
            style={{
              padding: '8px 16px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Create Lane
          </button>
        </div>
      </div>

      {/* Recording Modes (233-234) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Recording Mode (Features 233-234)
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button
            onClick={() => {
              setRecordingMode('touch');
              engine.startRecording(parameter, 'touch');
            }}
            style={{
              padding: '8px 16px',
              backgroundColor: recordingMode === 'touch' ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Touch
          </button>
          <button
            onClick={() => {
              setRecordingMode('latch');
              engine.startRecording(parameter, 'latch');
            }}
            style={{
              padding: '8px 16px',
              backgroundColor: recordingMode === 'latch' ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Latch
          </button>
          <button
            onClick={() => {
              setRecordingMode('write');
              engine.startRecording(parameter, 'write');
            }}
            style={{
              padding: '8px 16px',
              backgroundColor: recordingMode === 'write' ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Write
          </button>
          <button
            onClick={() => engine.stopRecording()}
            disabled={!state.recording}
            style={{
              padding: '8px 16px',
              backgroundColor: state.recording ? '#f44336' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: state.recording ? 'pointer' : 'not-allowed',
            }}
          >
            Stop
          </button>
        </div>
      </div>

      {/* Read/Trim Modes (235-236) */}
      <div style={{ display: 'flex', gap: '10px' }}>
        <button
          onClick={() => engine.setReadMode(!state.readMode)}
          style={{
            padding: '8px 16px',
            backgroundColor: state.readMode ? '#6366f1' : '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
          }}
        >
          Read (235)
        </button>
        <button
          onClick={() => engine.setTrimMode(!state.trimMode)}
          style={{
            padding: '8px 16px',
            backgroundColor: state.trimMode ? '#6366f1' : '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
          }}
        >
          Trim (236)
        </button>
      </div>

      {/* Automation Lanes List */}
      <div>
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Automation Lanes ({state.lanes.length})
        </div>
        <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
          {state.lanes.length === 0 ? (
            <div style={{ color: '#888', fontSize: '0.85em', textAlign: 'center', padding: '20px' }}>
              No automation lanes created yet
            </div>
          ) : (
            state.lanes.map((lane) => (
              <div
                key={lane.id}
                style={{
                  padding: '12px',
                  backgroundColor: '#2a2a2a',
                  borderRadius: '4px',
                  marginBottom: '8px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ color: '#fff', fontWeight: 'bold' }}>
                      {lane.parameter} ({lane.trackId})
                    </div>
                    <div style={{ color: '#888', fontSize: '0.85em' }}>
                      {lane.points.length} points
                    </div>
                  </div>
                  <button
                    onClick={() => {
                      engine.deleteLane(lane.id);
                      onLanesChange?.();
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
                    Delete (229)
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
