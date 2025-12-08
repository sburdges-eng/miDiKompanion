// AudioEditingPanel - UI Component for Features 108-182

import React, { useState, useEffect } from 'react';
import { AudioEditingEngine } from './AudioEditingEngine';

interface AudioEditingPanelProps {
  engine: AudioEditingEngine;
  onRegionsChange?: () => void;
}

export const AudioEditingPanel: React.FC<AudioEditingPanelProps> = ({
  engine,
  onRegionsChange,
}) => {
  const [state, setState] = useState(engine.getState());
  const [selectedRegionIds, setSelectedRegionIds] = useState<string[]>([]);
  const [splitTime, setSplitTime] = useState<number>(0);
  const [fadeDuration, setFadeDuration] = useState<number>(0.1);
  const [normalizeLevel, setNormalizeLevel] = useState<number>(-0.1);

  useEffect(() => {
    const interval = setInterval(() => {
      setState(engine.getState());
      setSelectedRegionIds(Array.from(engine.getSelectedRegions()));
    }, 100);
    return () => clearInterval(interval);
  }, [engine]);

  const handleCut = () => {
    if (selectedRegionIds.length > 0) {
      engine.cut(selectedRegionIds);
      onRegionsChange?.();
    }
  };

  const handleCopy = () => {
    if (selectedRegionIds.length > 0) {
      engine.copy(selectedRegionIds);
    }
  };

  const handlePaste = () => {
    if (state.clipboard.length > 0) {
      engine.paste('track-1', 0); // Would use actual track and position
      onRegionsChange?.();
    }
  };

  const handleDelete = () => {
    if (selectedRegionIds.length > 0) {
      engine.deleteRegions(selectedRegionIds);
      onRegionsChange?.();
    }
  };

  const handleSplit = () => {
    if (selectedRegionIds.length === 1 && splitTime > 0) {
      engine.split(selectedRegionIds[0], splitTime);
      onRegionsChange?.();
    }
  };

  const handleUndo = () => {
    engine.undo();
    onRegionsChange?.();
  };

  const handleRedo = () => {
    engine.redo();
    onRegionsChange?.();
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
      <h3 style={{ color: '#fff', margin: '0 0 15px 0' }}>Audio Editing (Features 108-182)</h3>

      {/* Basic Editing Tools (108-125) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px', fontWeight: 'bold' }}>
          Basic Editing (Features 108-125)
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={handleCut}
            disabled={selectedRegionIds.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedRegionIds.length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedRegionIds.length > 0 ? 'pointer' : 'not-allowed',
              fontSize: '0.9em',
            }}
          >
            ✂️ Cut (108)
          </button>
          <button
            onClick={handleCopy}
            disabled={selectedRegionIds.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedRegionIds.length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedRegionIds.length > 0 ? 'pointer' : 'not-allowed',
              fontSize: '0.9em',
            }}
          >
            📋 Copy (109)
          </button>
          <button
            onClick={handlePaste}
            disabled={state.clipboard.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: state.clipboard.length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: state.clipboard.length > 0 ? 'pointer' : 'not-allowed',
              fontSize: '0.9em',
            }}
          >
            📄 Paste (110)
          </button>
          <button
            onClick={handleDelete}
            disabled={selectedRegionIds.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedRegionIds.length > 0 ? '#f44336' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedRegionIds.length > 0 ? 'pointer' : 'not-allowed',
              fontSize: '0.9em',
            }}
          >
            🗑️ Delete (111)
          </button>
          <button
            onClick={handleUndo}
            disabled={!engine.canUndo()}
            style={{
              padding: '8px 16px',
              backgroundColor: engine.canUndo() ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: engine.canUndo() ? 'pointer' : 'not-allowed',
              fontSize: '0.9em',
            }}
          >
            ↶ Undo (122)
          </button>
          <button
            onClick={handleRedo}
            disabled={!engine.canRedo()}
            style={{
              padding: '8px 16px',
              backgroundColor: engine.canRedo() ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: engine.canRedo() ? 'pointer' : 'not-allowed',
              fontSize: '0.9em',
            }}
          >
            ↷ Redo (123)
          </button>
        </div>
      </div>

      {/* Split Tool (112) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>Split (Feature 112)</div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="number"
            value={splitTime}
            onChange={(e) => setSplitTime(parseFloat(e.target.value))}
            step="0.1"
            min="0"
            placeholder="Split time (seconds)"
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
            onClick={handleSplit}
            disabled={selectedRegionIds.length !== 1}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedRegionIds.length === 1 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedRegionIds.length === 1 ? 'pointer' : 'not-allowed',
            }}
          >
            Split
          </button>
        </div>
      </div>

      {/* Fade Controls (114-115) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Fade In/Out (Features 114-115)
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="number"
            value={fadeDuration}
            onChange={(e) => setFadeDuration(parseFloat(e.target.value))}
            step="0.01"
            min="0"
            max="10"
            style={{
              width: '120px',
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          />
          <span style={{ color: '#aaa', fontSize: '0.85em' }}>seconds</span>
          <button
            onClick={() => {
              selectedRegionIds.forEach(id => engine.setFadeIn(id, fadeDuration));
              onRegionsChange?.();
            }}
            disabled={selectedRegionIds.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedRegionIds.length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedRegionIds.length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            Fade In
          </button>
          <button
            onClick={() => {
              selectedRegionIds.forEach(id => engine.setFadeOut(id, fadeDuration));
              onRegionsChange?.();
            }}
            disabled={selectedRegionIds.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedRegionIds.length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedRegionIds.length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            Fade Out
          </button>
        </div>
      </div>

      {/* Normalize (117) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>Normalize (Feature 117)</div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="number"
            value={normalizeLevel}
            onChange={(e) => setNormalizeLevel(parseFloat(e.target.value))}
            step="0.1"
            min="-60"
            max="0"
            style={{
              width: '120px',
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          />
          <span style={{ color: '#aaa', fontSize: '0.85em' }}>dB</span>
          <button
            onClick={() => {
              selectedRegionIds.forEach(id => engine.normalize(id, normalizeLevel));
              onRegionsChange?.();
            }}
            disabled={selectedRegionIds.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedRegionIds.length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedRegionIds.length > 0 ? 'pointer' : 'not-allowed',
            }}
          >
            Normalize
          </button>
        </div>
      </div>

      {/* Reverse (118) */}
      <div style={{ display: 'flex', gap: '10px' }}>
        <button
          onClick={() => {
            selectedRegionIds.forEach(id => engine.reverse(id));
            onRegionsChange?.();
          }}
          disabled={selectedRegionIds.length === 0}
          style={{
            padding: '8px 16px',
            backgroundColor: selectedRegionIds.length > 0 ? '#6366f1' : '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: selectedRegionIds.length > 0 ? 'pointer' : 'not-allowed',
            fontSize: '0.9em',
          }}
        >
          ↻ Reverse (118)
        </button>
      </div>

      {/* Snap Settings (124-125) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Snap Settings (Features 124-125)
        </div>
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#fff' }}>
            <input
              type="checkbox"
              checked={state.snapToGrid}
              onChange={(e) => engine.setSnapToGrid(e.target.checked)}
              style={{ width: '18px', height: '18px' }}
            />
            Snap to Grid (124)
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#fff' }}>
            <input
              type="checkbox"
              checked={state.snapToZero}
              onChange={(e) => engine.setSnapToZero(e.target.checked)}
              style={{ width: '18px', height: '18px' }}
            />
            Snap to Zero (125)
          </label>
        </div>
      </div>

      {/* Time Manipulation (147-163) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px', fontWeight: 'bold' }}>
          Time Manipulation (Features 147-163)
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <button
            onClick={() => {
              selectedRegionIds.forEach(id => engine.timeStretch(id, 1.5));
              onRegionsChange?.();
            }}
            disabled={selectedRegionIds.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedRegionIds.length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedRegionIds.length > 0 ? 'pointer' : 'not-allowed',
              fontSize: '0.9em',
            }}
          >
            ⏱️ Time Stretch (147)
          </button>
          <button
            onClick={() => {
              selectedRegionIds.forEach(id => engine.pitchShift(id, 12));
              onRegionsChange?.();
            }}
            disabled={selectedRegionIds.length === 0}
            style={{
              padding: '8px 16px',
              backgroundColor: selectedRegionIds.length > 0 ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: selectedRegionIds.length > 0 ? 'pointer' : 'not-allowed',
              fontSize: '0.9em',
            }}
          >
            🎵 Pitch Shift (148)
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
        <div>Regions: {state.regions.length}</div>
        <div>Selected: {selectedRegionIds.length}</div>
        <div>Clipboard: {state.clipboard.length} items</div>
      </div>
    </div>
  );
};
