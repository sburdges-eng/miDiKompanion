// MarkersLocatorsPanel - Features 95-107: Markers & Locators

import React, { useState, useEffect } from 'react';
import { TransportEngine, Locator, TimePosition } from './TransportEngine';

interface MarkersLocatorsPanelProps {
  engine: TransportEngine;
  onMarkerChange?: () => void;
}

export const MarkersLocatorsPanel: React.FC<MarkersLocatorsPanelProps> = ({
  engine,
  onMarkerChange,
}) => {
  const [state, setState] = useState(engine.getState());
  const [newMarkerName, setNewMarkerName] = useState('');
  const [newMarkerColor, setNewMarkerColor] = useState('#ff6b6b');
  const [newLocatorName, setNewLocatorName] = useState('');
  const [locatorType, setLocatorType] = useState<Locator['type']>('start');

  useEffect(() => {
    const interval = setInterval(() => {
      setState(engine.getState());
    }, 100);
    return () => clearInterval(interval);
  }, [engine]);

  const formatPosition = (pos: TimePosition): string => {
    return engine.getTimeFormatString(pos, state.displayFormat);
  };

  // Feature 95: Create Marker
  const handleCreateMarker = () => {
    if (newMarkerName.trim()) {
      engine.createMarker(newMarkerName, state.position, newMarkerColor);
      setNewMarkerName('');
      onMarkerChange?.();
    }
  };

  // Feature 100: Create Locator
  const handleCreateLocator = () => {
    if (newLocatorName.trim()) {
      engine.createLocator(newLocatorName, state.position, locatorType);
      setNewLocatorName('');
      onMarkerChange?.();
    }
  };

  // Feature 96: Delete Marker
  const handleDeleteMarker = (id: string) => {
    engine.deleteMarker(id);
    onMarkerChange?.();
  };

  // Feature 97: Go to Marker
  const handleGoToMarker = (id: string) => {
    engine.goToMarker(id);
    onMarkerChange?.();
  };

  // Feature 98: Previous Marker
  const handlePreviousMarker = () => {
    engine.previousMarker();
    onMarkerChange?.();
  };

  // Feature 99: Next Marker
  const handleNextMarker = () => {
    engine.nextMarker();
    onMarkerChange?.();
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '20px',
        padding: '20px',
        backgroundColor: '#1a1a1a',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <h3 style={{ color: '#fff', margin: '0 0 15px 0' }}>Markers & Locators (Features 95-107)</h3>

      {/* Marker Navigation */}
      <div style={{ display: 'flex', gap: '10px' }}>
        <button
          onClick={handlePreviousMarker}
          disabled={state.markers.length === 0}
          style={{
            padding: '8px 16px',
            backgroundColor: state.markers.length === 0 ? '#2a2a2a' : '#6366f1',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: state.markers.length === 0 ? 'not-allowed' : 'pointer',
            fontSize: '0.9em',
          }}
        >
          ⏮ Previous (Feature 98)
        </button>
        <button
          onClick={handleNextMarker}
          disabled={state.markers.length === 0}
          style={{
            padding: '8px 16px',
            backgroundColor: state.markers.length === 0 ? '#2a2a2a' : '#6366f1',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: state.markers.length === 0 ? 'not-allowed' : 'pointer',
            fontSize: '0.9em',
          }}
        >
          Next (Feature 99) ⏭
        </button>
      </div>

      {/* Create Marker (Feature 95) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Create Marker (Feature 95)
        </div>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="text"
            placeholder="Marker name"
            value={newMarkerName}
            onChange={(e) => setNewMarkerName(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleCreateMarker();
              }
            }}
            style={{
              flex: 1,
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          />
          <input
            type="color"
            value={newMarkerColor}
            onChange={(e) => setNewMarkerColor(e.target.value)}
            style={{
              width: '50px',
              height: '38px',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          />
          <button
            onClick={handleCreateMarker}
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

      {/* Markers List */}
      <div>
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Markers ({state.markers.length})
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
          {state.markers.length === 0 ? (
            <div style={{ color: '#888', fontSize: '0.85em', textAlign: 'center', padding: '20px' }}>
              No markers created yet
            </div>
          ) : (
            state.markers.map((marker) => (
              <div
                key={marker.id}
                style={{
                  padding: '12px',
                  backgroundColor: state.selectedMarker === marker.id ? '#6366f120' : '#2a2a2a',
                  borderRadius: '4px',
                  border: `2px solid ${state.selectedMarker === marker.id ? '#6366f1' : marker.color}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                }}
              >
                <div
                  style={{
                    width: '20px',
                    height: '20px',
                    borderRadius: '50%',
                    backgroundColor: marker.color,
                    border: '2px solid #fff',
                  }}
                />
                <div style={{ flex: 1 }}>
                  <div style={{ color: '#fff', fontWeight: 'bold', fontSize: '0.95em' }}>
                    {marker.name}
                  </div>
                  <div style={{ color: '#888', fontSize: '0.85em' }}>
                    {formatPosition(marker.position)}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '5px' }}>
                  <button
                    onClick={() => handleGoToMarker(marker.id)}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: '#6366f1',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      cursor: 'pointer',
                      fontSize: '0.85em',
                    }}
                  >
                    Go (Feature 97)
                  </button>
                  <button
                    onClick={() => handleDeleteMarker(marker.id)}
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
                    Delete (Feature 96)
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Create Locator (Feature 100) */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Create Locator (Feature 100)
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <select
            value={locatorType}
            onChange={(e) => setLocatorType(e.target.value as Locator['type'])}
            style={{
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          >
            <option value="start">Start</option>
            <option value="end">End</option>
            <option value="loop-start">Loop Start</option>
            <option value="loop-end">Loop End</option>
            <option value="punch-in">Punch In</option>
            <option value="punch-out">Punch Out</option>
          </select>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              type="text"
              placeholder="Locator name"
              value={newLocatorName}
              onChange={(e) => setNewLocatorName(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleCreateLocator();
                }
              }}
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
              onClick={handleCreateLocator}
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
      </div>

      {/* Locators List */}
      <div>
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Locators ({state.locators.length})
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
          {state.locators.length === 0 ? (
            <div style={{ color: '#888', fontSize: '0.85em', textAlign: 'center', padding: '20px' }}>
              No locators created yet
            </div>
          ) : (
            state.locators.map((locator) => (
              <div
                key={locator.id}
                style={{
                  padding: '12px',
                  backgroundColor: state.selectedLocator === locator.id ? '#6366f120' : '#2a2a2a',
                  borderRadius: '4px',
                  border: `2px solid ${state.selectedLocator === locator.id ? '#6366f1' : 'rgba(255, 255, 255, 0.2)'}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                }}
              >
                <div
                  style={{
                    padding: '4px 8px',
                    backgroundColor: '#6366f1',
                    borderRadius: '4px',
                    fontSize: '0.75em',
                    color: '#fff',
                    fontWeight: 'bold',
                  }}
                >
                  {locator.type.toUpperCase()}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ color: '#fff', fontWeight: 'bold', fontSize: '0.95em' }}>
                    {locator.name}
                  </div>
                  <div style={{ color: '#888', fontSize: '0.85em' }}>
                    {formatPosition(locator.position)}
                  </div>
                </div>
                <button
                  onClick={() => {
                    engine.goToLocator(locator.id);
                    onMarkerChange?.();
                  }}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: '#6366f1',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer',
                    fontSize: '0.85em',
                  }}
                >
                  Go
                </button>
                <button
                  onClick={() => {
                    engine.deleteLocator(locator.id);
                    onMarkerChange?.();
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
                  Delete
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
