// PluginPanel - UI Component for Features 347-474

import React, { useState, useEffect } from 'react';
import { PluginEngine } from './PluginEngine';

interface PluginPanelProps {
  engine: PluginEngine;
  trackId: string;
  onPluginsChange?: () => void;
}

export const PluginPanel: React.FC<PluginPanelProps> = ({
  engine,
  trackId,
  onPluginsChange,
}) => {
  const [, setState] = useState(engine.getState());
  const [selectedPlugin, setSelectedPlugin] = useState<string | null>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setState(engine.getState());
    }, 100);
    return () => clearInterval(interval);
  }, [engine]);

  const chain = engine.getChain(trackId);

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
      <h3 style={{ color: '#fff', margin: '0 0 15px 0' }}>Plugins (Features 347-474)</h3>

      {/* Add Plugin Buttons */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Add Plugin
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
          <button
            onClick={() => {
              engine.createParametricEQ(trackId);
              onPluginsChange?.();
            }}
            style={{
              padding: '10px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Parametric EQ (381)
          </button>
          <button
            onClick={() => {
              engine.createCompressor(trackId);
              onPluginsChange?.();
            }}
            style={{
              padding: '10px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Compressor (399)
          </button>
          <button
            onClick={() => {
              engine.createReverb(trackId);
              onPluginsChange?.();
            }}
            style={{
              padding: '10px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Reverb (421)
          </button>
          <button
            onClick={() => {
              engine.createChorus(trackId);
              onPluginsChange?.();
            }}
            style={{
              padding: '10px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Chorus (443)
          </button>
          <button
            onClick={() => {
              engine.createSaturation(trackId);
              onPluginsChange?.();
            }}
            style={{
              padding: '10px',
              backgroundColor: '#6366f1',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            Saturation (458)
          </button>
        </div>
      </div>

      {/* Plugin Chain */}
      <div>
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '10px' }}>
          Plugin Chain ({chain?.plugins.length || 0})
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {chain && chain.plugins.length > 0 ? (
            chain.plugins.map((plugin) => (
              <div
                key={plugin.id}
                style={{
                  padding: '12px',
                  backgroundColor: selectedPlugin === plugin.id ? '#6366f120' : '#2a2a2a',
                  borderRadius: '4px',
                  border: `2px solid ${selectedPlugin === plugin.id ? '#6366f1' : 'rgba(255, 255, 255, 0.1)'}`,
                  cursor: 'pointer',
                }}
                onClick={() => setSelectedPlugin(plugin.id)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ color: '#fff', fontWeight: 'bold' }}>{plugin.name}</div>
                    <div style={{ color: '#888', fontSize: '0.85em' }}>
                      {plugin.manufacturer} • {plugin.format.toUpperCase()}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '5px' }}>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        engine.bypassPlugin(plugin.id, !plugin.bypassed);
                        onPluginsChange?.();
                      }}
                      style={{
                        padding: '6px 12px',
                        backgroundColor: plugin.bypassed ? '#f44336' : '#2a2a2a',
                        border: 'none',
                        borderRadius: '4px',
                        color: '#fff',
                        cursor: 'pointer',
                        fontSize: '0.85em',
                      }}
                    >
                      {plugin.bypassed ? 'Bypassed' : 'Active'}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        engine.removePluginFromChain(trackId, plugin.id);
                        onPluginsChange?.();
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
                      Remove (361)
                    </button>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div style={{ color: '#888', fontSize: '0.85em', textAlign: 'center', padding: '20px' }}>
              No plugins in chain
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
