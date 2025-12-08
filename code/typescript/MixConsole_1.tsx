import React, { useState } from 'react';
import { EQ } from './EQ';

interface CompressorSettings {
  threshold: number;
  ratio: number;
  attack: number;
  release: number;
  makeup: number;
  enabled: boolean;
}

interface LimiterSettings {
  ceiling: number;
  release: number;
  enabled: boolean;
}

interface ReverbSettings {
  roomSize: number;
  damping: number;
  wetDry: number;
  enabled: boolean;
}

interface MixConsoleProps {
  channelName?: string;
  onSettingsChange?: (settings: any) => void;
}

const MIX_PRESETS: { [key: string]: { compressor: Partial<CompressorSettings>; limiter: Partial<LimiterSettings> } } = {
  mastering: {
    compressor: { threshold: -6, ratio: 2, attack: 30, release: 200, makeup: 2 },
    limiter: { ceiling: -0.3, release: 50 },
  },
  vocals: {
    compressor: { threshold: -18, ratio: 4, attack: 10, release: 100, makeup: 4 },
    limiter: { ceiling: -1, release: 100 },
  },
  drums: {
    compressor: { threshold: -12, ratio: 6, attack: 5, release: 50, makeup: 3 },
    limiter: { ceiling: -0.5, release: 30 },
  },
  bass: {
    compressor: { threshold: -15, ratio: 4, attack: 20, release: 150, makeup: 3 },
    limiter: { ceiling: -1, release: 80 },
  },
  gentle: {
    compressor: { threshold: -20, ratio: 2, attack: 50, release: 300, makeup: 1 },
    limiter: { ceiling: -0.5, release: 150 },
  },
};

export const MixConsole: React.FC<MixConsoleProps> = ({
  channelName = 'Master',
  onSettingsChange,
}) => {
  const [activeTab, setActiveTab] = useState<'eq' | 'dynamics' | 'reverb'>('dynamics');
  const [compressor, setCompressor] = useState<CompressorSettings>({
    threshold: -12,
    ratio: 4,
    attack: 20,
    release: 150,
    makeup: 0,
    enabled: true,
  });
  const [limiter, setLimiter] = useState<LimiterSettings>({
    ceiling: -0.3,
    release: 50,
    enabled: true,
  });
  const [reverb, setReverb] = useState<ReverbSettings>({
    roomSize: 50,
    damping: 50,
    wetDry: 20,
    enabled: false,
  });
  const [gainReduction, setGainReduction] = useState(0);

  const updateCompressor = (changes: Partial<CompressorSettings>) => {
    const newSettings = { ...compressor, ...changes };
    setCompressor(newSettings);
    onSettingsChange?.({ compressor: newSettings, limiter, reverb });
  };

  const updateLimiter = (changes: Partial<LimiterSettings>) => {
    const newSettings = { ...limiter, ...changes };
    setLimiter(newSettings);
    onSettingsChange?.({ compressor, limiter: newSettings, reverb });
  };

  const updateReverb = (changes: Partial<ReverbSettings>) => {
    const newSettings = { ...reverb, ...changes };
    setReverb(newSettings);
    onSettingsChange?.({ compressor, limiter, reverb: newSettings });
  };

  const applyPreset = (presetName: string) => {
    const preset = MIX_PRESETS[presetName];
    if (preset) {
      setCompressor({ ...compressor, ...preset.compressor });
      setLimiter({ ...limiter, ...preset.limiter });
    }
  };

  // Simulate gain reduction meter
  React.useEffect(() => {
    if (compressor.enabled) {
      const interval = setInterval(() => {
        setGainReduction(Math.random() * 6);
      }, 100);
      return () => clearInterval(interval);
    } else {
      setGainReduction(0);
    }
  }, [compressor.enabled]);

  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      borderRadius: '8px',
      overflow: 'hidden',
      color: '#fff',
    }}>
      {/* Header */}
      <div style={{
        padding: '15px',
        backgroundColor: '#0f0f0f',
        borderBottom: '1px solid #333',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{ fontWeight: 'bold' }}>Mix Console - {channelName}</span>
        <select
          onChange={(e) => applyPreset(e.target.value)}
          style={{
            padding: '4px 8px',
            backgroundColor: '#333',
            border: '1px solid #555',
            borderRadius: '4px',
            color: '#fff',
            fontSize: '0.85em',
          }}
        >
          <option value="">Presets...</option>
          <option value="mastering">Mastering</option>
          <option value="vocals">Vocals</option>
          <option value="drums">Drums</option>
          <option value="bass">Bass</option>
          <option value="gentle">Gentle</option>
        </select>
      </div>

      {/* Tabs */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid #333',
      }}>
        {[
          { id: 'eq', label: 'EQ' },
          { id: 'dynamics', label: 'Dynamics' },
          { id: 'reverb', label: 'Reverb' },
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            style={{
              flex: 1,
              padding: '10px',
              backgroundColor: activeTab === tab.id ? '#2a2a2a' : 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid #6366f1' : '2px solid transparent',
              color: activeTab === tab.id ? '#fff' : '#888',
              cursor: 'pointer',
              fontWeight: activeTab === tab.id ? 'bold' : 'normal',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: '20px' }}>
        {activeTab === 'eq' && (
          <EQ channelName={channelName} />
        )}

        {activeTab === 'dynamics' && (
          <div>
            {/* Compressor */}
            <div style={{ marginBottom: '25px' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '15px',
              }}>
                <span style={{ fontWeight: 'bold' }}>Compressor</span>
                <button
                  onClick={() => updateCompressor({ enabled: !compressor.enabled })}
                  style={{
                    padding: '4px 12px',
                    backgroundColor: compressor.enabled ? '#4caf50' : '#666',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer',
                    fontSize: '0.8em',
                  }}
                >
                  {compressor.enabled ? 'ON' : 'OFF'}
                </button>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(5, 1fr)',
                gap: '15px',
                opacity: compressor.enabled ? 1 : 0.5,
              }}>
                <div>
                  <label style={{ fontSize: '0.75em', color: '#888' }}>Threshold</label>
                  <input
                    type="range"
                    min="-40"
                    max="0"
                    value={compressor.threshold}
                    onChange={(e) => updateCompressor({ threshold: Number(e.target.value) })}
                    disabled={!compressor.enabled}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '0.75em', textAlign: 'center' }}>{compressor.threshold}dB</div>
                </div>

                <div>
                  <label style={{ fontSize: '0.75em', color: '#888' }}>Ratio</label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    value={compressor.ratio}
                    onChange={(e) => updateCompressor({ ratio: Number(e.target.value) })}
                    disabled={!compressor.enabled}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '0.75em', textAlign: 'center' }}>{compressor.ratio}:1</div>
                </div>

                <div>
                  <label style={{ fontSize: '0.75em', color: '#888' }}>Attack</label>
                  <input
                    type="range"
                    min="0.1"
                    max="100"
                    value={compressor.attack}
                    onChange={(e) => updateCompressor({ attack: Number(e.target.value) })}
                    disabled={!compressor.enabled}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '0.75em', textAlign: 'center' }}>{compressor.attack}ms</div>
                </div>

                <div>
                  <label style={{ fontSize: '0.75em', color: '#888' }}>Release</label>
                  <input
                    type="range"
                    min="10"
                    max="1000"
                    value={compressor.release}
                    onChange={(e) => updateCompressor({ release: Number(e.target.value) })}
                    disabled={!compressor.enabled}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '0.75em', textAlign: 'center' }}>{compressor.release}ms</div>
                </div>

                <div>
                  <label style={{ fontSize: '0.75em', color: '#888' }}>Makeup</label>
                  <input
                    type="range"
                    min="0"
                    max="24"
                    value={compressor.makeup}
                    onChange={(e) => updateCompressor({ makeup: Number(e.target.value) })}
                    disabled={!compressor.enabled}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '0.75em', textAlign: 'center' }}>+{compressor.makeup}dB</div>
                </div>
              </div>

              {/* Gain Reduction Meter */}
              <div style={{ marginTop: '15px' }}>
                <label style={{ fontSize: '0.75em', color: '#888' }}>Gain Reduction</label>
                <div style={{
                  height: '8px',
                  backgroundColor: '#333',
                  borderRadius: '4px',
                  overflow: 'hidden',
                  marginTop: '4px',
                }}>
                  <div style={{
                    width: `${(gainReduction / 12) * 100}%`,
                    height: '100%',
                    backgroundColor: gainReduction > 6 ? '#f44336' : '#f59e0b',
                    transition: 'width 0.05s',
                  }} />
                </div>
                <div style={{ fontSize: '0.7em', textAlign: 'right', color: '#888' }}>
                  -{gainReduction.toFixed(1)}dB
                </div>
              </div>
            </div>

            {/* Limiter */}
            <div style={{
              padding: '15px',
              backgroundColor: '#222',
              borderRadius: '8px',
            }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '15px',
              }}>
                <span style={{ fontWeight: 'bold' }}>Limiter</span>
                <button
                  onClick={() => updateLimiter({ enabled: !limiter.enabled })}
                  style={{
                    padding: '4px 12px',
                    backgroundColor: limiter.enabled ? '#4caf50' : '#666',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer',
                    fontSize: '0.8em',
                  }}
                >
                  {limiter.enabled ? 'ON' : 'OFF'}
                </button>
              </div>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, 1fr)',
                gap: '20px',
                opacity: limiter.enabled ? 1 : 0.5,
              }}>
                <div>
                  <label style={{ fontSize: '0.75em', color: '#888' }}>Ceiling</label>
                  <input
                    type="range"
                    min="-6"
                    max="0"
                    step="0.1"
                    value={limiter.ceiling}
                    onChange={(e) => updateLimiter({ ceiling: Number(e.target.value) })}
                    disabled={!limiter.enabled}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '0.75em', textAlign: 'center' }}>{limiter.ceiling}dB</div>
                </div>

                <div>
                  <label style={{ fontSize: '0.75em', color: '#888' }}>Release</label>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    value={limiter.release}
                    onChange={(e) => updateLimiter({ release: Number(e.target.value) })}
                    disabled={!limiter.enabled}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '0.75em', textAlign: 'center' }}>{limiter.release}ms</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'reverb' && (
          <div>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '20px',
            }}>
              <span style={{ fontWeight: 'bold' }}>Reverb</span>
              <button
                onClick={() => updateReverb({ enabled: !reverb.enabled })}
                style={{
                  padding: '4px 12px',
                  backgroundColor: reverb.enabled ? '#4caf50' : '#666',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: 'pointer',
                  fontSize: '0.8em',
                }}
              >
                {reverb.enabled ? 'ON' : 'OFF'}
              </button>
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '20px',
              opacity: reverb.enabled ? 1 : 0.5,
            }}>
              <div>
                <label style={{ fontSize: '0.75em', color: '#888' }}>Room Size</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={reverb.roomSize}
                  onChange={(e) => updateReverb({ roomSize: Number(e.target.value) })}
                  disabled={!reverb.enabled}
                  style={{ width: '100%' }}
                />
                <div style={{ fontSize: '0.75em', textAlign: 'center' }}>{reverb.roomSize}%</div>
              </div>

              <div>
                <label style={{ fontSize: '0.75em', color: '#888' }}>Damping</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={reverb.damping}
                  onChange={(e) => updateReverb({ damping: Number(e.target.value) })}
                  disabled={!reverb.enabled}
                  style={{ width: '100%' }}
                />
                <div style={{ fontSize: '0.75em', textAlign: 'center' }}>{reverb.damping}%</div>
              </div>

              <div>
                <label style={{ fontSize: '0.75em', color: '#888' }}>Wet/Dry</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={reverb.wetDry}
                  onChange={(e) => updateReverb({ wetDry: Number(e.target.value) })}
                  disabled={!reverb.enabled}
                  style={{ width: '100%' }}
                />
                <div style={{ fontSize: '0.75em', textAlign: 'center' }}>{reverb.wetDry}%</div>
              </div>
            </div>

            {/* Reverb Presets */}
            <div style={{ marginTop: '20px' }}>
              <label style={{ fontSize: '0.8em', color: '#888', marginBottom: '8px', display: 'block' }}>
                Quick Presets
              </label>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {[
                  { name: 'Room', size: 30, damp: 60, wet: 15 },
                  { name: 'Hall', size: 70, damp: 40, wet: 25 },
                  { name: 'Plate', size: 50, damp: 30, wet: 20 },
                  { name: 'Cathedral', size: 90, damp: 20, wet: 35 },
                  { name: 'Ambient', size: 80, damp: 50, wet: 40 },
                ].map(preset => (
                  <button
                    key={preset.name}
                    onClick={() => updateReverb({
                      roomSize: preset.size,
                      damping: preset.damp,
                      wetDry: preset.wet,
                      enabled: true,
                    })}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: '#333',
                      border: 'none',
                      borderRadius: '4px',
                      color: '#fff',
                      cursor: 'pointer',
                      fontSize: '0.8em',
                    }}
                  >
                    {preset.name}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
