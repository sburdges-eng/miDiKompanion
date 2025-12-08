import React, { useState } from 'react';

interface EQBand {
  id: number;
  frequency: number;
  gain: number;
  q: number;
  type: 'lowshelf' | 'highshelf' | 'peaking' | 'lowpass' | 'highpass';
  enabled: boolean;
}

interface EQProps {
  channelId?: string;
  channelName?: string;
  onEQChange?: (bands: EQBand[]) => void;
}

const DEFAULT_BANDS: EQBand[] = [
  { id: 1, frequency: 80, gain: 0, q: 0.7, type: 'lowshelf', enabled: true },
  { id: 2, frequency: 250, gain: 0, q: 1.0, type: 'peaking', enabled: true },
  { id: 3, frequency: 1000, gain: 0, q: 1.0, type: 'peaking', enabled: true },
  { id: 4, frequency: 4000, gain: 0, q: 1.0, type: 'peaking', enabled: true },
  { id: 5, frequency: 12000, gain: 0, q: 0.7, type: 'highshelf', enabled: true },
];

const EQ_PRESETS: { [key: string]: Partial<EQBand>[] } = {
  flat: [
    { gain: 0 }, { gain: 0 }, { gain: 0 }, { gain: 0 }, { gain: 0 }
  ],
  warmth: [
    { gain: 3 }, { gain: 1 }, { gain: -1 }, { gain: 0 }, { gain: -2 }
  ],
  brightness: [
    { gain: -2 }, { gain: 0 }, { gain: 1 }, { gain: 3 }, { gain: 4 }
  ],
  vocal_presence: [
    { gain: -2 }, { gain: -1 }, { gain: 2 }, { gain: 4 }, { gain: 1 }
  ],
  bass_boost: [
    { gain: 6 }, { gain: 3 }, { gain: 0 }, { gain: 0 }, { gain: 0 }
  ],
  telephone: [
    { gain: -12 }, { gain: 0 }, { gain: 6 }, { gain: 0 }, { gain: -12 }
  ],
  air: [
    { gain: 0 }, { gain: 0 }, { gain: 0 }, { gain: 2 }, { gain: 5 }
  ],
  mud_cut: [
    { gain: 0 }, { gain: -4 }, { gain: 0 }, { gain: 0 }, { gain: 0 }
  ],
};

export const EQ: React.FC<EQProps> = ({
  channelId,
  channelName = 'Master',
  onEQChange,
}) => {
  const [bands, setBands] = useState<EQBand[]>(DEFAULT_BANDS);
  const [selectedBand, setSelectedBand] = useState<number | null>(null);
  const [bypass, setBypass] = useState(false);

  const updateBand = (id: number, changes: Partial<EQBand>) => {
    const newBands = bands.map(band =>
      band.id === id ? { ...band, ...changes } : band
    );
    setBands(newBands);
    onEQChange?.(newBands);
  };

  const applyPreset = (presetName: string) => {
    const preset = EQ_PRESETS[presetName];
    if (!preset) return;

    const newBands = bands.map((band, idx) => ({
      ...band,
      ...preset[idx],
    }));
    setBands(newBands);
    onEQChange?.(newBands);
  };

  const frequencyToX = (freq: number): number => {
    const minFreq = 20;
    const maxFreq = 20000;
    return ((Math.log10(freq) - Math.log10(minFreq)) / (Math.log10(maxFreq) - Math.log10(minFreq))) * 100;
  };

  const gainToY = (gain: number): number => {
    return 50 - (gain / 12) * 50;
  };

  return (
    <div style={{
      backgroundColor: '#1a1a1a',
      borderRadius: '8px',
      padding: '15px',
      color: '#fff',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '15px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontWeight: 'bold' }}>EQ - {channelName}</span>
          <button
            onClick={() => setBypass(!bypass)}
            style={{
              padding: '4px 8px',
              backgroundColor: bypass ? '#f44336' : '#4caf50',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '0.75em',
              cursor: 'pointer',
            }}
          >
            {bypass ? 'BYPASSED' : 'ACTIVE'}
          </button>
        </div>
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
          <option value="flat">Flat</option>
          <option value="warmth">Warmth</option>
          <option value="brightness">Brightness</option>
          <option value="vocal_presence">Vocal Presence</option>
          <option value="bass_boost">Bass Boost</option>
          <option value="telephone">Telephone</option>
          <option value="air">Air</option>
          <option value="mud_cut">Mud Cut</option>
        </select>
      </div>

      {/* EQ Curve Display */}
      <div style={{
        position: 'relative',
        height: '150px',
        backgroundColor: '#0f0f0f',
        borderRadius: '4px',
        marginBottom: '15px',
        overflow: 'hidden',
        opacity: bypass ? 0.5 : 1,
      }}>
        {/* Grid lines */}
        <svg width="100%" height="100%" style={{ position: 'absolute' }}>
          {/* Horizontal grid (gain) */}
          {[-12, -6, 0, 6, 12].map(gain => (
            <g key={gain}>
              <line
                x1="0"
                y1={`${gainToY(gain)}%`}
                x2="100%"
                y2={`${gainToY(gain)}%`}
                stroke={gain === 0 ? '#444' : '#222'}
                strokeWidth={gain === 0 ? 2 : 1}
              />
              <text
                x="5"
                y={`${gainToY(gain)}%`}
                fill="#666"
                fontSize="10"
                dominantBaseline="middle"
              >
                {gain > 0 ? '+' : ''}{gain}dB
              </text>
            </g>
          ))}
          {/* Vertical grid (frequency) */}
          {[100, 1000, 10000].map(freq => (
            <g key={freq}>
              <line
                x1={`${frequencyToX(freq)}%`}
                y1="0"
                x2={`${frequencyToX(freq)}%`}
                y2="100%"
                stroke="#222"
                strokeWidth="1"
              />
              <text
                x={`${frequencyToX(freq)}%`}
                y="95%"
                fill="#666"
                fontSize="10"
                textAnchor="middle"
              >
                {freq >= 1000 ? `${freq / 1000}k` : freq}
              </text>
            </g>
          ))}
        </svg>

        {/* Band points */}
        {bands.map(band => (
          <div
            key={band.id}
            onClick={() => setSelectedBand(selectedBand === band.id ? null : band.id)}
            style={{
              position: 'absolute',
              left: `${frequencyToX(band.frequency)}%`,
              top: `${gainToY(band.gain)}%`,
              width: '16px',
              height: '16px',
              borderRadius: '50%',
              backgroundColor: selectedBand === band.id ? '#6366f1' : band.enabled ? '#4caf50' : '#666',
              border: '2px solid #fff',
              transform: 'translate(-50%, -50%)',
              cursor: 'pointer',
              zIndex: 10,
              boxShadow: selectedBand === band.id ? '0 0 10px #6366f1' : 'none',
            }}
          />
        ))}
      </div>

      {/* Band Controls */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(5, 1fr)',
        gap: '10px',
      }}>
        {bands.map(band => (
          <div
            key={band.id}
            onClick={() => setSelectedBand(band.id)}
            style={{
              padding: '10px',
              backgroundColor: selectedBand === band.id ? '#2a2a4a' : '#222',
              borderRadius: '4px',
              border: selectedBand === band.id ? '2px solid #6366f1' : '2px solid transparent',
              cursor: 'pointer',
            }}
          >
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '8px',
            }}>
              <span style={{ fontSize: '0.75em', color: '#888' }}>Band {band.id}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  updateBand(band.id, { enabled: !band.enabled });
                }}
                style={{
                  width: '16px',
                  height: '16px',
                  borderRadius: '50%',
                  backgroundColor: band.enabled ? '#4caf50' : '#666',
                  border: 'none',
                  cursor: 'pointer',
                }}
              />
            </div>

            <div style={{ marginBottom: '6px' }}>
              <label style={{ fontSize: '0.7em', color: '#888' }}>Freq</label>
              <input
                type="range"
                min="20"
                max="20000"
                value={band.frequency}
                onChange={(e) => updateBand(band.id, { frequency: Number(e.target.value) })}
                style={{ width: '100%', height: '4px' }}
              />
              <div style={{ fontSize: '0.75em', textAlign: 'center' }}>
                {band.frequency >= 1000 ? `${(band.frequency / 1000).toFixed(1)}k` : band.frequency}Hz
              </div>
            </div>

            <div style={{ marginBottom: '6px' }}>
              <label style={{ fontSize: '0.7em', color: '#888' }}>Gain</label>
              <input
                type="range"
                min="-12"
                max="12"
                step="0.5"
                value={band.gain}
                onChange={(e) => updateBand(band.id, { gain: Number(e.target.value) })}
                style={{ width: '100%', height: '4px' }}
              />
              <div style={{ fontSize: '0.75em', textAlign: 'center' }}>
                {band.gain > 0 ? '+' : ''}{band.gain}dB
              </div>
            </div>

            <div>
              <label style={{ fontSize: '0.7em', color: '#888' }}>Q</label>
              <input
                type="range"
                min="0.1"
                max="10"
                step="0.1"
                value={band.q}
                onChange={(e) => updateBand(band.id, { q: Number(e.target.value) })}
                style={{ width: '100%', height: '4px' }}
              />
              <div style={{ fontSize: '0.75em', textAlign: 'center' }}>
                {band.q.toFixed(1)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
