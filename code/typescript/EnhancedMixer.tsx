import React, { useState, useEffect } from 'react';
import { AdvancedSlider } from './AdvancedSlider';
import { VUMeter } from './VUMeter';
import { WaveformVisualizer } from './WaveformVisualizer';

interface Channel {
  id: string;
  name: string;
  volume: number;
  pan: number;
  mute: boolean;
  solo: boolean;
  send1: number;
  send2: number;
  lowCut: number;
  highCut: number;
}

interface EnhancedMixerProps {
  channels?: Channel[];
  onChannelChange?: (channelId: string, changes: Partial<Channel>) => void;
  showWaveform?: boolean;
}

export const EnhancedMixer: React.FC<EnhancedMixerProps> = ({
  channels = [],
  onChannelChange,
  showWaveform = true,
}) => {
  const [masterVolume, setMasterVolume] = useState(0.8);
  const [masterVuLevel, setMasterVuLevel] = useState(0.3);

  // Default channels
  const defaultChannels: Channel[] = [
    { id: '1', name: 'Drums', volume: 0.8, pan: 0, mute: false, solo: false, send1: 0, send2: 0, lowCut: 20, highCut: 20000 },
    { id: '2', name: 'Bass', volume: 0.75, pan: 0, mute: false, solo: false, send1: 0, send2: 0, lowCut: 20, highCut: 20000 },
    { id: '3', name: 'Chords', volume: 0.7, pan: 0, mute: false, solo: false, send1: 0.3, send2: 0, lowCut: 20, highCut: 20000 },
    { id: '4', name: 'Melody', volume: 0.75, pan: 0, mute: false, solo: false, send1: 0.2, send2: 0, lowCut: 20, highCut: 20000 },
    { id: '5', name: 'Vocal', volume: 0.8, pan: 0, mute: false, solo: false, send1: 0.4, send2: 0.2, lowCut: 80, highCut: 15000 },
  ];

  const displayChannels = channels.length > 0 ? channels : defaultChannels;

  // Simulate VU meter updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMasterVuLevel(0.2 + Math.random() * 0.6);
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: '#1a1a1a',
      }}
    >
      {/* Waveform visualizer */}
      {showWaveform && (
        <div style={{ padding: '10px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <WaveformVisualizer
            width={800}
            height={120}
            color="#6366f1"
            syncToPlayback={true}
            isPlaying={true}
          />
        </div>
      )}

      {/* Channel strips */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          overflowX: 'auto',
          overflowY: 'hidden',
          padding: '10px',
          gap: '8px',
        }}
      >
        {displayChannels.map((channel) => (
          <ChannelStripEnhanced
            key={channel.id}
            channel={channel}
            onChange={(changes) => onChannelChange?.(channel.id, changes)}
          />
        ))}
      </div>

      {/* Master section */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderTop: '2px solid rgba(255, 255, 255, 0.2)',
          display: 'flex',
          alignItems: 'center',
          gap: '20px',
        }}
      >
        <div style={{ fontSize: '0.9em', color: '#fff', fontWeight: 'bold', minWidth: '60px' }}>
          Master
        </div>

        <VUMeter level={masterVuLevel} />

        <AdvancedSlider
          value={masterVolume}
          min={0}
          max={1}
          step={0.01}
          orientation="vertical"
          width={40}
          height={150}
          onChange={setMasterVolume}
          showValue={true}
          unit="%"
          color="#4caf50"
        />

        <div
          style={{
            flex: 1,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            color: '#888',
            fontSize: '0.85em',
          }}
        >
          <div>Output: {Math.round(masterVolume * 100)}%</div>
          <div>Peak: {Math.round(masterVuLevel * 100)}%</div>
        </div>
      </div>
    </div>
  );
};

interface ChannelStripEnhancedProps {
  channel: Channel;
  onChange: (changes: Partial<Channel>) => void;
}

const ChannelStripEnhanced: React.FC<ChannelStripEnhancedProps> = ({
  channel,
  onChange,
}) => {
  const [vuLevel, setVuLevel] = useState(0.3 + Math.random() * 0.4);

  useEffect(() => {
    const interval = setInterval(() => {
      setVuLevel(0.2 + Math.random() * 0.6);
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <div
      style={{
        width: '80px',
        height: '100%',
        backgroundColor: '#1a1a1a',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        borderRadius: '4px',
        padding: '10px 8px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '10px',
      }}
    >
      {/* Channel name */}
      <div
        style={{
          fontSize: '0.75em',
          color: '#fff',
          fontWeight: 'bold',
          textAlign: 'center',
          width: '100%',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {channel.name}
      </div>

      {/* VU Meter */}
      <VUMeter level={vuLevel} label={channel.id} />

      {/* Volume fader */}
      <AdvancedSlider
        value={channel.volume}
        min={0}
        max={1}
        step={0.01}
        orientation="vertical"
        width={30}
        height={200}
        onChange={(val) => onChange({ volume: val })}
        showValue={true}
        unit="%"
        color={channel.mute ? '#666' : '#6366f1'}
      />

      {/* Pan control */}
      <div style={{ width: '100%' }}>
        <div
          style={{
            fontSize: '0.7em',
            color: '#888',
            marginBottom: '4px',
            textAlign: 'center',
          }}
        >
          Pan
        </div>
        <AdvancedSlider
          value={channel.pan}
          min={-1}
          max={1}
          step={0.01}
          orientation="horizontal"
          width={60}
          height={30}
          onChange={(val) => onChange({ pan: val })}
          showValue={false}
          color="#888"
        />
        <div
          style={{
            fontSize: '0.65em',
            color: '#666',
            display: 'flex',
            justifyContent: 'space-between',
            padding: '0 4px',
            marginTop: '2px',
          }}
        >
          <span>L</span>
          <span>C</span>
          <span>R</span>
        </div>
      </div>

      {/* Sends */}
      <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <div style={{ fontSize: '0.65em', color: '#888', textAlign: 'center' }}>Sends</div>
        <AdvancedSlider
          value={channel.send1}
          min={0}
          max={1}
          step={0.01}
          orientation="vertical"
          width={20}
          height={60}
          onChange={(val) => onChange({ send1: val })}
          showValue={false}
          color="#ff9800"
        />
        <AdvancedSlider
          value={channel.send2}
          min={0}
          max={1}
          step={0.01}
          orientation="vertical"
          width={20}
          height={60}
          onChange={(val) => onChange({ send2: val })}
          showValue={false}
          color="#9c27b0"
        />
      </div>

      {/* Mute/Solo buttons */}
      <div style={{ display: 'flex', gap: '4px', width: '100%' }}>
        <button
          onClick={() => onChange({ mute: !channel.mute })}
          style={{
            flex: 1,
            padding: '6px',
            backgroundColor: channel.mute ? '#f44336' : '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '0.7em',
            fontWeight: 'bold',
          }}
          title="Mute"
        >
          M
        </button>
        <button
          onClick={() => onChange({ solo: !channel.solo })}
          style={{
            flex: 1,
            padding: '6px',
            backgroundColor: channel.solo ? '#ffeb3b' : '#333',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: channel.solo ? '#000' : '#fff',
            cursor: 'pointer',
            fontSize: '0.7em',
            fontWeight: 'bold',
          }}
          title="Solo"
        >
          S
        </button>
      </div>
    </div>
  );
};
