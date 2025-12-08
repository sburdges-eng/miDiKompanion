import { useState } from 'react';
import { ChannelStrip } from './ChannelStrip';

interface MixerChannel {
  id: string;
  name: string;
  volume: number;
  pan: number;
  muted: boolean;
  solo: boolean;
}

interface MixerProps {
  channels?: MixerChannel[];
  onChannelChange?: (channelId: string, changes: Partial<MixerChannel>) => void;
}

export const Mixer: React.FC<MixerProps> = ({
  channels = [],
  onChannelChange,
}) => {
  const [masterVolume, setMasterVolume] = useState(0.8);

  // Default channels if none provided
  const defaultChannels: MixerChannel[] = [
    { id: '1', name: 'Drums', volume: 0.8, pan: 0, muted: false, solo: false },
    { id: '2', name: 'Bass', volume: 0.75, pan: 0, muted: false, solo: false },
    { id: '3', name: 'Chords', volume: 0.7, pan: 0, muted: false, solo: false },
    { id: '4', name: 'Melody', volume: 0.75, pan: 0, muted: false, solo: false },
    { id: '5', name: 'Track 5', volume: 0.7, pan: 0, muted: false, solo: false },
    { id: '6', name: 'Track 6', volume: 0.7, pan: 0, muted: false, solo: false },
    { id: '7', name: 'Track 7', volume: 0.7, pan: 0, muted: false, solo: false },
    { id: '8', name: 'Track 8', volume: 0.7, pan: 0, muted: false, solo: false },
  ];

  const displayChannels = channels.length > 0 ? channels : defaultChannels;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      backgroundColor: '#1a1a1a',
      borderLeft: '1px solid rgba(255, 255, 255, 0.1)'
    }}>
      {/* Mixer header */}
      <div style={{
        padding: '12px',
        backgroundColor: '#0f0f0f',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        fontSize: '0.9em',
        fontWeight: 'bold',
        color: '#fff',
        textAlign: 'center'
      }}>
        Mixer
      </div>

      {/* Channel strips */}
      <div style={{
        flex: 1,
        display: 'flex',
        overflowX: 'auto',
        overflowY: 'hidden'
      }}>
        {displayChannels.map((channel) => (
          <ChannelStrip
            key={channel.id}
            channel={parseInt(channel.id)}
            name={channel.name}
            volume={channel.volume}
            pan={channel.pan}
            muted={channel.muted}
            solo={channel.solo}
            onVolumeChange={(volume) => onChannelChange?.(channel.id, { volume })}
            onPanChange={(pan) => onChannelChange?.(channel.id, { pan })}
            onMute={() => onChannelChange?.(channel.id, { muted: !channel.muted })}
            onSolo={() => onChannelChange?.(channel.id, { solo: !channel.solo })}
          />
        ))}
      </div>

      {/* Master fader */}
      <div style={{
        padding: '12px',
        backgroundColor: '#0f0f0f',
        borderTop: '2px solid rgba(255, 255, 255, 0.2)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px'
      }}>
        <div style={{
          fontSize: '0.8em',
          color: '#fff',
          fontWeight: 'bold'
        }}>
          Master
        </div>
        <div
          style={{
            width: '30px',
            height: '200px',
            backgroundColor: '#1a1a1a',
            border: '2px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '4px',
            position: 'relative',
            cursor: 'pointer'
          }}
          onMouseDown={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const y = e.clientY - rect.top;
            const newVolume = 1 - (y / rect.height);
            setMasterVolume(Math.max(0, Math.min(1, newVolume)));
          }}
        >
          <div style={{
            position: 'absolute',
            bottom: `${masterVolume * 100}%`,
            left: '50%',
            transform: 'translateX(-50%)',
            width: '34px',
            height: '12px',
            backgroundColor: '#4caf50',
            border: '2px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '2px',
            cursor: 'grab',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)'
          }} />
        </div>
        <div style={{
          fontSize: '0.75em',
          color: '#888'
        }}>
          {Math.round(masterVolume * 100)}%
        </div>
      </div>
    </div>
  );
};
