import React from 'react';
import { AudioPreview } from './AudioPreview';

interface MidiPlayerProps {
  midiData?: string | null;
  midiPath?: string | null;
  musicConfig?: {
    key?: string;
    mode?: string;
    tempo?: number;
    progression?: string;
  } | null;
}

export const MidiPlayer: React.FC<MidiPlayerProps> = ({
  midiData,
  midiPath,
  musicConfig,
}) => {
  if (!midiData && !midiPath) {
    return null;
  }

  // Calculate file size from base64 data
  const fileSize = midiData
    ? `${Math.round((midiData.length * 3) / 4 / 1024)} KB`
    : 'Unknown';

  const filename = midiPath
    ? midiPath.split('/').pop() || 'generated_music.mid'
    : 'generated_music.mid';

  return (
    <div
      style={{
        marginTop: '10px',
        padding: '15px',
        backgroundColor: 'rgba(76, 175, 80, 0.1)',
        borderRadius: '4px',
        border: '1px solid #4caf50',
      }}
    >
      <div style={{ marginBottom: '10px' }}>
        <strong>✅ MIDI Generated</strong>
      </div>

      <div style={{ fontSize: '0.9em', marginBottom: '8px' }}>
        <div>
          <strong>Filename:</strong> {filename}
        </div>
        <div>
          <strong>Size:</strong> {fileSize}
        </div>
        {musicConfig && (
          <>
            <div>
              <strong>Key:</strong> {musicConfig.key || 'N/A'}{' '}
              {musicConfig.mode || ''}
            </div>
            <div>
              <strong>Tempo:</strong> {musicConfig.tempo || 'N/A'} BPM
            </div>
            {musicConfig.progression && (
              <div>
                <strong>Progression:</strong> {musicConfig.progression}
              </div>
            )}
          </>
        )}
      </div>

      {midiPath && (
        <div
          style={{
            marginTop: '5px',
            fontSize: '0.8em',
            fontFamily: 'monospace',
            wordBreak: 'break-all',
            color: '#666',
          }}
        >
          Path: {midiPath}
        </div>
      )}

      {midiData && (
        <div style={{ marginTop: '15px' }}>
          <AudioPreview midiData={midiData} />
        </div>
      )}
    </div>
  );
};
