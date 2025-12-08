import React, { useState, useEffect, useRef } from 'react';
import { RecordingEngine, Track, RecordingConfig } from './RecordingEngine';
import { VUMeter } from './VUMeter';
import { WaveformVisualizer } from './WaveformVisualizer';

interface RecordingStudioProps {
  engine: RecordingEngine;
  tracks: Track[];
  onTracksChange: (tracks: Track[]) => void;
}

export const RecordingStudio: React.FC<RecordingStudioProps> = ({
  engine,
  tracks,
  onTracksChange,
}) => {
  const [selectedTracks, setSelectedTracks] = useState<Set<string>>(new Set());
  const [recordingConfig, setRecordingConfig] = useState<RecordingConfig>({
    channels: 2, // Stereo by default
    sampleRate: 44100,
    bitDepth: 24,
    format: 'wav',
  });
  const [punchMode, setPunchMode] = useState<'manual' | 'auto' | 'off'>('off');
  const [punchInTime, setPunchInTime] = useState<number>(0);
  const [punchOutTime, setPunchOutTime] = useState<number>(16);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingStartTime, setRecordingStartTime] = useState<number>(0);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // Initialize audio context
  useEffect(() => {
    const initAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStreamRef.current = stream;
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioContextRef.current = ctx;
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 2048;
        const source = ctx.createMediaStreamSource(stream);
        source.connect(analyser);
        analyserRef.current = analyser;
      } catch (error) {
        console.error('Failed to initialize audio:', error);
      }
    };

    initAudio();

    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Time update loop
  useEffect(() => {
    if (!isRecording) return;

    const interval = setInterval(() => {
      setCurrentTime((prev) => prev + 0.1);
    }, 100);

    return () => clearInterval(interval);
  }, [isRecording]);

  const toggleTrackSelection = (trackId: string) => {
    const newSelected = new Set(selectedTracks);
    if (newSelected.has(trackId)) {
      newSelected.delete(trackId);
    } else {
      newSelected.add(trackId);
    }
    setSelectedTracks(newSelected);
  };

  const handleStartRecording = async () => {
    if (selectedTracks.size === 0) {
      alert('Please select at least one track to record');
      return;
    }

    try {
      await engine.initialize();

      // Start recording on all selected tracks
      for (const trackId of selectedTracks) {
        const track = tracks.find((t) => t.id === trackId);
        if (!track || track.recordSafe || !track.armed) continue;

        if (punchMode === 'manual') {
          // Feature 4: Manual punch-in
          await engine.punchIn(trackId, punchInTime);
        } else if (punchMode === 'auto') {
          // Feature 6: Auto punch-in/out
          engine.setAutoPunch(punchInTime, punchOutTime);
          await engine.startRecording(trackId, recordingConfig);
        } else {
          // Features 1-3: Normal recording (mono/stereo/multi-track)
          await engine.startRecording(trackId, recordingConfig);
        }
      }

      setIsRecording(true);
      setRecordingStartTime(currentTime);
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert(`Recording failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleStopRecording = () => {
    if (punchMode === 'manual') {
      // Feature 5: Manual punch-out
      engine.punchOut(punchOutTime);
    } else {
      engine.stopRecording();
    }
    setIsRecording(false);
    onTracksChange(engine.getTracks());
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        backgroundColor: '#1a1a1a',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <h3 style={{ margin: '0 0 15px 0', color: '#fff' }}>Recording Studio</h3>

        {/* Recording Mode Selection */}
        <div style={{ display: 'flex', gap: '10px', marginBottom: '15px', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
              Recording Mode
            </label>
            <select
              value={punchMode}
              onChange={(e) => setPunchMode(e.target.value as any)}
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#2a2a2a',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                fontSize: '0.9em',
              }}
            >
              <option value="off">Normal Recording</option>
              <option value="manual">Manual Punch-In/Out (Feature 4-5)</option>
              <option value="auto">Auto Punch-In/Out (Feature 6)</option>
            </select>
          </div>

          {/* Channel Configuration */}
          <div style={{ flex: 1, minWidth: '200px' }}>
            <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
              Channels (Feature 1-3)
            </label>
            <select
              value={recordingConfig.channels}
              onChange={(e) =>
                setRecordingConfig({ ...recordingConfig, channels: Number(e.target.value) })
              }
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#2a2a2a',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
                fontSize: '0.9em',
              }}
            >
              <option value="1">1. Mono Recording</option>
              <option value="2">2. Stereo Recording</option>
              <option value="4">3. Multi-track (4 channels)</option>
              <option value="8">3. Multi-track (8 channels)</option>
              <option value="16">3. Multi-track (16 channels)</option>
            </select>
          </div>
        </div>

        {/* Punch-In/Out Times */}
        {punchMode !== 'off' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '15px' }}>
            <div>
              <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
                Punch-In Time (seconds)
              </label>
              <input
                type="number"
                value={punchInTime}
                onChange={(e) => setPunchInTime(parseFloat(e.target.value))}
                step="0.1"
                min="0"
                style={{
                  width: '100%',
                  padding: '8px',
                  backgroundColor: '#2a2a2a',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '4px',
                  color: '#fff',
                  fontSize: '0.9em',
                }}
              />
            </div>
            <div>
              <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
                Punch-Out Time (seconds)
              </label>
              <input
                type="number"
                value={punchOutTime}
                onChange={(e) => setPunchOutTime(parseFloat(e.target.value))}
                step="0.1"
                min="0"
                style={{
                  width: '100%',
                  padding: '8px',
                  backgroundColor: '#2a2a2a',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '4px',
                  color: '#fff',
                  fontSize: '0.9em',
                }}
              />
            </div>
          </div>
        )}

        {/* Recording Controls */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            onClick={isRecording ? handleStopRecording : handleStartRecording}
            style={{
              padding: '12px 24px',
              backgroundColor: isRecording ? '#f44336' : '#4caf50',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '1em',
              fontWeight: 'bold',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            {isRecording ? (
              <>
                <span>⏹</span> Stop Recording
              </>
            ) : (
              <>
                <span>⏺</span> Start Recording
              </>
            )}
          </button>

          {isRecording && (
            <div
              style={{
                padding: '8px 16px',
                backgroundColor: '#f4433620',
                border: '1px solid #f44336',
                borderRadius: '4px',
                color: '#f44336',
                fontSize: '0.9em',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <span
                style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  backgroundColor: '#f44336',
                  animation: 'pulse 1s infinite',
                }}
              />
              Recording: {formatTime(currentTime - recordingStartTime)}
            </div>
          )}
        </div>
      </div>

      {/* Waveform Visualizer */}
      {analyserRef.current && (
        <div style={{ padding: '15px', borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <WaveformVisualizer
            audioSource={analyserRef.current as any}
            width={800}
            height={150}
            color="#f44336"
            syncToPlayback={isRecording}
            isPlaying={isRecording}
          />
        </div>
      )}

      {/* Track Selection */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '15px' }}>
        <div style={{ color: '#fff', fontWeight: 'bold', marginBottom: '15px' }}>
          Select Tracks to Record ({selectedTracks.size} selected)
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {tracks.map((track) => {
            const isSelected = selectedTracks.has(track.id);
            const canRecord = track.armed && !track.recordSafe;

            return (
              <div
                key={track.id}
                onClick={() => toggleTrackSelection(track.id)}
                style={{
                  padding: '15px',
                  backgroundColor: isSelected ? '#6366f120' : '#2a2a2a',
                  border: `2px solid ${isSelected ? '#6366f1' : 'rgba(255, 255, 255, 0.1)'}`,
                  borderRadius: '4px',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '15px',
                }}
              >
                {/* Selection Checkbox */}
                <div
                  style={{
                    width: '24px',
                    height: '24px',
                    borderRadius: '4px',
                    border: '2px solid #6366f1',
                    backgroundColor: isSelected ? '#6366f1' : 'transparent',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#fff',
                    fontSize: '0.9em',
                    fontWeight: 'bold',
                  }}
                >
                  {isSelected && '✓'}
                </div>

                {/* Track Info */}
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                    <span style={{ color: '#fff', fontWeight: 'bold', fontSize: '1em' }}>{track.name}</span>
                    {track.armed && (
                      <span
                        style={{
                          padding: '2px 8px',
                          backgroundColor: '#4caf50',
                          borderRadius: '4px',
                          fontSize: '0.75em',
                          color: '#fff',
                          fontWeight: 'bold',
                        }}
                      >
                        ⚡ ARMED
                      </span>
                    )}
                    {track.recordSafe && (
                      <span
                        style={{
                          padding: '2px 8px',
                          backgroundColor: '#f44336',
                          borderRadius: '4px',
                          fontSize: '0.75em',
                          color: '#fff',
                          fontWeight: 'bold',
                        }}
                      >
                        🔒 SAFE
                      </span>
                    )}
                    {!canRecord && (
                      <span style={{ color: '#f44336', fontSize: '0.85em' }}>
                        {track.recordSafe ? 'Record Safe' : 'Not Armed'}
                      </span>
                    )}
                  </div>

                  {/* Channel Configuration Preview */}
                  <div style={{ fontSize: '0.85em', color: '#aaa' }}>
                    {recordingConfig.channels === 1 && 'Mono (1 channel)'}
                    {recordingConfig.channels === 2 && 'Stereo (2 channels)'}
                    {recordingConfig.channels > 2 && `Multi-track (${recordingConfig.channels} channels)`}
                    {' • '}
                    {recordingConfig.sampleRate / 1000}kHz • {recordingConfig.bitDepth}-bit
                  </div>
                </div>

                {/* VU Meter */}
                <VUMeter level={0.3 + Math.random() * 0.4} label={track.id} />

                {/* Recording Status */}
                {isRecording && isSelected && (
                  <div
                    style={{
                      padding: '6px 12px',
                      backgroundColor: '#f44336',
                      borderRadius: '4px',
                      color: '#fff',
                      fontSize: '0.85em',
                      fontWeight: 'bold',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                    }}
                  >
                    <span
                      style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: '#fff',
                        animation: 'pulse 1s infinite',
                      }}
                    />
                    REC
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {tracks.length === 0 && (
          <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
            No tracks available. Add tracks to start recording.
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div
        style={{
          padding: '10px 15px',
          backgroundColor: '#0f0f0f',
          borderTop: '1px solid rgba(255, 255, 255, 0.1)',
          fontSize: '0.85em',
          color: '#aaa',
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <div>
          Mode: {punchMode === 'off' ? 'Normal' : punchMode === 'manual' ? 'Manual Punch' : 'Auto Punch'}
        </div>
        <div>
          Channels: {recordingConfig.channels} ({recordingConfig.channels === 1 ? 'Mono' : recordingConfig.channels === 2 ? 'Stereo' : 'Multi-track'})
        </div>
        <div>Selected: {selectedTracks.size} track{selectedTracks.size !== 1 ? 's' : ''}</div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
};
