import React, { useState } from 'react';
import { RecordingEngine, Track, RecordingConfig } from './RecordingEngine';

interface RecordingControlsProps {
  engine: RecordingEngine;
  tracks: Track[];
  onTracksChange?: (tracks: Track[]) => void;
}

export const RecordingControls: React.FC<RecordingControlsProps> = ({
  engine,
  tracks,
  onTracksChange,
}) => {
  const [selectedTrack, setSelectedTrack] = useState<string | null>(tracks[0]?.id || null);
  const [recordingMode, setRecordingMode] = useState<
    'normal' | 'punch' | 'auto-punch' | 'loop' | 'take' | 'quick-punch'
  >('normal');
  const [punchInTime, setPunchInTime] = useState<string>('');
  const [punchOutTime, setPunchOutTime] = useState<string>('');
  const [preRoll, setPreRoll] = useState<number>(2);
  const [postRoll, setPostRoll] = useState<number>(1);
  const [loopStart, setLoopStart] = useState<string>('');
  const [loopEnd, setLoopEnd] = useState<string>('');
  const [autoIncrement, setAutoIncrement] = useState<boolean>(true);
  const [retrospectiveEnabled, setRetrospectiveEnabled] = useState<boolean>(true);
  const [voiceActivated, setVoiceActivated] = useState<boolean>(false);
  const [autoRecordThreshold, setAutoRecordThreshold] = useState<number>(0.01);

  const selectedTrackObj = tracks.find((t) => t.id === selectedTrack);

  const handleStartRecording = async () => {
    if (!selectedTrack) return;

    try {
      const config: Partial<RecordingConfig> = {
        channels: 2, // Stereo
        sampleRate: 44100,
        bitDepth: 24,
        format: 'wav',
      };

      switch (recordingMode) {
        case 'normal':
          await engine.startRecording(selectedTrack, config);
          break;
        case 'punch':
          if (punchInTime) {
            await engine.punchIn(selectedTrack, parseFloat(punchInTime));
          }
          break;
        case 'auto-punch':
          if (punchInTime && punchOutTime) {
            engine.setAutoPunch(parseFloat(punchInTime), parseFloat(punchOutTime));
            await engine.startRecording(selectedTrack, config);
          }
          break;
        case 'loop':
          if (loopStart && loopEnd) {
            engine.startLoopRecording(
              selectedTrack,
              parseFloat(loopStart),
              parseFloat(loopEnd),
              autoIncrement
            );
          }
          break;
        case 'quick-punch':
          // Quick punch is like normal but seamless
          await engine.startRecording(selectedTrack, config);
          break;
        case 'take':
          // Take recording - creates new take
          engine.createNewTake(selectedTrack);
          await engine.startRecording(selectedTrack, config);
          break;
      }

      if (voiceActivated) {
        await engine.startVoiceActivated(selectedTrack, autoRecordThreshold);
      }
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert(`Recording failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  const handleStopRecording = () => {
    engine.stopRecording();
  };

  const handlePunchOut = () => {
    if (punchOutTime) {
      engine.punchOut(parseFloat(punchOutTime));
    } else {
      engine.stopRecording();
    }
  };

  const handleCaptureRetrospective = (duration: number) => {
    const buffer = engine.captureRetrospective(duration);
    if (buffer && selectedTrack) {
      // Add as new take
      engine.createNewTake(selectedTrack);
      const track = tracks.find((t) => t.id === selectedTrack);
      if (track) {
        const take = track.takes[track.takes.length - 1];
        if (take) {
          take.audioBuffer = buffer;
        }
      }
      onTracksChange?.(engine.getTracks());
    }
  };

  return (
    <div
      style={{
        padding: '20px',
        backgroundColor: '#1a1a1a',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <h3 style={{ marginTop: 0, color: '#fff' }}>Recording Controls</h3>

      {/* Track Selection */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
          Track
        </label>
        <select
          value={selectedTrack || ''}
          onChange={(e) => setSelectedTrack(e.target.value)}
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
          }}
        >
          {tracks.map((track) => (
            <option key={track.id} value={track.id}>
              {track.name} {track.armed ? '⚡' : ''} {track.recordSafe ? '🔒' : ''}
            </option>
          ))}
        </select>
      </div>

      {/* Recording Mode */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
          Recording Mode
        </label>
        <select
          value={recordingMode}
          onChange={(e) => setRecordingMode(e.target.value as any)}
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
          }}
        >
          <option value="normal">Normal Recording</option>
          <option value="punch">Punch In/Out (Manual)</option>
          <option value="auto-punch">Auto Punch In/Out</option>
          <option value="loop">Loop Recording</option>
          <option value="take">Take Recording</option>
          <option value="quick-punch">Quick Punch</option>
        </select>
      </div>

      {/* Punch In/Out Times */}
      {(recordingMode === 'punch' || recordingMode === 'auto-punch') && (
        <div style={{ marginBottom: '15px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
          <div>
            <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
              Punch In Time (seconds)
            </label>
            <input
              type="number"
              value={punchInTime}
              onChange={(e) => setPunchInTime(e.target.value)}
              step="0.1"
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#2a2a2a',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
              }}
            />
          </div>
          <div>
            <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
              Punch Out Time (seconds)
            </label>
            <input
              type="number"
              value={punchOutTime}
              onChange={(e) => setPunchOutTime(e.target.value)}
              step="0.1"
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#2a2a2a',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
              }}
            />
          </div>
        </div>
      )}

      {/* Loop Recording */}
      {recordingMode === 'loop' && (
        <div style={{ marginBottom: '15px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
          <div>
            <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
              Loop Start (seconds)
            </label>
            <input
              type="number"
              value={loopStart}
              onChange={(e) => setLoopStart(e.target.value)}
              step="0.1"
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#2a2a2a',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
              }}
            />
          </div>
          <div>
            <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
              Loop End (seconds)
            </label>
            <input
              type="number"
              value={loopEnd}
              onChange={(e) => setLoopEnd(e.target.value)}
              step="0.1"
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: '#2a2a2a',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
              }}
            />
          </div>
        </div>
      )}

      {/* Pre/Post Roll */}
      <div style={{ marginBottom: '15px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
        <div>
          <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
            Pre-Roll (seconds)
          </label>
          <input
            type="number"
            value={preRoll}
            onChange={(e) => setPreRoll(parseFloat(e.target.value))}
            step="0.1"
            min="0"
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          />
        </div>
        <div>
          <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
            Post-Roll (seconds)
          </label>
          <input
            type="number"
            value={postRoll}
            onChange={(e) => setPostRoll(parseFloat(e.target.value))}
            step="0.1"
            min="0"
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          />
        </div>
      </div>

      {/* Advanced Options */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#aaa', fontSize: '0.9em' }}>
          <input
            type="checkbox"
            checked={retrospectiveEnabled}
            onChange={(e) => setRetrospectiveEnabled(e.target.checked)}
          />
          Retrospective Recording (Capture Buffer)
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#aaa', fontSize: '0.9em', marginTop: '8px' }}>
          <input
            type="checkbox"
            checked={voiceActivated}
            onChange={(e) => setVoiceActivated(e.target.checked)}
          />
          Voice-Activated Recording
        </label>
        {voiceActivated && (
          <div style={{ marginTop: '8px', marginLeft: '24px' }}>
            <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '4px' }}>
              Sensitivity Threshold
            </label>
            <input
              type="range"
              min="0.001"
              max="0.1"
              step="0.001"
              value={autoRecordThreshold}
              onChange={(e) => setAutoRecordThreshold(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
            <span style={{ color: '#888', fontSize: '0.8em' }}>{autoRecordThreshold.toFixed(3)}</span>
          </div>
        )}
        {recordingMode === 'loop' && (
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#aaa', fontSize: '0.9em', marginTop: '8px' }}>
            <input
              type="checkbox"
              checked={autoIncrement}
              onChange={(e) => setAutoIncrement(e.target.checked)}
            />
            Auto-Increment Takes
          </label>
        )}
      </div>

      {/* Control Buttons */}
      <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
        <button
          onClick={handleStartRecording}
          style={{
            flex: 1,
            padding: '12px',
            backgroundColor: '#f44336',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '1em',
          }}
        >
          ⏺ Start Recording
        </button>
        <button
          onClick={recordingMode === 'punch' ? handlePunchOut : handleStopRecording}
          style={{
            flex: 1,
            padding: '12px',
            backgroundColor: '#333',
            color: 'white',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            cursor: 'pointer',
            fontWeight: 'bold',
            fontSize: '1em',
          }}
        >
          ⏹ Stop Recording
        </button>
      </div>

      {/* Retrospective Capture */}
      {retrospectiveEnabled && (
        <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px' }}>
          <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
            Capture Retrospective Recording (seconds)
          </label>
          <div style={{ display: 'flex', gap: '10px' }}>
            <input
              type="number"
              defaultValue="5"
              min="1"
              max="30"
              step="1"
              id="retrospective-duration"
              style={{
                flex: 1,
                padding: '8px',
                backgroundColor: '#1a1a1a',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '4px',
                color: '#fff',
              }}
            />
            <button
              onClick={() => {
                const duration = parseFloat((document.getElementById('retrospective-duration') as HTMLInputElement).value);
                handleCaptureRetrospective(duration);
              }}
              style={{
                padding: '8px 16px',
                backgroundColor: '#6366f1',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Capture
            </button>
          </div>
        </div>
      )}

      {/* Track Info */}
      {selectedTrackObj && (
        <div style={{ marginTop: '15px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px', fontSize: '0.85em' }}>
          <div style={{ color: '#aaa' }}>
            <div>Takes: {selectedTrackObj.takes.length}</div>
            <div>Current Take: {selectedTrackObj.currentTake + 1}</div>
            <div>Monitoring: {selectedTrackObj.inputMonitoring}</div>
            <div>Armed: {selectedTrackObj.armed ? 'Yes' : 'No'}</div>
            <div>Record Safe: {selectedTrackObj.recordSafe ? 'Yes' : 'No'}</div>
          </div>
        </div>
      )}
    </div>
  );
};
