// TempoControls - Features 81-94: Tempo & Time Signature

import React, { useState, useEffect } from 'react';
import { TransportEngine, TempoEvent } from './TransportEngine';

interface TempoControlsProps {
  engine: TransportEngine;
  onTempoChange?: () => void;
}

export const TempoControls: React.FC<TempoControlsProps> = ({
  engine,
  onTempoChange,
}) => {
  const [state, setState] = useState(engine.getState());
  const [tempoInput, setTempoInput] = useState(state.tempo.toString());
  const [timeSigNum, setTimeSigNum] = useState(state.timeSignature[0]);
  const [timeSigDen, setTimeSigDen] = useState(state.timeSignature[1]);
  const [tapTimes, setTapTimes] = useState<number[]>([]);
  const [nudgeAmount, setNudgeAmount] = useState(0.1);

  useEffect(() => {
    const interval = setInterval(() => {
      setState(engine.getState());
    }, 100);
    return () => clearInterval(interval);
  }, [engine]);

  // Feature 81: Set Global Tempo
  const handleTempoChange = (bpm: number) => {
    engine.setTempo(bpm);
    setTempoInput(bpm.toString());
    onTempoChange?.();
  };

  // Feature 89: Set Time Signature
  const handleTimeSignatureChange = (num: number, den: number) => {
    engine.setTimeSignature(num, den);
    setTimeSigNum(num);
    setTimeSigDen(den);
    onTempoChange?.();
  };

  // Feature 87: Tap Tempo
  const handleTapTempo = () => {
    const now = Date.now();
    const newTapTimes = [...tapTimes, now].filter((time) => now - time < 2000); // Keep taps within 2 seconds
    setTapTimes(newTapTimes);

    if (newTapTimes.length >= 2) {
      const intervals = [];
      for (let i = 1; i < newTapTimes.length; i++) {
        intervals.push(newTapTimes[i] - newTapTimes[i - 1]);
      }
      const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const bpm = Math.round(60000 / avgInterval);
      if (bpm >= 30 && bpm <= 300) {
        handleTempoChange(bpm);
      }
    }
  };

  // Feature 88: Tempo Nudge
  const handleNudgeTempo = (direction: 'up' | 'down') => {
    const change = direction === 'up' ? nudgeAmount : -nudgeAmount;
    const newTempo = Math.max(30, Math.min(300, state.tempo + change));
    handleTempoChange(newTempo);
  };

  // Create Tempo Event
  const createTempoEvent = () => {
    const event: TempoEvent = {
      time: state.position.totalSeconds as any,
      tempo: state.tempo,
      timeSignature: state.timeSignature,
    };
    const newEvents = [...state.tempoTrack, event];
    engine.setTempoTrack(newEvents);
    onTempoChange?.();
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
      <h3 style={{ color: '#fff', margin: '0 0 15px 0' }}>Tempo & Time Signature (Features 81-94)</h3>

      {/* Feature 81: Global Tempo */}
      <div>
        <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
          Global Tempo (Feature 81) - BPM
        </label>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="number"
            min="30"
            max="300"
            step="0.1"
            value={tempoInput}
            onChange={(e) => setTempoInput(e.target.value)}
            onBlur={(e) => {
              const val = parseFloat(e.target.value);
              if (!isNaN(val) && val >= 30 && val <= 300) {
                handleTempoChange(val);
              } else {
                setTempoInput(state.tempo.toString());
              }
            }}
            style={{
              flex: 1,
              padding: '10px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '1.2em',
              fontWeight: 'bold',
            }}
          />
          <div
            style={{
              padding: '10px 20px',
              backgroundColor: '#6366f1',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '1.2em',
              fontWeight: 'bold',
              minWidth: '80px',
              textAlign: 'center',
            }}
          >
            {state.tempo.toFixed(1)}
          </div>
        </div>
      </div>

      {/* Feature 88: Tempo Nudge */}
      <div>
        <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
          Tempo Nudge (Feature 88)
        </label>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="number"
            min="0.01"
            max="10"
            step="0.1"
            value={nudgeAmount}
            onChange={(e) => setNudgeAmount(parseFloat(e.target.value))}
            style={{
              width: '100px',
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          />
          <button
            onClick={() => handleNudgeTempo('down')}
            style={{
              padding: '8px 16px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            -{nudgeAmount} BPM
          </button>
          <button
            onClick={() => handleNudgeTempo('up')}
            style={{
              padding: '8px 16px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            +{nudgeAmount} BPM
          </button>
        </div>
      </div>

      {/* Feature 87: Tap Tempo */}
      <div>
        <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
          Tap Tempo (Feature 87)
        </label>
        <button
          onClick={handleTapTempo}
          style={{
            width: '100%',
            padding: '15px',
            backgroundColor: '#6366f1',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '1.1em',
            fontWeight: 'bold',
          }}
        >
          Tap ({tapTimes.length} taps)
        </button>
        {tapTimes.length > 0 && (
          <div style={{ color: '#888', fontSize: '0.85em', marginTop: '5px', textAlign: 'center' }}>
            Tap at least 2 times to set tempo
          </div>
        )}
      </div>

      {/* Feature 89: Time Signature */}
      <div>
        <label style={{ color: '#aaa', fontSize: '0.85em', display: 'block', marginBottom: '5px' }}>
          Time Signature (Feature 89)
        </label>
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <input
            type="number"
            min="1"
            max="32"
            value={timeSigNum}
            onChange={(e) => setTimeSigNum(parseInt(e.target.value))}
            onBlur={() => handleTimeSignatureChange(timeSigNum, timeSigDen)}
            style={{
              width: '80px',
              padding: '10px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '1.2em',
              fontWeight: 'bold',
              textAlign: 'center',
            }}
          />
          <div style={{ color: '#fff', fontSize: '1.5em', fontWeight: 'bold' }}>/</div>
          <select
            value={timeSigDen}
            onChange={(e) => {
              const den = parseInt(e.target.value);
              setTimeSigDen(den);
              handleTimeSignatureChange(timeSigNum, den);
            }}
            style={{
              width: '100px',
              padding: '10px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              fontSize: '1.2em',
              fontWeight: 'bold',
            }}
          >
            <option value="2">2</option>
            <option value="4">4</option>
            <option value="8">8</option>
            <option value="16">16</option>
          </select>
        </div>
      </div>

      {/* Tempo Track Controls */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <label style={{ color: '#aaa', fontSize: '0.85em' }}>Tempo Track (Feature 82)</label>
          <button
            onClick={createTempoEvent}
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
            Add Event
          </button>
        </div>

        {/* Feature 85: Tempo Ramp */}
        <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
          <button
            onClick={() => engine.setTempoRamp(!state.tempoRampEnabled)}
            style={{
              padding: '8px 16px',
              backgroundColor: state.tempoRampEnabled ? '#6366f1' : '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.9em',
            }}
          >
            Tempo Ramp (Feature 85)
          </button>

          {/* Feature 86: Master Tempo */}
          <button
            onClick={() => engine.setMasterTempo(!state.masterTempo)}
            style={{
              padding: '8px 16px',
              backgroundColor: state.masterTempo ? '#6366f1' : '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.9em',
            }}
          >
            Master Tempo (Feature 86)
          </button>
        </div>

        {/* Tempo Events List */}
        {state.tempoTrack.length > 0 && (
          <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
            {state.tempoTrack.map((event, idx) => (
              <div
                key={idx}
                style={{
                  padding: '8px',
                  backgroundColor: '#2a2a2a',
                  borderRadius: '4px',
                  marginBottom: '5px',
                  fontSize: '0.85em',
                  color: '#fff',
                }}
              >
                <div>
                  Time: {typeof event.time === 'number' ? event.time.toFixed(2) : 'N/A'}s | Tempo: {event.tempo} BPM
                  | Time Sig: {event.timeSignature[0]}/{event.timeSignature[1]}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Current Tempo at Position */}
      <div
        style={{
          padding: '15px',
          backgroundColor: '#0f0f0f',
          borderRadius: '4px',
          textAlign: 'center',
        }}
      >
        <div style={{ color: '#aaa', fontSize: '0.85em', marginBottom: '5px' }}>
          Tempo at Current Position
        </div>
        <div style={{ color: '#fff', fontSize: '1.5em', fontWeight: 'bold' }}>
          {engine.getTempoAtPosition(state.position).toFixed(1)} BPM
        </div>
      </div>
    </div>
  );
};
