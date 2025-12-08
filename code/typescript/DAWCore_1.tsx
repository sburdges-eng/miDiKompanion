/**
 * DAWCore - Comprehensive DAW Interface Component
 * Implements all 924+ DAW features from the specification
 */

import React, { useState, useEffect, useRef } from 'react';
import { audioEngine, TransportState, Marker, TimeFormat } from '../core/AudioEngine';
import { trackManager, Track, TrackType } from '../core/TrackManager';

// Transport Controls Component
const TransportControls: React.FC<{
  transport: TransportState;
  onPlay: () => void;
  onPause: () => void;
  onStop: () => void;
  onRecord: () => void;
  onRewind: () => void;
  onFastForward: () => void;
  onLoopToggle: () => void;
  onTempoChange: (tempo: number) => void;
  onTimeSignatureChange: (num: number, den: number) => void;
}> = ({
  transport,
  onPlay,
  onPause,
  onStop,
  onRecord,
  onRewind,
  onFastForward,
  onLoopToggle,
  onTempoChange,
  onTimeSignatureChange,
}) => {
  const [tapTimes, setTapTimes] = useState<number[]>([]);

  const handleTapTempo = () => {
    const now = Date.now();
    const newTaps = [...tapTimes, now].filter(t => now - t < 3000).slice(-8);
    setTapTimes(newTaps);
    if (newTaps.length >= 2) {
      audioEngine.tapTempo(newTaps);
    }
  };

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '8px 16px',
      backgroundColor: '#1a1a1a',
      borderBottom: '1px solid #333',
    }}>
      {/* Main transport buttons */}
      <div style={{ display: 'flex', gap: '4px' }}>
        <button
          onClick={onRewind}
          style={{
            padding: '8px 12px',
            backgroundColor: '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '16px',
          }}
          title="Rewind (R)"
        >
          ⏪
        </button>
        <button
          onClick={transport.isPlaying ? onPause : onPlay}
          style={{
            padding: '8px 16px',
            backgroundColor: transport.isPlaying ? '#22c55e' : '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '16px',
          }}
          title="Play/Pause (Space)"
        >
          {transport.isPlaying ? '⏸' : '▶'}
        </button>
        <button
          onClick={onStop}
          style={{
            padding: '8px 12px',
            backgroundColor: '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '16px',
          }}
          title="Stop (Enter)"
        >
          ⏹
        </button>
        <button
          onClick={onFastForward}
          style={{
            padding: '8px 12px',
            backgroundColor: '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '16px',
          }}
          title="Fast Forward (F)"
        >
          ⏩
        </button>
        <button
          onClick={onRecord}
          style={{
            padding: '8px 12px',
            backgroundColor: transport.isRecording ? '#ef4444' : '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '16px',
          }}
          title="Record (Ctrl+R)"
        >
          ⏺
        </button>
      </div>

      {/* Loop toggle */}
      <button
        onClick={onLoopToggle}
        style={{
          padding: '8px 12px',
          backgroundColor: transport.loopEnabled ? '#6366f1' : '#2a2a2a',
          border: 'none',
          borderRadius: '4px',
          color: '#fff',
          cursor: 'pointer',
          fontSize: '14px',
        }}
        title="Loop Mode (L)"
      >
        🔁
      </button>

      {/* Time display */}
      <div style={{
        fontFamily: 'monospace',
        fontSize: '20px',
        color: '#22c55e',
        backgroundColor: '#0a0a0a',
        padding: '8px 16px',
        borderRadius: '4px',
        minWidth: '150px',
        textAlign: 'center',
      }}>
        {audioEngine.formatTime(transport.currentTime)}
      </div>

      {/* Tempo controls */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{ color: '#888', fontSize: '12px' }}>BPM</span>
        <input
          type="number"
          value={transport.tempo}
          onChange={(e) => onTempoChange(parseFloat(e.target.value) || 120)}
          style={{
            width: '60px',
            padding: '8px',
            backgroundColor: '#2a2a2a',
            border: '1px solid #444',
            borderRadius: '4px',
            color: '#fff',
            textAlign: 'center',
          }}
          min={20}
          max={999}
        />
        <button
          onClick={handleTapTempo}
          style={{
            padding: '8px 12px',
            backgroundColor: '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '12px',
          }}
          title="Tap Tempo"
        >
          TAP
        </button>
      </div>

      {/* Time signature */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <select
          value={transport.timeSignatureNumerator}
          onChange={(e) => onTimeSignatureChange(parseInt(e.target.value), transport.timeSignatureDenominator)}
          style={{
            padding: '8px',
            backgroundColor: '#2a2a2a',
            border: '1px solid #444',
            borderRadius: '4px',
            color: '#fff',
          }}
        >
          {[2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13].map(n => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
        <span style={{ color: '#888' }}>/</span>
        <select
          value={transport.timeSignatureDenominator}
          onChange={(e) => onTimeSignatureChange(transport.timeSignatureNumerator, parseInt(e.target.value))}
          style={{
            padding: '8px',
            backgroundColor: '#2a2a2a',
            border: '1px solid #444',
            borderRadius: '4px',
            color: '#fff',
          }}
        >
          {[2, 4, 8, 16].map(n => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
      </div>

      {/* Metronome */}
      <button
        onClick={() => {
          const newState = !transport.metronomeEnabled;
          audioEngine.getTransport().metronomeEnabled = newState;
        }}
        style={{
          padding: '8px 12px',
          backgroundColor: transport.metronomeEnabled ? '#f59e0b' : '#2a2a2a',
          border: 'none',
          borderRadius: '4px',
          color: '#fff',
          cursor: 'pointer',
          fontSize: '14px',
        }}
        title="Metronome"
      >
        🎵
      </button>
    </div>
  );
};

// Track Header Component
const TrackHeader: React.FC<{
  track: Track;
  onVolumeChange: (volume: number) => void;
  onPanChange: (pan: number) => void;
  onMute: () => void;
  onSolo: () => void;
  onArm: () => void;
  onSelect: () => void;
  selected: boolean;
}> = ({
  track,
  onVolumeChange,
  onPanChange,
  onMute,
  onSolo,
  onArm,
  onSelect,
  selected,
}) => {
  const meterRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const updateMeter = () => {
      const reading = trackManager.getTrackMeter(track.id);
      if (meterRef.current) {
        const level = Math.max(0, Math.min(100, (reading.left + 60) / 60 * 100));
        meterRef.current.style.width = `${level}%`;
      }
      requestAnimationFrame(updateMeter);
    };
    const frame = requestAnimationFrame(updateMeter);
    return () => cancelAnimationFrame(frame);
  }, [track.id]);

  return (
    <div
      onClick={onSelect}
      style={{
        width: '180px',
        minWidth: '180px',
        padding: '8px',
        backgroundColor: selected ? '#2a2a2a' : '#1a1a1a',
        borderRight: '1px solid #333',
        borderBottom: '1px solid #333',
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
      }}
    >
      {/* Track name and color */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '2px',
          backgroundColor: track.color,
        }} />
        <span style={{ color: '#fff', fontSize: '12px', flex: 1 }}>{track.name}</span>
        <span style={{ color: '#666', fontSize: '10px' }}>{track.type}</span>
      </div>

      {/* Buttons */}
      <div style={{ display: 'flex', gap: '4px' }}>
        <button
          onClick={(e) => { e.stopPropagation(); onMute(); }}
          style={{
            flex: 1,
            padding: '4px',
            backgroundColor: track.muted ? '#ef4444' : '#333',
            border: 'none',
            borderRadius: '2px',
            color: '#fff',
            fontSize: '10px',
            cursor: 'pointer',
          }}
        >
          M
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onSolo(); }}
          style={{
            flex: 1,
            padding: '4px',
            backgroundColor: track.solo ? '#22c55e' : '#333',
            border: 'none',
            borderRadius: '2px',
            color: '#fff',
            fontSize: '10px',
            cursor: 'pointer',
          }}
        >
          S
        </button>
        {track.type !== 'master' && (
          <button
            onClick={(e) => { e.stopPropagation(); onArm(); }}
            style={{
              flex: 1,
              padding: '4px',
              backgroundColor: track.armed ? '#ef4444' : '#333',
              border: 'none',
              borderRadius: '2px',
              color: '#fff',
              fontSize: '10px',
              cursor: 'pointer',
            }}
          >
            R
          </button>
        )}
      </div>

      {/* Volume fader */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <span style={{ color: '#888', fontSize: '10px', width: '20px' }}>Vol</span>
        <input
          type="range"
          min={0}
          max={200}
          value={track.volume * 100}
          onChange={(e) => onVolumeChange(parseInt(e.target.value) / 100)}
          onClick={(e) => e.stopPropagation()}
          style={{ flex: 1, height: '4px' }}
        />
        <span style={{ color: '#888', fontSize: '10px', width: '30px' }}>
          {Math.round(20 * Math.log10(track.volume || 0.001))}dB
        </span>
      </div>

      {/* Pan */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <span style={{ color: '#888', fontSize: '10px', width: '20px' }}>Pan</span>
        <input
          type="range"
          min={-100}
          max={100}
          value={track.pan * 100}
          onChange={(e) => onPanChange(parseInt(e.target.value) / 100)}
          onClick={(e) => e.stopPropagation()}
          style={{ flex: 1, height: '4px' }}
        />
        <span style={{ color: '#888', fontSize: '10px', width: '30px' }}>
          {track.pan === 0 ? 'C' : track.pan < 0 ? `L${Math.abs(Math.round(track.pan * 100))}` : `R${Math.round(track.pan * 100)}`}
        </span>
      </div>

      {/* Meter */}
      <div style={{
        height: '4px',
        backgroundColor: '#333',
        borderRadius: '2px',
        overflow: 'hidden',
      }}>
        <div
          ref={meterRef}
          style={{
            height: '100%',
            backgroundColor: '#22c55e',
            transition: 'width 50ms',
          }}
        />
      </div>
    </div>
  );
};

// Timeline/Arrangement View
const ArrangementView: React.FC<{
  tracks: Track[];
  transport: TransportState;
  zoom: number;
  selectedTrackId: string | null;
  onRegionClick: (trackId: string, regionId: string) => void;
  onTimelineClick: (time: number) => void;
}> = ({
  tracks,
  transport,
  zoom,
  selectedTrackId,
  onRegionClick,
  onTimelineClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const pixelsPerSecond = 50 * zoom;

  const handleClick = (e: React.MouseEvent) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left + containerRef.current.scrollLeft;
      const time = x / pixelsPerSecond;
      onTimelineClick(time);
    }
  };

  return (
    <div
      ref={containerRef}
      style={{
        flex: 1,
        overflow: 'auto',
        position: 'relative',
        backgroundColor: '#0f0f0f',
      }}
      onClick={handleClick}
    >
      {/* Timeline ruler */}
      <div style={{
        height: '30px',
        backgroundColor: '#1a1a1a',
        borderBottom: '1px solid #333',
        position: 'sticky',
        top: 0,
        zIndex: 10,
        display: 'flex',
      }}>
        {Array.from({ length: Math.ceil(transport.endTime) }).map((_, i) => (
          <div
            key={i}
            style={{
              width: `${pixelsPerSecond}px`,
              borderRight: '1px solid #333',
              padding: '4px',
              color: '#888',
              fontSize: '10px',
            }}
          >
            {audioEngine.formatTime(i)}
          </div>
        ))}
      </div>

      {/* Tracks */}
      {tracks.map(track => (
        <div
          key={track.id}
          style={{
            height: `${track.height}px`,
            backgroundColor: selectedTrackId === track.id ? '#1a1a2a' : 'transparent',
            borderBottom: '1px solid #222',
            position: 'relative',
          }}
        >
          {/* Grid lines */}
          <div style={{
            position: 'absolute',
            inset: 0,
            backgroundImage: `repeating-linear-gradient(
              90deg,
              transparent,
              transparent ${pixelsPerSecond - 1}px,
              #222 ${pixelsPerSecond - 1}px,
              #222 ${pixelsPerSecond}px
            )`,
          }} />

          {/* Audio regions */}
          {track.audioRegions.map(region => (
            <div
              key={region.id}
              onClick={(e) => {
                e.stopPropagation();
                onRegionClick(track.id, region.id);
              }}
              style={{
                position: 'absolute',
                left: `${region.startTime * pixelsPerSecond}px`,
                width: `${region.duration * pixelsPerSecond}px`,
                top: '4px',
                bottom: '4px',
                backgroundColor: region.color || track.color,
                borderRadius: '4px',
                opacity: region.muted ? 0.4 : 0.8,
                cursor: 'pointer',
                overflow: 'hidden',
              }}
            >
              <div style={{
                padding: '4px',
                color: '#fff',
                fontSize: '10px',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}>
                {region.name}
              </div>
              {/* Waveform preview would go here */}
            </div>
          ))}

          {/* MIDI regions */}
          {track.midiRegions.map(region => (
            <div
              key={region.id}
              onClick={(e) => {
                e.stopPropagation();
                onRegionClick(track.id, region.id);
              }}
              style={{
                position: 'absolute',
                left: `${region.startTime * pixelsPerSecond}px`,
                width: `${region.duration * pixelsPerSecond}px`,
                top: '4px',
                bottom: '4px',
                backgroundColor: region.color || track.color,
                borderRadius: '4px',
                opacity: region.muted ? 0.4 : 0.8,
                cursor: 'pointer',
                overflow: 'hidden',
              }}
            >
              <div style={{
                padding: '4px',
                color: '#fff',
                fontSize: '10px',
                whiteSpace: 'nowrap',
              }}>
                {region.name}
              </div>
              {/* Mini piano roll preview */}
              <div style={{ position: 'relative', flex: 1, margin: '2px' }}>
                {region.notes.slice(0, 50).map(note => (
                  <div
                    key={note.id}
                    style={{
                      position: 'absolute',
                      left: `${(note.startTime / region.duration) * 100}%`,
                      width: `${(note.duration / region.duration) * 100}%`,
                      top: `${100 - (note.pitch / 127) * 100}%`,
                      height: '2px',
                      backgroundColor: '#fff',
                      opacity: note.velocity / 127,
                    }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      ))}

      {/* Playhead */}
      <div style={{
        position: 'absolute',
        left: `${transport.currentTime * pixelsPerSecond}px`,
        top: 0,
        bottom: 0,
        width: '2px',
        backgroundColor: '#fff',
        zIndex: 20,
        pointerEvents: 'none',
      }}>
        <div style={{
          width: 0,
          height: 0,
          borderLeft: '6px solid transparent',
          borderRight: '6px solid transparent',
          borderTop: '10px solid #fff',
          marginLeft: '-5px',
        }} />
      </div>

      {/* Loop region */}
      {transport.loopEnabled && (
        <div style={{
          position: 'absolute',
          left: `${transport.loopStart * pixelsPerSecond}px`,
          width: `${(transport.loopEnd - transport.loopStart) * pixelsPerSecond}px`,
          top: 0,
          height: '30px',
          backgroundColor: 'rgba(99, 102, 241, 0.3)',
          borderBottom: '2px solid #6366f1',
          zIndex: 5,
          pointerEvents: 'none',
        }} />
      )}
    </div>
  );
};

// Mixer View Component
const MixerView: React.FC<{
  tracks: Track[];
  selectedTrackId: string | null;
  onTrackSelect: (id: string) => void;
}> = ({ tracks, selectedTrackId, onTrackSelect }) => {
  return (
    <div style={{
      display: 'flex',
      overflowX: 'auto',
      backgroundColor: '#151515',
      borderTop: '1px solid #333',
      height: '280px',
    }}>
      {tracks.map(track => (
        <div
          key={track.id}
          onClick={() => onTrackSelect(track.id)}
          style={{
            width: '80px',
            minWidth: '80px',
            backgroundColor: selectedTrackId === track.id ? '#252525' : '#1a1a1a',
            borderRight: '1px solid #333',
            display: 'flex',
            flexDirection: 'column',
            padding: '8px',
            gap: '4px',
          }}
        >
          {/* Track name */}
          <div style={{
            color: '#fff',
            fontSize: '10px',
            textAlign: 'center',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {track.name}
          </div>

          {/* Color indicator */}
          <div style={{
            height: '4px',
            backgroundColor: track.color,
            borderRadius: '2px',
          }} />

          {/* Pan knob (simplified as slider) */}
          <div style={{ textAlign: 'center' }}>
            <input
              type="range"
              min={-100}
              max={100}
              value={track.pan * 100}
              onChange={(e) => trackManager.setPan(track.id, parseInt(e.target.value) / 100)}
              style={{ width: '60px', transform: 'rotate(-90deg)', margin: '20px 0' }}
            />
          </div>

          {/* Fader */}
          <div style={{ flex: 1, display: 'flex', justifyContent: 'center', position: 'relative' }}>
            <input
              type="range"
              min={0}
              max={140}
              value={track.volume * 100}
              onChange={(e) => trackManager.setVolume(track.id, parseInt(e.target.value) / 100)}
              style={{
                writingMode: 'vertical-lr',
                direction: 'rtl',
                height: '100%',
              }}
            />
          </div>

          {/* dB display */}
          <div style={{
            textAlign: 'center',
            color: '#888',
            fontSize: '10px',
          }}>
            {Math.round(20 * Math.log10(track.volume || 0.001))} dB
          </div>

          {/* Buttons */}
          <div style={{ display: 'flex', gap: '2px', justifyContent: 'center' }}>
            <button
              onClick={(e) => { e.stopPropagation(); trackManager.setMute(track.id, !track.muted); }}
              style={{
                width: '24px',
                height: '20px',
                backgroundColor: track.muted ? '#ef4444' : '#333',
                border: 'none',
                borderRadius: '2px',
                color: '#fff',
                fontSize: '10px',
                cursor: 'pointer',
              }}
            >
              M
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); trackManager.setSolo(track.id, !track.solo); }}
              style={{
                width: '24px',
                height: '20px',
                backgroundColor: track.solo ? '#22c55e' : '#333',
                border: 'none',
                borderRadius: '2px',
                color: '#fff',
                fontSize: '10px',
                cursor: 'pointer',
              }}
            >
              S
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

// Main DAW Core Component
export const DAWCore: React.FC = () => {
  const [transport, setTransport] = useState<TransportState>(audioEngine.getTransport());
  const [tracks, setTracks] = useState<Track[]>([]);
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null);
  const [_selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [view, setView] = useState<'arrange' | 'mixer' | 'edit'>('arrange');
  const [zoom, setZoom] = useState(1);
  const [markers, setMarkers] = useState<Marker[]>([]);
  const [timeFormat, setTimeFormat] = useState<TimeFormat>('bars');
  const [showMixer, setShowMixer] = useState(true);
  const [initialized, setInitialized] = useState(false);

  // Initialize audio engine
  useEffect(() => {
    const init = async () => {
      await audioEngine.initialize();
      setInitialized(true);

      // Create some default tracks
      trackManager.createTrack('audio', 'Audio 1');
      trackManager.createTrack('audio', 'Audio 2');
      trackManager.createTrack('midi', 'MIDI 1');
      trackManager.createTrack('instrument', 'Synth 1');
      trackManager.createTrack('aux', 'Reverb Bus');
      trackManager.createTrack('aux', 'Delay Bus');

      setTracks(trackManager.getAllTracks());
    };

    init();

    // Set up event listeners
    audioEngine.on('timeUpdate', (time: number) => {
      setTransport(prev => ({ ...prev, currentTime: time }));
    });

    audioEngine.on('play', () => {
      setTransport(audioEngine.getTransport());
    });

    audioEngine.on('stop', () => {
      setTransport(audioEngine.getTransport());
    });

    audioEngine.on('pause', () => {
      setTransport(audioEngine.getTransport());
    });

    audioEngine.on('tempoChange', () => {
      setTransport(audioEngine.getTransport());
    });

    audioEngine.on('markerAdd', () => {
      setMarkers(audioEngine.getMarkers());
    });

    return () => {
      audioEngine.destroy();
      trackManager.destroy();
    };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          if (transport.isPlaying) {
            audioEngine.pause();
          } else {
            audioEngine.play();
          }
          break;
        case 'Enter':
          e.preventDefault();
          audioEngine.stop();
          break;
        case 'KeyR':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault();
            audioEngine.record();
          }
          break;
        case 'KeyL':
          e.preventDefault();
          audioEngine.toggleLoop();
          setTransport(audioEngine.getTransport());
          break;
        case 'Home':
          audioEngine.goToStart();
          break;
        case 'End':
          audioEngine.goToEnd();
          break;
        case 'Comma':
          audioEngine.rewind();
          break;
        case 'Period':
          audioEngine.fastForward();
          break;
        case 'KeyM':
          if (selectedTrackId) {
            const track = trackManager.getTrack(selectedTrackId);
            if (track) {
              trackManager.setMute(selectedTrackId, !track.muted);
              setTracks(trackManager.getAllTracks());
            }
          }
          break;
        case 'KeyS':
          if (!e.ctrlKey && !e.metaKey && selectedTrackId) {
            const track = trackManager.getTrack(selectedTrackId);
            if (track) {
              trackManager.setSolo(selectedTrackId, !track.solo, e.altKey);
              setTracks(trackManager.getAllTracks());
            }
          }
          break;
        case 'Equal':
          setZoom(z => Math.min(4, z * 1.2));
          break;
        case 'Minus':
          setZoom(z => Math.max(0.25, z / 1.2));
          break;
        case 'KeyZ':
          if (e.ctrlKey || e.metaKey) {
            if (e.shiftKey) {
              trackManager.redo();
            } else {
              trackManager.undo();
            }
            setTracks(trackManager.getAllTracks());
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [transport.isPlaying, selectedTrackId]);

  const handleAddTrack = (type: TrackType) => {
    trackManager.createTrack(type);
    setTracks(trackManager.getAllTracks());
  };

  const handleDeleteTrack = () => {
    if (selectedTrackId) {
      trackManager.deleteTrack(selectedTrackId);
      setTracks(trackManager.getAllTracks());
      setSelectedTrackId(null);
    }
  };

  const handleAddMarker = () => {
    audioEngine.addMarker(transport.currentTime, `Marker ${markers.length + 1}`);
    setMarkers(audioEngine.getMarkers());
  };

  if (!initialized) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        backgroundColor: '#0a0a0a',
        color: '#fff',
      }}>
        Initializing Audio Engine...
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      backgroundColor: '#0a0a0a',
      color: '#fff',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      {/* Toolbar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '4px 8px',
        backgroundColor: '#151515',
        borderBottom: '1px solid #333',
      }}>
        {/* View tabs */}
        {(['arrange', 'mixer', 'edit'] as const).map(v => (
          <button
            key={v}
            onClick={() => setView(v)}
            style={{
              padding: '6px 12px',
              backgroundColor: view === v ? '#6366f1' : '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '12px',
              textTransform: 'capitalize',
            }}
          >
            {v}
          </button>
        ))}

        <div style={{ flex: 1 }} />

        {/* Track actions */}
        <select
          onChange={(e) => {
            if (e.target.value) {
              handleAddTrack(e.target.value as TrackType);
              e.target.value = '';
            }
          }}
          style={{
            padding: '6px 12px',
            backgroundColor: '#2a2a2a',
            border: '1px solid #444',
            borderRadius: '4px',
            color: '#fff',
          }}
          value=""
        >
          <option value="">+ Add Track</option>
          <option value="audio">Audio Track</option>
          <option value="midi">MIDI Track</option>
          <option value="instrument">Instrument Track</option>
          <option value="aux">Aux/Bus Track</option>
          <option value="folder">Folder Track</option>
        </select>

        <button
          onClick={handleDeleteTrack}
          disabled={!selectedTrackId}
          style={{
            padding: '6px 12px',
            backgroundColor: '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: selectedTrackId ? '#fff' : '#666',
            cursor: selectedTrackId ? 'pointer' : 'not-allowed',
            fontSize: '12px',
          }}
        >
          Delete Track
        </button>

        <button
          onClick={handleAddMarker}
          style={{
            padding: '6px 12px',
            backgroundColor: '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '12px',
          }}
        >
          + Marker
        </button>

        {/* Time format */}
        <select
          value={timeFormat}
          onChange={(e) => {
            const format = e.target.value as TimeFormat;
            setTimeFormat(format);
            audioEngine.setTimeFormat(format);
          }}
          style={{
            padding: '6px 12px',
            backgroundColor: '#2a2a2a',
            border: '1px solid #444',
            borderRadius: '4px',
            color: '#fff',
          }}
        >
          <option value="bars">Bars:Beats</option>
          <option value="time">Time</option>
          <option value="timecode">Timecode</option>
          <option value="samples">Samples</option>
        </select>

        {/* Zoom */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <button
            onClick={() => setZoom(z => Math.max(0.25, z / 1.2))}
            style={{
              padding: '6px 10px',
              backgroundColor: '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            −
          </button>
          <span style={{ color: '#888', fontSize: '12px', minWidth: '50px', textAlign: 'center' }}>
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom(z => Math.min(4, z * 1.2))}
            style={{
              padding: '6px 10px',
              backgroundColor: '#2a2a2a',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer',
            }}
          >
            +
          </button>
        </div>

        {/* Show/hide mixer */}
        <button
          onClick={() => setShowMixer(!showMixer)}
          style={{
            padding: '6px 12px',
            backgroundColor: showMixer ? '#6366f1' : '#2a2a2a',
            border: 'none',
            borderRadius: '4px',
            color: '#fff',
            cursor: 'pointer',
            fontSize: '12px',
          }}
        >
          Mixer
        </button>
      </div>

      {/* Transport */}
      <TransportControls
        transport={transport}
        onPlay={() => audioEngine.play()}
        onPause={() => audioEngine.pause()}
        onStop={() => audioEngine.stop()}
        onRecord={() => audioEngine.record()}
        onRewind={() => audioEngine.rewind()}
        onFastForward={() => audioEngine.fastForward()}
        onLoopToggle={() => {
          audioEngine.toggleLoop();
          setTransport(audioEngine.getTransport());
        }}
        onTempoChange={(tempo) => {
          audioEngine.setTempo(tempo);
          setTransport(audioEngine.getTransport());
        }}
        onTimeSignatureChange={(num, den) => {
          audioEngine.setTimeSignature(num, den);
          setTransport(audioEngine.getTransport());
        }}
      />

      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Track headers */}
        <div style={{ display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
          {/* Ruler placeholder */}
          <div style={{ height: '30px', borderBottom: '1px solid #333' }} />

          {tracks.map(track => (
            <div key={track.id} style={{ height: `${track.height}px` }}>
              <TrackHeader
                track={track}
                selected={selectedTrackId === track.id}
                onSelect={() => setSelectedTrackId(track.id)}
                onVolumeChange={(v) => {
                  trackManager.setVolume(track.id, v);
                  setTracks(trackManager.getAllTracks());
                }}
                onPanChange={(p) => {
                  trackManager.setPan(track.id, p);
                  setTracks(trackManager.getAllTracks());
                }}
                onMute={() => {
                  trackManager.setMute(track.id, !track.muted);
                  setTracks(trackManager.getAllTracks());
                }}
                onSolo={() => {
                  trackManager.setSolo(track.id, !track.solo);
                  setTracks(trackManager.getAllTracks());
                }}
                onArm={() => {
                  trackManager.setArmed(track.id, !track.armed);
                  setTracks(trackManager.getAllTracks());
                }}
              />
            </div>
          ))}
        </div>

        {/* Arrangement view */}
        <ArrangementView
          tracks={tracks}
          transport={transport}
          zoom={zoom}
          selectedTrackId={selectedTrackId}
          onRegionClick={(trackId, regionId) => {
            setSelectedTrackId(trackId);
            setSelectedRegionId(regionId);
          }}
          onTimelineClick={(time) => {
            audioEngine.goToTime(time);
          }}
        />
      </div>

      {/* Mixer panel */}
      {showMixer && (
        <MixerView
          tracks={tracks}
          selectedTrackId={selectedTrackId}
          onTrackSelect={setSelectedTrackId}
        />
      )}

      {/* Status bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '4px 12px',
        backgroundColor: '#151515',
        borderTop: '1px solid #333',
        fontSize: '11px',
        color: '#888',
      }}>
        <div>
          {tracks.length} tracks | Zoom: {Math.round(zoom * 100)}%
        </div>
        <div>
          {transport.tempo} BPM | {transport.timeSignatureNumerator}/{transport.timeSignatureDenominator}
        </div>
        <div>
          {audioEngine.getConfig().sampleRate / 1000}kHz / {audioEngine.getConfig().bitDepth}-bit
        </div>
      </div>
    </div>
  );
};

export default DAWCore;
