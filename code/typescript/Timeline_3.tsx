import { useState, useRef } from 'react';

interface MIDIRegion {
  id: string;
  start: number; // in bars
  length: number; // in bars
  track: number;
  name: string;
  color?: string;
}

interface TimelineTrack {
  id: string;
  name: string;
  regions: MIDIRegion[];
}

interface TimelineProps {
  tracks?: TimelineTrack[];
  tempo?: number;
  timeSignature?: [number, number];
  onRegionClick?: (region: MIDIRegion) => void;
}

export const Timeline: React.FC<TimelineProps> = ({
  tracks = [],
  tempo: _tempo = 120,
  timeSignature: _timeSignature = [4, 4],
  onRegionClick,
}) => {
  const [zoom, setZoom] = useState(1);
  const [scrollPosition, setScrollPosition] = useState(0);
  const [playheadPosition, _setPlayheadPosition] = useState(0);
  const timelineRef = useRef<HTMLDivElement>(null);
  const [_isDragging, _setIsDragging] = useState(false);

  // const barsPerView = 16; // Reserved for future use
  const pixelsPerBar = 80 * zoom;
  const totalBars = Math.max(32, ...tracks.flatMap(t => t.regions.map(r => r.start + r.length)));

  const handleZoom = (delta: number) => {
    setZoom(prev => Math.max(0.5, Math.min(4, prev + delta)));
  };

  const handleScroll = (e: React.WheelEvent) => {
    if (e.shiftKey) {
      e.preventDefault();
      handleZoom(e.deltaY > 0 ? -0.1 : 0.1);
    } else {
      setScrollPosition(prev => {
        const maxScroll = Math.max(0, totalBars * pixelsPerBar - (timelineRef.current?.clientWidth || 0));
        return Math.max(0, Math.min(maxScroll, prev + e.deltaY));
      });
    }
  };

  const getBarPosition = (bar: number) => {
    return bar * pixelsPerBar;
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      backgroundColor: '#1a1a1a',
      color: '#fff',
      overflow: 'hidden'
    }}>
      {/* Time ruler */}
      <div style={{
        height: '30px',
        backgroundColor: '#0f0f0f',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        display: 'flex',
        alignItems: 'center',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div
          style={{
            position: 'absolute',
            left: `${-scrollPosition}px`,
            display: 'flex',
            height: '100%'
          }}
        >
          {Array.from({ length: totalBars + 1 }).map((_, i) => (
            <div
              key={i}
              style={{
                width: `${pixelsPerBar}px`,
                borderLeft: '1px solid rgba(255, 255, 255, 0.2)',
                paddingLeft: '4px',
                fontSize: '0.75em',
                color: '#888',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              {i % 4 === 0 && <span>{i}</span>}
            </div>
          ))}
        </div>
      </div>

      {/* Timeline tracks */}
      <div
        ref={timelineRef}
        onWheel={handleScroll}
        style={{
          flex: 1,
          overflow: 'auto',
          position: 'relative',
          backgroundColor: '#1a1a1a'
        }}
      >
        {/* Grid background */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: `${-scrollPosition}px`,
            width: `${totalBars * pixelsPerBar}px`,
            height: '100%',
            backgroundImage: `
              repeating-linear-gradient(
                to right,
                transparent,
                transparent ${pixelsPerBar - 1}px,
                rgba(255, 255, 255, 0.05) ${pixelsPerBar - 1}px,
                rgba(255, 255, 255, 0.05) ${pixelsPerBar}px
              )
            `,
            pointerEvents: 'none'
          }}
        />

        {/* Playhead */}
        <div
          style={{
            position: 'absolute',
            left: `${getBarPosition(playheadPosition) - scrollPosition}px`,
            top: 0,
            bottom: 0,
            width: '2px',
            backgroundColor: '#4caf50',
            zIndex: 10,
            pointerEvents: 'none',
            boxShadow: '0 0 4px #4caf50'
          }}
        >
          <div style={{
            position: 'absolute',
            top: 0,
            left: '-6px',
            width: '14px',
            height: '14px',
            backgroundColor: '#4caf50',
            borderRadius: '50%',
            border: '2px solid #1a1a1a'
          }} />
        </div>

        {/* Tracks */}
        <div style={{ position: 'relative', zIndex: 1 }}>
          {tracks.length === 0 ? (
            <div style={{
              padding: '40px',
              textAlign: 'center',
              color: '#666',
              fontSize: '0.9em'
            }}>
              No tracks yet. Generate music to see it here.
            </div>
          ) : (
            tracks.map((track, _trackIdx) => (
              <div
                key={track.id}
                style={{
                  height: '80px',
                  borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                  display: 'flex',
                  position: 'relative'
                }}
              >
                {/* Track name */}
                <div style={{
                  width: '150px',
                  padding: '10px',
                  backgroundColor: '#0f0f0f',
                  borderRight: '1px solid rgba(255, 255, 255, 0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: '0.85em',
                  flexShrink: 0
                }}>
                  {track.name}
                </div>

                {/* Track content */}
                <div style={{
                  flex: 1,
                  position: 'relative',
                  overflow: 'hidden'
                }}>
                  {track.regions.map((region) => (
                    <div
                      key={region.id}
                      onClick={() => onRegionClick?.(region)}
                      style={{
                        position: 'absolute',
                        left: `${getBarPosition(region.start) - scrollPosition}px`,
                        top: '10px',
                        width: `${getBarPosition(region.length)}px`,
                        height: '60px',
                        backgroundColor: region.color || '#6366f1',
                        borderRadius: '4px',
                        border: '1px solid rgba(255, 255, 255, 0.3)',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        padding: '0 8px',
                        fontSize: '0.8em',
                        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
                        transition: 'transform 0.1s',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = 'scale(1.02)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'scale(1)';
                      }}
                    >
                      {region.name}
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Zoom controls */}
      <div style={{
        padding: '8px',
        backgroundColor: '#0f0f0f',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '0.85em'
      }}>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            onClick={() => handleZoom(-0.1)}
            style={{
              padding: '4px 8px',
              backgroundColor: '#333',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer'
            }}
          >
            −
          </button>
          <span style={{ minWidth: '60px', textAlign: 'center' }}>
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => handleZoom(0.1)}
            style={{
              padding: '4px 8px',
              backgroundColor: '#333',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
              cursor: 'pointer'
            }}
          >
            +
          </button>
        </div>
        <div style={{ color: '#888' }}>
          {totalBars} bars
        </div>
      </div>
    </div>
  );
};
