import { useState, useEffect } from 'react';

interface VUMeterProps {
  level: number; // 0-1
  peak?: number;
  label?: string;
}

export const VUMeter: React.FC<VUMeterProps> = ({ level, peak, label }) => {
  const [peakHold, setPeakHold] = useState(0);

  useEffect(() => {
    if (peak !== undefined && peak > peakHold) {
      setPeakHold(peak);
      const timer = setTimeout(() => setPeakHold(0), 1000);
      return () => clearTimeout(timer);
    }
  }, [peak, peakHold]);

  const getColor = (value: number) => {
    if (value < 0.6) return '#4caf50'; // Green
    if (value < 0.8) return '#ffeb3b'; // Yellow
    return '#f44336'; // Red
  };

  const clipLevel = Math.min(1, Math.max(0, level));

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: '4px',
      width: '20px'
    }}>
      {label && (
        <span style={{ fontSize: '0.7em', color: '#888' }}>{label}</span>
      )}
      <div style={{
        width: '20px',
        height: '120px',
        backgroundColor: '#1a1a1a',
        border: '1px solid rgba(255, 255, 255, 0.2)',
        borderRadius: '2px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Level bar */}
        <div style={{
          position: 'absolute',
          bottom: 0,
          width: '100%',
          height: `${clipLevel * 100}%`,
          backgroundColor: getColor(clipLevel),
          transition: 'height 0.05s linear',
          boxShadow: `0 0 4px ${getColor(clipLevel)}`
        }} />

        {/* Peak hold indicator */}
        {peakHold > 0 && (
          <div style={{
            position: 'absolute',
            bottom: `${peakHold * 100}%`,
            width: '100%',
            height: '2px',
            backgroundColor: '#fff',
            boxShadow: '0 0 2px #fff'
          }} />
        )}

        {/* Clip indicator */}
        {clipLevel >= 1 && (
          <div style={{
            position: 'absolute',
            top: 0,
            width: '100%',
            height: '4px',
            backgroundColor: '#f44336',
            animation: 'blink 0.5s infinite'
          }} />
        )}
      </div>
    </div>
  );
};
