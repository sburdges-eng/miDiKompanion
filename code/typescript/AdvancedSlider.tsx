import React, { useState, useRef, useEffect } from 'react';

interface AdvancedSliderProps {
  value: number;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  unit?: string;
  orientation?: 'vertical' | 'horizontal';
  onChange?: (value: number) => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
  color?: string;
  showValue?: boolean;
  logarithmic?: boolean;
  width?: number;
  height?: number;
}

export const AdvancedSlider: React.FC<AdvancedSliderProps> = ({
  value,
  min = 0,
  max = 1,
  step = 0.01,
  label,
  unit = '',
  orientation = 'vertical',
  onChange,
  onDragStart,
  onDragEnd,
  color = '#6366f1',
  showValue = true,
  logarithmic = false,
  width = 30,
  height = 200,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [localValue, setLocalValue] = useState(value);
  const sliderRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const normalizeValue = (val: number): number => {
    if (logarithmic) {
      const logMin = Math.log(min || 0.001);
      const logMax = Math.log(max);
      const logVal = Math.log(val);
      return (logVal - logMin) / (logMax - logMin);
    }
    return (val - min) / (max - min);
  };

  const denormalizeValue = (normalized: number): number => {
    if (logarithmic) {
      const logMin = Math.log(min || 0.001);
      const logMax = Math.log(max);
      const logVal = logMin + normalized * (logMax - logMin);
      return Math.exp(logVal);
    }
    return min + normalized * (max - min);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    onDragStart?.();
    handleMouseMove(e);
  };

  const handleMouseMove = (e: React.MouseEvent | MouseEvent) => {
    if (!isDragging && !(e.type === 'mousedown')) return;
    if (!sliderRef.current) return;

    const rect = sliderRef.current.getBoundingClientRect();
    let normalized: number;

    if (orientation === 'vertical') {
      const y = e.clientY - rect.top;
      normalized = 1 - Math.max(0, Math.min(1, y / rect.height));
    } else {
      const x = e.clientX - rect.left;
      normalized = Math.max(0, Math.min(1, x / rect.width));
    }

    const newValue = denormalizeValue(normalized);
    const steppedValue = Math.round(newValue / step) * step;
    const clampedValue = Math.max(min, Math.min(max, steppedValue));

    setLocalValue(clampedValue);
    onChange?.(clampedValue);
  };

  const handleMouseUp = () => {
    if (isDragging) {
      setIsDragging(false);
      onDragEnd?.();
    }
  };

  useEffect(() => {
    if (isDragging) {
      const handleGlobalMouseMove = (e: MouseEvent) => handleMouseMove(e);
      const handleGlobalMouseUp = () => handleMouseUp();

      window.addEventListener('mousemove', handleGlobalMouseMove);
      window.addEventListener('mouseup', handleGlobalMouseUp);

      return () => {
        window.removeEventListener('mousemove', handleGlobalMouseMove);
        window.removeEventListener('mouseup', handleGlobalMouseUp);
      };
    }
  }, [isDragging]);

  const normalizedValue = normalizeValue(localValue);
  const percentage = orientation === 'vertical' 
    ? normalizedValue * 100 
    : normalizedValue * 100;

  const formatValue = (val: number): string => {
    if (unit === 'dB') {
      return val <= 0 ? '-∞' : `${val.toFixed(1)}${unit}`;
    }
    if (unit === '%') {
      return `${Math.round(val * 100)}${unit}`;
    }
    if (unit === 'Hz' || unit === 'ms' || unit === 's') {
      return `${val.toFixed(1)}${unit}`;
    }
    return val.toFixed(2);
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: orientation === 'vertical' ? 'column' : 'row',
        alignItems: 'center',
        gap: '8px',
        width: orientation === 'vertical' ? width : 'auto',
        height: orientation === 'vertical' ? 'auto' : height,
      }}
    >
      {label && (
        <div
          style={{
            fontSize: '0.75em',
            color: '#fff',
            fontWeight: 'bold',
            textAlign: 'center',
            width: orientation === 'vertical' ? '100%' : '60px',
          }}
        >
          {label}
        </div>
      )}

      <div
        ref={sliderRef}
        style={{
          position: 'relative',
          width: orientation === 'vertical' ? width : height,
          height: orientation === 'vertical' ? height : width,
          backgroundColor: '#0f0f0f',
          border: '2px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '4px',
          cursor: 'pointer',
          boxShadow: isDragging ? `0 0 10px ${color}` : 'none',
          transition: 'box-shadow 0.2s',
        }}
        onMouseDown={handleMouseDown}
      >
        {/* Track fill */}
        {orientation === 'vertical' ? (
          <div
            style={{
              position: 'absolute',
              bottom: 0,
              width: '100%',
              height: `${percentage}%`,
              backgroundColor: color,
              opacity: 0.3,
              borderRadius: '2px 2px 0 0',
              transition: isDragging ? 'none' : 'height 0.1s',
            }}
          />
        ) : (
          <div
            style={{
              position: 'absolute',
              left: 0,
              width: `${percentage}%`,
              height: '100%',
              backgroundColor: color,
              opacity: 0.3,
              borderRadius: '2px 0 0 2px',
              transition: isDragging ? 'none' : 'width 0.1s',
            }}
          />
        )}

        {/* Scale marks */}
        {orientation === 'vertical' && (
          <div
            style={{
              position: 'absolute',
              right: '-25px',
              top: 0,
              bottom: 0,
              width: '20px',
              fontSize: '0.65em',
              color: '#888',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              padding: '4px 0',
            }}
          >
            <span>{formatValue(max)}</span>
            <span>{formatValue((max + min) / 2)}</span>
            <span>{formatValue(min)}</span>
          </div>
        )}

        {/* Handle */}
        {orientation === 'vertical' ? (
          <div
            style={{
              position: 'absolute',
              bottom: `${percentage}%`,
              left: '50%',
              transform: 'translate(-50%, 50%)',
              width: width + 8,
              height: '12px',
              backgroundColor: color,
              border: '2px solid rgba(255, 255, 255, 0.3)',
              borderRadius: '4px',
              cursor: 'grab',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.5)',
              transition: isDragging ? 'none' : 'bottom 0.1s',
            }}
          />
        ) : (
          <div
            style={{
              position: 'absolute',
              left: `${percentage}%`,
              top: '50%',
              transform: 'translate(-50%, -50%)',
              width: '12px',
              height: height + 8,
              backgroundColor: color,
              border: '2px solid rgba(255, 255, 255, 0.3)',
              borderRadius: '4px',
              cursor: 'grab',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.5)',
              transition: isDragging ? 'none' : 'left 0.1s',
            }}
          />
        )}

        {/* Glow effect when dragging */}
        {isDragging && (
          <div
            style={{
              position: 'absolute',
              inset: '-4px',
              borderRadius: '8px',
              boxShadow: `0 0 20px ${color}`,
              pointerEvents: 'none',
              animation: 'pulse 1s ease-in-out infinite',
            }}
          />
        )}
      </div>

      {showValue && (
        <div
          style={{
            fontSize: '0.7em',
            color: '#888',
            minWidth: '50px',
            textAlign: 'center',
            fontFamily: 'monospace',
          }}
        >
          {formatValue(localValue)}
        </div>
      )}
    </div>
  );
};
