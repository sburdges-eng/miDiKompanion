import React, { useRef, useEffect, useState, useCallback } from 'react';

interface DoodleCanvasProps {
  width?: number;
  height?: number;
  color?: string;
  lineWidth?: number;
  enabled?: boolean;
  onDoodleComplete?: (paths: Array<Array<{ x: number; y: number }>>) => void;
}

export const DoodleCanvas: React.FC<DoodleCanvasProps> = ({
  width = 800,
  height = 400,
  color = '#6366f1',
  lineWidth = 3,
  enabled = true,
  onDoodleComplete,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentPath, setCurrentPath] = useState<Array<{ x: number; y: number }>>([]);
  const [paths, setPaths] = useState<Array<Array<{ x: number; y: number }>>>([]);
  const [time, setTime] = useState(0);

  // Redraw all paths
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid (blueprint style)
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.1)';
    ctx.lineWidth = 1;
    const gridSize = 20;

    for (let x = 0; x < canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }

    for (let y = 0; y < canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Draw all paths with hand-drawn wobble
    paths.forEach((path) => {
      if (path.length < 2) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);

      for (let i = 1; i < path.length; i++) {
        // Add slight wobble for hand-drawn effect
        const wobbleX = Math.sin(time + i * 0.1) * 0.5;
        const wobbleY = Math.cos(time + i * 0.1) * 0.5;
        ctx.lineTo(path[i].x + wobbleX, path[i].y + wobbleY);
      }

      ctx.stroke();
    });

    // Draw current path
    if (currentPath.length > 1) {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.globalAlpha = 0.8;

      ctx.beginPath();
      ctx.moveTo(currentPath[0].x, currentPath[0].y);

      for (let i = 1; i < currentPath.length; i++) {
        const wobbleX = Math.sin(time + i * 0.1) * 0.5;
        const wobbleY = Math.cos(time + i * 0.1) * 0.5;
        ctx.lineTo(currentPath[i].x + wobbleX, currentPath[i].y + wobbleY);
      }

      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }
  }, [paths, currentPath, color, lineWidth, time]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  // Animation loop for wobble effect
  useEffect(() => {
    const interval = setInterval(() => {
      setTime((t) => t + 0.1);
    }, 100);
    return () => clearInterval(interval);
  }, []);

  const getCanvasCoordinates = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!enabled) return;

    const coords = getCanvasCoordinates(e);
    if (!coords) return;

    setIsDrawing(true);
    setCurrentPath([coords]);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!enabled || !isDrawing) return;

    const coords = getCanvasCoordinates(e);
    if (!coords) return;

    setCurrentPath((prev) => [...prev, coords]);
  };

  const handleMouseUp = () => {
    if (!enabled || !isDrawing) return;

    if (currentPath.length > 0) {
      const newPaths = [...paths, currentPath];
      setPaths(newPaths);
      setCurrentPath([]);
      onDoodleComplete?.(newPaths);
    }

    setIsDrawing(false);
  };

  const clearCanvas = () => {
    setPaths([]);
    setCurrentPath([]);
  };

  return (
    <div style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: '#0a0a0a',
          borderRadius: '8px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          cursor: enabled ? 'crosshair' : 'default',
        }}
      />
      {enabled && (
        <div
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            display: 'flex',
            gap: '8px',
          }}
        >
          <button
            onClick={clearCanvas}
            style={{
              padding: '6px 12px',
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              color: '#fff',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.85em',
            }}
          >
            Clear
          </button>
        </div>
      )}
      {!enabled && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            color: '#666',
            fontSize: '0.9em',
            fontStyle: 'italic',
          }}
        >
          Click to enable doodling
        </div>
      )}
    </div>
  );
};
