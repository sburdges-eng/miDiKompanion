/**
 * WaveformDisplay - Real-time waveform visualization synced to audio
 * Features: Scrolling waveform, playhead, zoom, and audio-reactive animations
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

interface WaveformDisplayProps {
  audioData?: Float32Array | number[];
  isPlaying?: boolean;
  currentTime?: number;
  duration?: number;
  zoom?: number;
  color?: string;
  backgroundColor?: string;
  onSeek?: (time: number) => void;
  height?: number;
  showGrid?: boolean;
  showPlayhead?: boolean;
}

export const WaveformDisplay: React.FC<WaveformDisplayProps> = ({
  audioData,
  isPlaying = false,
  currentTime = 0,
  duration = 60,
  zoom = 1,
  color = '#6366f1',
  backgroundColor = '#0f0f0f',
  onSeek,
  height = 120,
  showGrid = true,
  showPlayhead = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | undefined>(undefined);
  const [dimensions, setDimensions] = useState({ width: 800, height });
  const [hoverTime, setHoverTime] = useState<number | null>(null);

  // Generate demo waveform data if none provided
  const generateDemoWaveform = useCallback((length: number): number[] => {
    const data: number[] = [];
    for (let i = 0; i < length; i++) {
      const t = i / length;
      // Create realistic-looking waveform with multiple frequencies
      const wave =
        Math.sin(t * Math.PI * 20) * 0.3 +
        Math.sin(t * Math.PI * 47) * 0.2 +
        Math.sin(t * Math.PI * 123) * 0.15 +
        (Math.random() - 0.5) * 0.2;
      // Add envelope variation
      const envelope = Math.sin(t * Math.PI * 4) * 0.3 + 0.5;
      data.push(wave * envelope);
    }
    return data;
  }, []);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: height,
        });
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [height]);

  // Draw waveform
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = dimensions;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear canvas
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    if (showGrid) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
      ctx.lineWidth = 1;

      // Horizontal center line
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();

      // Vertical grid lines (every second based on zoom)
      const pixelsPerSecond = (width * zoom) / duration;
      const gridInterval = pixelsPerSecond > 50 ? 1 : pixelsPerSecond > 20 ? 5 : 10;

      for (let t = 0; t < duration; t += gridInterval) {
        const x = (t / duration) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        // Time labels
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.font = '10px monospace';
        const minutes = Math.floor(t / 60);
        const seconds = Math.floor(t % 60);
        ctx.fillText(`${minutes}:${seconds.toString().padStart(2, '0')}`, x + 2, height - 4);
      }
    }

    // Get waveform data
    const waveData = audioData || generateDemoWaveform(width * 2);
    const dataLength = waveData.length;
    const samplesPerPixel = dataLength / width;

    // Draw waveform with gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, color);
    gradient.addColorStop(0.5, color + '80');
    gradient.addColorStop(1, color);

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);

    // Draw upper half
    for (let x = 0; x < width; x++) {
      const startSample = Math.floor(x * samplesPerPixel);
      const endSample = Math.floor((x + 1) * samplesPerPixel);

      let max = 0;
      for (let i = startSample; i < endSample && i < dataLength; i++) {
        const value = typeof waveData[i] === 'number' ? waveData[i] : 0;
        max = Math.max(max, Math.abs(value));
      }

      const y = (height / 2) - (max * height / 2 * 0.9);
      ctx.lineTo(x, y);
    }

    // Draw lower half (mirror)
    for (let x = width - 1; x >= 0; x--) {
      const startSample = Math.floor(x * samplesPerPixel);
      const endSample = Math.floor((x + 1) * samplesPerPixel);

      let max = 0;
      for (let i = startSample; i < endSample && i < dataLength; i++) {
        const value = typeof waveData[i] === 'number' ? waveData[i] : 0;
        max = Math.max(max, Math.abs(value));
      }

      const y = (height / 2) + (max * height / 2 * 0.9);
      ctx.lineTo(x, y);
    }

    ctx.closePath();
    ctx.fill();

    // Draw played portion with different color
    if (currentTime > 0) {
      const playedWidth = (currentTime / duration) * width;

      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, playedWidth, height);
      ctx.clip();

      const playedGradient = ctx.createLinearGradient(0, 0, 0, height);
      playedGradient.addColorStop(0, '#22c55e');
      playedGradient.addColorStop(0.5, '#22c55e80');
      playedGradient.addColorStop(1, '#22c55e');

      ctx.fillStyle = playedGradient;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);

      for (let x = 0; x < width; x++) {
        const startSample = Math.floor(x * samplesPerPixel);
        const endSample = Math.floor((x + 1) * samplesPerPixel);

        let max = 0;
        for (let i = startSample; i < endSample && i < dataLength; i++) {
          const value = typeof waveData[i] === 'number' ? waveData[i] : 0;
          max = Math.max(max, Math.abs(value));
        }

        const y = (height / 2) - (max * height / 2 * 0.9);
        ctx.lineTo(x, y);
      }

      for (let x = width - 1; x >= 0; x--) {
        const startSample = Math.floor(x * samplesPerPixel);
        const endSample = Math.floor((x + 1) * samplesPerPixel);

        let max = 0;
        for (let i = startSample; i < endSample && i < dataLength; i++) {
          const value = typeof waveData[i] === 'number' ? waveData[i] : 0;
          max = Math.max(max, Math.abs(value));
        }

        const y = (height / 2) + (max * height / 2 * 0.9);
        ctx.lineTo(x, y);
      }

      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    // Draw playhead
    if (showPlayhead) {
      const playheadX = (currentTime / duration) * width;

      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();

      // Playhead glow when playing
      if (isPlaying) {
        ctx.shadowColor = '#fff';
        ctx.shadowBlur = 10;
        ctx.strokeStyle = '#fff';
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, height);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    }

    // Draw hover position
    if (hoverTime !== null) {
      const hoverX = (hoverTime / duration) * width;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(hoverX, 0);
      ctx.lineTo(hoverX, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Time tooltip
      const minutes = Math.floor(hoverTime / 60);
      const seconds = Math.floor(hoverTime % 60);
      const ms = Math.floor((hoverTime % 1) * 1000);
      const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(hoverX + 5, 5, 70, 18);
      ctx.fillStyle = '#fff';
      ctx.font = '11px monospace';
      ctx.fillText(timeStr, hoverX + 8, 18);
    }
  }, [audioData, currentTime, duration, dimensions, color, backgroundColor, showGrid, showPlayhead, isPlaying, hoverTime, generateDemoWaveform, zoom]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      drawWaveform();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [drawWaveform]);

  // Handle click to seek
  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const time = (x / dimensions.width) * duration;
    onSeek(Math.max(0, Math.min(duration, time)));
  };

  // Handle mouse move for hover
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const time = (x / dimensions.width) * duration;
    setHoverTime(Math.max(0, Math.min(duration, time)));
  };

  const handleMouseLeave = () => {
    setHoverTime(null);
  };

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: height,
        position: 'relative',
        borderRadius: '4px',
        overflow: 'hidden',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: '100%',
          height: '100%',
          cursor: onSeek ? 'pointer' : 'default',
        }}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
    </div>
  );
};
