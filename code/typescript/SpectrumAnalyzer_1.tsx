/**
 * SpectrumAnalyzer - Real-time frequency spectrum visualization
 * Features: FFT visualization, peak hold, multiple display modes
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

type DisplayMode = 'bars' | 'line' | 'filled' | 'mirror';

interface SpectrumAnalyzerProps {
  audioContext?: AudioContext;
  analyzerNode?: AnalyserNode;
  mode?: DisplayMode;
  barCount?: number;
  color?: string;
  peakColor?: string;
  backgroundColor?: string;
  height?: number;
  showPeaks?: boolean;
  smoothing?: number;
  minDecibels?: number;
  maxDecibels?: number;
}

export const SpectrumAnalyzer: React.FC<SpectrumAnalyzerProps> = ({
  audioContext: _audioContext,
  analyzerNode,
  mode = 'bars',
  barCount = 64,
  color = '#6366f1',
  peakColor = '#f59e0b',
  backgroundColor = '#0f0f0f',
  height = 150,
  showPeaks = true,
  smoothing = 0.8,
  minDecibels: _minDecibels = -90,
  maxDecibels: _maxDecibels = -10,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationRef = useRef<number | undefined>(undefined);
  const peaksRef = useRef<number[]>(new Array(barCount).fill(0));
  const peakHoldRef = useRef<number[]>(new Array(barCount).fill(0));
  const [dimensions, setDimensions] = useState({ width: 400, height });

  // Demo data generator for when no audio context is provided
  const generateDemoData = useCallback((): Uint8Array => {
    const data = new Uint8Array(barCount);
    const time = Date.now() / 1000;

    for (let i = 0; i < barCount; i++) {
      const freq = i / barCount;
      // Simulate frequency distribution (more energy in low/mid frequencies)
      const base = Math.exp(-freq * 2) * 200;
      // Add some movement
      const movement = Math.sin(time * 2 + i * 0.3) * 30;
      const noise = Math.random() * 20;
      data[i] = Math.max(0, Math.min(255, base + movement + noise));
    }

    return data;
  }, [barCount]);

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

  // Draw spectrum
  const drawSpectrum = useCallback(() => {
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

    // Get frequency data
    let frequencyData: Uint8Array;

    if (analyzerNode) {
      frequencyData = new Uint8Array(analyzerNode.frequencyBinCount);
      analyzerNode.getByteFrequencyData(frequencyData);
    } else {
      frequencyData = generateDemoData();
    }

    // Calculate bar dimensions
    const barWidth = width / barCount;
    const barGap = Math.max(1, barWidth * 0.1);
    const actualBarWidth = barWidth - barGap;

    // Frequency bin to bar index mapping (logarithmic scale for better visualization)
    const getBarValue = (barIndex: number): number => {
      const binCount = frequencyData.length;
      // Logarithmic mapping for more musical frequency representation
      const logMin = Math.log(1);
      const logMax = Math.log(binCount);
      const logValue = logMin + (logMax - logMin) * (barIndex / barCount);
      const binIndex = Math.min(binCount - 1, Math.floor(Math.exp(logValue)));

      // Average nearby bins for smoother visualization
      let sum = 0;
      let count = 0;
      const spread = Math.max(1, Math.floor(binCount / barCount / 2));

      for (let i = Math.max(0, binIndex - spread); i <= Math.min(binCount - 1, binIndex + spread); i++) {
        sum += frequencyData[i];
        count++;
      }

      return sum / count;
    };

    // Create gradient
    const gradient = ctx.createLinearGradient(0, height, 0, 0);
    gradient.addColorStop(0, color);
    gradient.addColorStop(0.5, color);
    gradient.addColorStop(0.8, '#f59e0b');
    gradient.addColorStop(1, '#ef4444');

    // Draw based on mode
    switch (mode) {
      case 'bars':
        for (let i = 0; i < barCount; i++) {
          const value = getBarValue(i);
          const barHeight = (value / 255) * height;
          const x = i * barWidth;

          // Apply smoothing to current values
          const smoothedHeight = peaksRef.current[i] * smoothing + barHeight * (1 - smoothing);
          peaksRef.current[i] = smoothedHeight;

          // Draw bar
          ctx.fillStyle = gradient;
          ctx.fillRect(x, height - smoothedHeight, actualBarWidth, smoothedHeight);

          // Update and draw peak
          if (showPeaks) {
            if (smoothedHeight > peakHoldRef.current[i]) {
              peakHoldRef.current[i] = smoothedHeight;
            } else {
              peakHoldRef.current[i] = Math.max(0, peakHoldRef.current[i] - 1);
            }

            ctx.fillStyle = peakColor;
            ctx.fillRect(x, height - peakHoldRef.current[i] - 3, actualBarWidth, 3);
          }
        }
        break;

      case 'line':
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i < barCount; i++) {
          const value = getBarValue(i);
          const y = height - (value / 255) * height;
          const x = (i / barCount) * width;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.stroke();

        // Draw glow
        ctx.strokeStyle = color + '40';
        ctx.lineWidth = 6;
        ctx.stroke();
        break;

      case 'filled':
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.moveTo(0, height);

        for (let i = 0; i < barCount; i++) {
          const value = getBarValue(i);
          const y = height - (value / 255) * height;
          const x = (i / barCount) * width;
          ctx.lineTo(x, y);
        }

        ctx.lineTo(width, height);
        ctx.closePath();
        ctx.fill();

        // Draw outline
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let i = 0; i < barCount; i++) {
          const value = getBarValue(i);
          const y = height - (value / 255) * height;
          const x = (i / barCount) * width;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.stroke();
        break;

      case 'mirror':
        const midY = height / 2;

        for (let i = 0; i < barCount; i++) {
          const value = getBarValue(i);
          const barHeight = (value / 255) * (height / 2);
          const x = i * barWidth;

          // Apply smoothing
          const smoothedHeight = peaksRef.current[i] * smoothing + barHeight * (1 - smoothing);
          peaksRef.current[i] = smoothedHeight;

          // Draw top half
          ctx.fillStyle = gradient;
          ctx.fillRect(x, midY - smoothedHeight, actualBarWidth, smoothedHeight);

          // Draw bottom half (mirrored)
          const mirrorGradient = ctx.createLinearGradient(0, midY, 0, height);
          mirrorGradient.addColorStop(0, color);
          mirrorGradient.addColorStop(0.5, color + '80');
          mirrorGradient.addColorStop(1, color + '20');
          ctx.fillStyle = mirrorGradient;
          ctx.fillRect(x, midY, actualBarWidth, smoothedHeight);
        }

        // Draw center line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(width, midY);
        ctx.stroke();
        break;
    }

    // Draw frequency labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '9px monospace';
    const freqLabels = ['20', '100', '500', '1k', '5k', '10k', '20k'];
    freqLabels.forEach((label, i) => {
      const x = (i / (freqLabels.length - 1)) * (width - 30) + 5;
      ctx.fillText(label, x, height - 4);
    });

    // Draw dB scale
    ctx.textAlign = 'right';
    for (let db = 0; db >= -60; db -= 20) {
      const y = height * (-db / 60);
      ctx.fillText(`${db}dB`, width - 5, y + 4);
    }
    ctx.textAlign = 'left';
  }, [analyzerNode, barCount, color, peakColor, backgroundColor, dimensions, mode, showPeaks, smoothing, generateDemoData]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      drawSpectrum();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [drawSpectrum]);

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
        }}
      />
    </div>
  );
};
