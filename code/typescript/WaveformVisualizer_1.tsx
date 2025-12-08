import React, { useRef, useEffect, useState } from 'react';
import * as Tone from 'tone';

interface WaveformVisualizerProps {
  audioSource?: Tone.ToneAudioNode;
  width?: number;
  height?: number;
  color?: string;
  syncToPlayback?: boolean;
  isPlaying?: boolean;
}

export const WaveformVisualizer: React.FC<WaveformVisualizerProps> = ({
  audioSource,
  width = 800,
  height = 200,
  color = '#6366f1',
  syncToPlayback = true,
  isPlaying = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const analyserRef = useRef<Tone.Analyser | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [waveformData, setWaveformData] = useState<Float32Array>(new Float32Array(1024));

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Create analyser if audio source provided
    if (audioSource) {
      analyserRef.current = new Tone.Analyser('waveform', 1024);
      audioSource.connect(analyserRef.current);
    }

    const drawWaveform = () => {
      if (!ctx || !canvas) return;

      // Clear canvas
      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Get waveform data
      let data = waveformData;
      if (analyserRef.current && isPlaying) {
        data = analyserRef.current.getValue() as Float32Array;
        setWaveformData(data);
      }

      // Draw waveform
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      const sliceWidth = canvas.width / data.length;
      let x = 0;

      for (let i = 0; i < data.length; i++) {
        const v = data[i] * 0.5 + 0.5; // Normalize -1 to 1 -> 0 to 1
        const y = v * canvas.height;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.stroke();

      // Draw center line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, canvas.height / 2);
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();

      // Draw grid lines
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      for (let i = 0; i < 10; i++) {
        const y = (canvas.height / 10) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }

      // Add glow effect
      ctx.shadowBlur = 10;
      ctx.shadowColor = color;
      ctx.stroke();

      if (syncToPlayback && isPlaying) {
        animationFrameRef.current = requestAnimationFrame(drawWaveform);
      }
    };

    if (syncToPlayback && isPlaying) {
      drawWaveform();
    } else {
      drawWaveform(); // Draw once even when not playing
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (analyserRef.current) {
        analyserRef.current.dispose();
      }
    };
  }, [audioSource, isPlaying, syncToPlayback, color, waveformData]);

  return (
    <div style={{ position: 'relative', width, height }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          width: '100%',
          height: '100%',
          backgroundColor: '#0a0a0a',
          borderRadius: '4px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      />
      {!isPlaying && (
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
          Waveform will appear during playback
        </div>
      )}
    </div>
  );
};
