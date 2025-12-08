import React, { useRef, useEffect } from 'react';

interface BrushstrokeCanvasProps {
  width?: number;
  height?: number;
  intensity?: number; // 0-1, controls animation intensity
  color?: string;
  syncToAudio?: boolean;
  audioLevel?: number; // 0-1, audio level for sync
}

export const BrushstrokeCanvas: React.FC<BrushstrokeCanvasProps> = ({
  width = 800,
  height = 400,
  intensity = 0.5,
  color = '#6366f1',
  syncToAudio = false,
  audioLevel = 0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const timeRef = useRef<number>(0);
  const brushstrokesRef = useRef<Array<{
    x: number;
    y: number;
    angle: number;
    length: number;
    width: number;
    opacity: number;
    speed: number;
  }>>([]);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Initialize brushstrokes
    const numStrokes = 20;
    brushstrokesRef.current = Array.from({ length: numStrokes }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      angle: Math.random() * Math.PI * 2,
      length: 50 + Math.random() * 100,
      width: 2 + Math.random() * 8,
      opacity: 0.3 + Math.random() * 0.7,
      speed: 0.5 + Math.random() * 2,
    }));

    const drawBrushstroke = (
      x: number,
      y: number,
      angle: number,
      length: number,
      strokeWidth: number,
      opacity: number
    ) => {
      ctx.save();
      ctx.globalAlpha = opacity;
      ctx.strokeStyle = color;
      ctx.lineWidth = strokeWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Create brushstroke path with variation
      ctx.beginPath();
      const segments = 10;
      let currentX = x;
      let currentY = y;

      ctx.moveTo(currentX, currentY);

      for (let i = 1; i <= segments; i++) {
        const segmentLength = length / segments;

        // Add wobble for hand-drawn effect
        const wobble = Math.sin(timeRef.current * 2 + i) * 2;
        const perpAngle = angle + Math.PI / 2;
        const wobbleX = Math.cos(perpAngle) * wobble;
        const wobbleY = Math.sin(perpAngle) * wobble;

        currentX += Math.cos(angle) * segmentLength + wobbleX;
        currentY += Math.sin(angle) * segmentLength + wobbleY;

        ctx.lineTo(currentX, currentY);
      }

      ctx.stroke();

      // Add bristle texture
      ctx.globalAlpha = opacity * 0.3;
      for (let i = 0; i < 5; i++) {
        const offset = (i - 2) * 2;
        const perpAngle = angle + Math.PI / 2;
        const offsetX = Math.cos(perpAngle) * offset;
        const offsetY = Math.sin(perpAngle) * offset;

        ctx.beginPath();
        ctx.moveTo(x + offsetX, y + offsetY);
        let bx = x;
        let by = y;

        for (let j = 1; j <= segments; j++) {
          const segmentLength = length / segments;
          const wobble = Math.sin(timeRef.current * 2 + j) * 1;
          const perpAngle2 = angle + Math.PI / 2;
          const wobbleX = Math.cos(perpAngle2) * wobble;
          const wobbleY = Math.sin(perpAngle2) * wobble;

          bx += Math.cos(angle) * segmentLength + wobbleX;
          by += Math.sin(angle) * segmentLength + wobbleY;

          ctx.lineTo(bx + offsetX, by + offsetY);
        }

        ctx.stroke();
      }

      ctx.restore();
    };

    const animate = () => {
      if (!ctx) return;

      // Clear with fade for trail effect
      ctx.fillStyle = 'rgba(10, 10, 10, 0.1)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      timeRef.current += 0.016; // ~60fps

      // Update and draw brushstrokes
      brushstrokesRef.current.forEach((stroke) => {
        // Move stroke
        stroke.x += Math.cos(stroke.angle) * stroke.speed * intensity;
        stroke.y += Math.sin(stroke.angle) * stroke.speed * intensity;

        // Wrap around edges
        if (stroke.x < 0) stroke.x = width;
        if (stroke.x > width) stroke.x = 0;
        if (stroke.y < 0) stroke.y = height;
        if (stroke.y > height) stroke.y = 0;

        // Modify based on audio if synced
        let strokeIntensity = intensity;
        let strokeOpacity = stroke.opacity;
        if (syncToAudio && audioLevel > 0) {
          strokeIntensity *= 1 + audioLevel * 2;
          strokeOpacity *= 1 + audioLevel;
          stroke.angle += audioLevel * 0.1;
        }

        // Draw brushstroke
        drawBrushstroke(
          stroke.x,
          stroke.y,
          stroke.angle,
          stroke.length * strokeIntensity,
          stroke.width * (1 + audioLevel),
          Math.min(1, strokeOpacity)
        );
      });

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height, intensity, color, syncToAudio, audioLevel]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        width: '100%',
        height: '100%',
        backgroundColor: '#0a0a0a',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    />
  );
};
