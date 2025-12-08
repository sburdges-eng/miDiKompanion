import React, { useState, useEffect } from 'react';
import { ShaderCanvas } from './ShaderCanvas';

interface ShaderViewerProps {
  shaderName: 'brushstroke' | 'handdrawn';
  width?: number;
  height?: number;
  animated?: boolean;
  intensity?: number;
}

export const ShaderViewer: React.FC<ShaderViewerProps> = ({
  shaderName,
  width = 800,
  height = 400,
  animated = true,
  intensity = 0.5,
}) => {
  const [shaderSource, setShaderSource] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Use inline shaders (fallback to file loading in production)
    if (shaderName === 'brushstroke') {
      setShaderSource(getBrushstrokeShader());
    } else if (shaderName === 'handdrawn') {
      setShaderSource(getHanddrawnShader());
    }
    setLoading(false);
  }, [shaderName]);

  if (loading) {
    return (
      <div
        style={{
          width,
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#0a0a0a',
          borderRadius: '8px',
          color: '#888',
        }}
      >
        Loading shader...
      </div>
    );
  }

  const uniforms: Record<string, number | number[]> =
    shaderName === 'brushstroke'
      ? {
          strokePosition: 0.5 + Math.sin(Date.now() / 1000) * 0.2,
          strokeWidth: 0.1,
          strokeIntensity: intensity,
          strokeAngle: Date.now() / 1000,
          wetness: 0.3,
          bristleSpread: 0.5,
        }
      : {
          lineColor: [0.0, 1.0, 1.0, 0.5],
          backgroundColor: [0.05, 0.1, 0.2, 1.0],
          wobbleIntensity: intensity,
          gridSpacing: 50.0,
        };

  return (
    <ShaderCanvas
      shaderSource={shaderSource}
      width={width}
      height={height}
      uniforms={uniforms}
      animated={animated}
    />
  );
};

// Fallback shader sources (inline)
const getBrushstrokeShader = (): string => {
  return `
precision mediump float;
varying vec2 fragTexCoord;
uniform float time;
uniform vec2 resolution;
uniform float strokePosition;
uniform float strokeWidth;
uniform float strokeIntensity;
uniform float strokeAngle;
uniform float wetness;
uniform float bristleSpread;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    vec2 uv = fragTexCoord;
    vec3 canvasColor = vec3(0.95, 0.93, 0.9);
    float paperTex = noise(uv * 100.0) * 0.03;
    canvasColor -= paperTex;
    
    float c = cos(strokeAngle * 0.1);
    float s = sin(strokeAngle * 0.1);
    vec2 ruv = vec2(
        (uv.x - 0.5) * c - (uv.y - 0.5) * s + 0.5,
        (uv.x - 0.5) * s + (uv.y - 0.5) * c + 0.5
    );
    
    float yPos = strokePosition;
    float dist = abs(ruv.y - yPos);
    float stroke = smoothstep(strokeWidth * 0.5, strokeWidth * 0.2, dist);
    stroke *= 0.8 + 0.2 * noise(ruv * 30.0);
    
    vec3 paintColor = vec3(0.2, 0.4, 0.8);
    paintColor.r += strokeIntensity * 0.3;
    
    vec3 finalColor = mix(canvasColor, paintColor, stroke);
    float glow = stroke * strokeIntensity * 0.5;
    finalColor += vec3(0.8, 0.6, 1.0) * glow;
    
    float vignette = 1.0 - length(uv - 0.5) * 0.4;
    finalColor *= vignette;
    
    gl_FragColor = vec4(finalColor, 1.0);
}
  `;
};

const getHanddrawnShader = (): string => {
  return `
precision mediump float;
varying vec2 fragTexCoord;
uniform float time;
uniform vec2 resolution;
uniform float wobbleIntensity;
uniform float gridSpacing;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 4; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

void main() {
    vec2 uv = fragTexCoord;
    vec3 color = vec3(0.05, 0.1, 0.2);
    
    float wobbleX = fbm(vec2(uv.y * 5.0, time * 0.3)) * wobbleIntensity * 0.02;
    float wobbleY = fbm(vec2(uv.x * 5.0, time * 0.3 + 100.0)) * wobbleIntensity * 0.02;
    vec2 wobbledUV = uv + vec2(wobbleX, wobbleY);
    
    float gridSize = gridSpacing / resolution.x;
    float vLine = mod(wobbledUV.x, gridSize);
    float vLineAlpha = 1.0 - smoothstep(0.0, 0.003, min(vLine, gridSize - vLine));
    float hLine = mod(wobbledUV.y, gridSize);
    float hLineAlpha = 1.0 - smoothstep(0.0, 0.003, min(hLine, gridSize - hLine));
    float lineAlpha = max(vLineAlpha, hLineAlpha);
    lineAlpha *= 0.3 + 0.1 * noise(wobbledUV * 100.0);
    
    color = mix(color, vec3(0.0, 1.0, 1.0), lineAlpha);
    
    float vignette = 1.0 - length(uv - 0.5) * 0.5;
    color *= vignette;
    
    float paperNoise = noise(uv * 200.0) * 0.02;
    color += vec3(paperNoise);
    
    float pulse = sin(time * 0.5) * 0.1 + 0.9;
    color *= pulse;
    
    gl_FragColor = vec4(color, 1.0);
}
  `;
};
