import React, { useState } from 'react';
import { AudioQualityEngine, SampleRate, BitDepth, DitheringType, NoiseShapingType } from './AudioQualityEngine';

interface AudioQualityControlsProps {
  engine: AudioQualityEngine;
  onConfigChange?: () => void;
}

export const AudioQualityControls: React.FC<AudioQualityControlsProps> = ({
  engine,
  onConfigChange,
}) => {
  const config = engine.getConfig();
  const [sampleRate, setSampleRate] = useState<SampleRate>(config.sampleRate as SampleRate);
  const [bitDepth, setBitDepth] = useState<BitDepth>(config.bitDepth);
  const [dithering, setDithering] = useState<DitheringType>(config.dithering);
  const [noiseShaping, setNoiseShaping] = useState<NoiseShapingType>(config.noiseShaping);
  const [oversampling, setOversampling] = useState(config.oversampling);
  const [antiAliasing, setAntiAliasing] = useState(config.antiAliasing);

  const sampleRates: SampleRate[] = [22050, 32000, 44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000];
  const bitDepths: BitDepth[] = [16, 24, 32, '32-float', '64-float'];
  const ditheringTypes: DitheringType[] = ['none', 'rectangular', 'triangular', 'high-pass triangular', 'shaped', 'pow-r 1', 'pow-r 2', 'pow-r 3'];
  const noiseShapingTypes: NoiseShapingType[] = ['none', 'light', 'moderate', 'heavy', 'ultra'];

  const handleSampleRateChange = (rate: SampleRate) => {
    engine.setSampleRate(rate);
    setSampleRate(rate);
    onConfigChange?.();
  };

  const handleBitDepthChange = (depth: BitDepth) => {
    engine.setBitDepth(depth);
    setBitDepth(depth);
    onConfigChange?.();
  };

  const handleDitheringChange = (type: DitheringType) => {
    engine.setDithering(type);
    setDithering(type);
    onConfigChange?.();
  };

  const handleNoiseShapingChange = (type: NoiseShapingType) => {
    engine.setNoiseShaping(type);
    setNoiseShaping(type);
    onConfigChange?.();
  };

  const handleOversamplingChange = (factor: number) => {
    engine.setOversampling(factor);
    setOversampling(factor);
    onConfigChange?.();
  };

  const handleAntiAliasingChange = (enabled: boolean) => {
    engine.setAntiAliasing(enabled);
    setAntiAliasing(enabled);
    onConfigChange?.();
  };

  return (
    <div
      style={{
        padding: '20px',
        backgroundColor: '#1a1a1a',
        borderRadius: '8px',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <h3 style={{ marginTop: 0, color: '#fff' }}>Audio Quality Settings</h3>

      {/* Sample Rate */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
          49. Sample Rate (kHz)
        </label>
        <select
          value={sampleRate}
          onChange={(e) => handleSampleRateChange(Number(e.target.value) as SampleRate)}
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
          }}
        >
          {sampleRates.map((rate) => (
            <option key={rate} value={rate}>
              {rate / 1000} kHz
            </option>
          ))}
        </select>
      </div>

      {/* Bit Depth */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
          50. Bit Depth
        </label>
        <select
          value={bitDepth}
          onChange={(e) => handleBitDepthChange(e.target.value as BitDepth)}
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
          }}
        >
          {bitDepths.map((depth) => (
            <option key={String(depth)} value={depth}>
              {depth === '32-float' ? '32-bit Float' : depth === '64-float' ? '64-bit Float' : `${depth}-bit`}
            </option>
          ))}
        </select>
      </div>

      {/* Dithering */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
          51. Dithering
        </label>
        <select
          value={dithering}
          onChange={(e) => handleDitheringChange(e.target.value as DitheringType)}
          style={{
            width: '100%',
            padding: '8px',
            backgroundColor: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '4px',
            color: '#fff',
          }}
        >
          {ditheringTypes.map((type) => (
            <option key={type} value={type}>
              {type.charAt(0).toUpperCase() + type.slice(1).replace(/-/g, ' ')}
            </option>
          ))}
        </select>
      </div>

      {/* Noise Shaping */}
      {dithering !== 'none' && (
        <div style={{ marginBottom: '15px' }}>
          <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
            52. Noise Shaping
          </label>
          <select
            value={noiseShaping}
            onChange={(e) => handleNoiseShapingChange(e.target.value as NoiseShapingType)}
            style={{
              width: '100%',
              padding: '8px',
              backgroundColor: '#2a2a2a',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              color: '#fff',
            }}
          >
            {noiseShapingTypes.map((type) => (
              <option key={type} value={type}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Oversampling */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ color: '#aaa', fontSize: '0.9em', display: 'block', marginBottom: '5px' }}>
          56. Oversampling (x{oversampling})
        </label>
        <input
          type="range"
          min="1"
          max="8"
          step="1"
          value={oversampling}
          onChange={(e) => handleOversamplingChange(Number(e.target.value))}
          style={{ width: '100%' }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8em', color: '#888', marginTop: '4px' }}>
          <span>1x</span>
          <span>2x</span>
          <span>4x</span>
          <span>8x</span>
        </div>
      </div>

      {/* Anti-aliasing */}
      <div style={{ marginBottom: '15px' }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#aaa', fontSize: '0.9em' }}>
          <input
            type="checkbox"
            checked={antiAliasing}
            onChange={(e) => handleAntiAliasingChange(e.target.checked)}
          />
          57. Anti-aliasing Filters
        </label>
      </div>

      {/* Info */}
      <div style={{ marginTop: '20px', padding: '10px', backgroundColor: '#2a2a2a', borderRadius: '4px', fontSize: '0.85em', color: '#aaa' }}>
        <div>Current Settings:</div>
        <div>Sample Rate: {sampleRate / 1000} kHz</div>
        <div>Bit Depth: {bitDepth === '32-float' ? '32-bit Float' : bitDepth === '64-float' ? '64-bit Float' : `${bitDepth}-bit`}</div>
        <div>Dithering: {dithering}</div>
        {dithering !== 'none' && <div>Noise Shaping: {noiseShaping}</div>}
        <div>Oversampling: {oversampling}x</div>
        <div>Anti-aliasing: {antiAliasing ? 'Enabled' : 'Disabled'}</div>
      </div>
    </div>
  );
};
