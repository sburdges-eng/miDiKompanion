// AudioQualityEngine - Audio quality and format management

export interface AudioQualityConfig {
  sampleRate: number;
  bitDepth: BitDepth;
  dithering: DitheringType;
  noiseShaping: NoiseShapingType;
  oversampling: number;
  antiAliasing: boolean;
}

export type SampleRate = 
  | 22050 | 32000 | 44100 | 48000 | 88200 | 96000 | 176400 | 192000 | 352800 | 384000;

export type BitDepth = 16 | 24 | 32 | '32-float' | '64-float';

export type DitheringType = 
  | 'none' 
  | 'rectangular' 
  | 'triangular' 
  | 'high-pass triangular' 
  | 'shaped' 
  | 'pow-r 1' 
  | 'pow-r 2' 
  | 'pow-r 3';

export type NoiseShapingType = 
  | 'none' 
  | 'light' 
  | 'moderate' 
  | 'heavy' 
  | 'ultra';

export type SRCAlgorithm = 
  | 'linear' 
  | 'sinc' 
  | 'zoh' 
  | 'blep' 
  | 'min-phase' 
  | 'linear-phase' 
  | 'high-quality';

export class AudioQualityEngine {
  private config: AudioQualityConfig;

  constructor(initialConfig?: Partial<AudioQualityConfig>) {
    this.config = {
      sampleRate: 44100,
      bitDepth: 24 as BitDepth,
      dithering: 'none',
      noiseShaping: 'none',
      oversampling: 1,
      antiAliasing: true,
      ...initialConfig,
    };
  }

  // 49. Sample rate selection
  setSampleRate(rate: SampleRate): void {
    this.config.sampleRate = rate;
  }

  getSampleRate(): number {
    return this.config.sampleRate;
  }

  // 50. Bit depth selection
  setBitDepth(depth: BitDepth): void {
    this.config.bitDepth = depth;
  }

  getBitDepth(): BitDepth {
    return this.config.bitDepth;
  }

  getBitDepthNumeric(): number {
    if (typeof this.config.bitDepth === 'number') {
      return this.config.bitDepth;
    }
    // For float types, return equivalent numeric bit depth
    return this.config.bitDepth === '32-float' ? 32 : 64;
  }

  // 51. Dithering options
  setDithering(type: DitheringType): void {
    this.config.dithering = type;
  }

  getDithering(): DitheringType {
    return this.config.dithering;
  }

  // 52. Noise shaping
  setNoiseShaping(type: NoiseShapingType): void {
    this.config.noiseShaping = type;
  }

  getNoiseShaping(): NoiseShapingType {
    return this.config.noiseShaping;
  }

  // 53-55. Sample rate conversion
  convertSampleRate(
    audioBuffer: AudioBuffer,
    targetRate: number,
    algorithm: SRCAlgorithm = 'high-quality',
    realTime: boolean = false
  ): AudioBuffer {
    if (audioBuffer.sampleRate === targetRate) {
      return audioBuffer;
    }

    if (realTime) {
      return this.realTimeSRC(audioBuffer, targetRate, algorithm);
    } else {
      return this.offlineSRC(audioBuffer, targetRate, algorithm);
    }
  }

  // 54. Real-time SRC
  private realTimeSRC(
    audioBuffer: AudioBuffer,
    targetRate: number,
    _algorithm: SRCAlgorithm
  ): AudioBuffer {
    // Real-time conversion (faster, lower quality)
    const ratio = targetRate / audioBuffer.sampleRate;
    const newLength = Math.round(audioBuffer.length * ratio);
    const newBuffer = new AudioContext().createBuffer(
      audioBuffer.numberOfChannels,
      newLength,
      targetRate
    );

    // Linear interpolation for real-time
    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const inputData = audioBuffer.getChannelData(channel);
      const outputData = newBuffer.getChannelData(channel);

      for (let i = 0; i < newLength; i++) {
        const srcIndex = i / ratio;
        const index = Math.floor(srcIndex);
        const fraction = srcIndex - index;

        if (index + 1 < inputData.length) {
          outputData[i] = inputData[index] * (1 - fraction) + inputData[index + 1] * fraction;
        } else {
          outputData[i] = inputData[index] || 0;
        }
      }
    }

    return newBuffer;
  }

  // 55. Offline SRC (high quality)
  private offlineSRC(
    audioBuffer: AudioBuffer,
    targetRate: number,
    algorithm: SRCAlgorithm
  ): AudioBuffer {
    // Use sinc interpolation for high-quality offline conversion
    const ratio = targetRate / audioBuffer.sampleRate;
    const newLength = Math.round(audioBuffer.length * ratio);
    const newBuffer = new AudioContext().createBuffer(
      audioBuffer.numberOfChannels,
      newLength,
      targetRate
    );

    // Sinc-based resampling for high quality
    const sincFilterSize = algorithm === 'high-quality' ? 64 : 32;

    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const inputData = audioBuffer.getChannelData(channel);
      const outputData = newBuffer.getChannelData(channel);

      for (let i = 0; i < newLength; i++) {
        const srcIndex = i / ratio;
        let sum = 0;
        let weightSum = 0;

        for (let j = -sincFilterSize; j <= sincFilterSize; j++) {
          const index = Math.round(srcIndex + j);
          if (index >= 0 && index < inputData.length) {
            const x = srcIndex - index;
            const sinc = this.sinc(x);
            const window = this.blackmanWindow(x / sincFilterSize);
            const weight = sinc * window;
            sum += inputData[index] * weight;
            weightSum += weight;
          }
        }

        outputData[i] = weightSum > 0 ? sum / weightSum : 0;
      }
    }

    return newBuffer;
  }

  // 56. Oversampling
  setOversampling(factor: number): void {
    this.config.oversampling = Math.max(1, Math.min(8, factor));
  }

  getOversampling(): number {
    return this.config.oversampling;
  }

  applyOversampling(audioBuffer: AudioBuffer): AudioBuffer {
    if (this.config.oversampling === 1) {
      return audioBuffer;
    }

    // Upsample
    const upsampledRate = audioBuffer.sampleRate * this.config.oversampling;
    const upsampled = this.convertSampleRate(audioBuffer, upsampledRate, 'linear-phase');

    // Process at higher rate (would apply effects here)

    // Downsample with anti-aliasing
    const downsampled = this.convertSampleRate(upsampled, audioBuffer.sampleRate, 'linear-phase');

    return downsampled;
  }

  // 57. Anti-aliasing filters
  setAntiAliasing(enabled: boolean): void {
    this.config.antiAliasing = enabled;
  }

  applyAntiAliasingFilter(audioBuffer: AudioBuffer, cutoffRatio: number = 0.45): AudioBuffer {
    if (!this.config.antiAliasing) {
      return audioBuffer;
    }

    const cutoff = audioBuffer.sampleRate * cutoffRatio;
    return this.applyLowPassFilter(audioBuffer, cutoff);
  }

  private applyLowPassFilter(audioBuffer: AudioBuffer, cutoff: number): AudioBuffer {
    // Simple IIR low-pass filter
    const nyquist = audioBuffer.sampleRate / 2;
    const normalizedCutoff = cutoff / nyquist;
    const rc = 1.0 / (2.0 * Math.PI * normalizedCutoff);
    const dt = 1.0 / audioBuffer.sampleRate;
    const alpha = dt / (rc + dt);

    const filtered = new AudioContext().createBuffer(
      audioBuffer.numberOfChannels,
      audioBuffer.length,
      audioBuffer.sampleRate
    );

    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const inputData = audioBuffer.getChannelData(channel);
      const outputData = filtered.getChannelData(channel);
      let lastOutput = 0;

      for (let i = 0; i < inputData.length; i++) {
        lastOutput = lastOutput + alpha * (inputData[i] - lastOutput);
        outputData[i] = lastOutput;
      }
    }

    return filtered;
  }

  // Apply dithering
  applyDithering(audioBuffer: AudioBuffer, targetBitDepth: number | BitDepth): AudioBuffer {
    const numericDepth = typeof targetBitDepth === 'number' 
      ? targetBitDepth 
      : (targetBitDepth === '32-float' ? 32 : 64);
    
    if (this.config.dithering === 'none' || numericDepth >= 32) {
      return audioBuffer;
    }

    const dithered = new AudioContext().createBuffer(
      audioBuffer.numberOfChannels,
      audioBuffer.length,
      audioBuffer.sampleRate
    );

    const lsb = 1.0 / Math.pow(2, numericDepth - 1);
    const noise = this.generateDitherNoise(audioBuffer.length, this.config.dithering);

    for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
      const inputData = audioBuffer.getChannelData(channel);
      const outputData = dithered.getChannelData(channel);

      for (let i = 0; i < inputData.length; i++) {
        let sample = inputData[i] + noise[i] * lsb;
        
        // Quantize
        sample = Math.round(sample / lsb) * lsb;
        outputData[i] = Math.max(-1, Math.min(1, sample));
      }
    }

    return dithered;
  }

  private generateDitherNoise(length: number, type: DitheringType): Float32Array {
    const noise = new Float32Array(length);

    switch (type) {
      case 'rectangular':
        for (let i = 0; i < length; i++) {
          noise[i] = Math.random() * 2 - 1;
        }
        break;
      case 'triangular':
        for (let i = 0; i < length; i++) {
          noise[i] = (Math.random() + Math.random() - 1) * 2;
        }
        break;
      case 'high-pass triangular':
        // High-pass filtered triangular
        let last = 0;
        for (let i = 0; i < length; i++) {
          const triangular = (Math.random() + Math.random() - 1) * 2;
          noise[i] = triangular - last;
          last = triangular;
        }
        break;
      case 'shaped':
      case 'pow-r 1':
      case 'pow-r 2':
      case 'pow-r 3':
        // Shaped dithering (frequency-shaped noise)
        for (let i = 0; i < length; i++) {
          noise[i] = (Math.random() + Math.random() - 1) * 2;
        }
        // Apply noise shaping filter
        this.applyNoiseShaping(noise, this.config.noiseShaping);
        break;
    }

    return noise;
  }

  private applyNoiseShaping(noise: Float32Array, type: NoiseShapingType): void {
    // Simple noise shaping filter
    let lastError = 0;
    const shapingAmount = type === 'light' ? 0.5 : type === 'moderate' ? 0.75 : type === 'heavy' ? 0.9 : 0.95;

    for (let i = 0; i < noise.length; i++) {
      const error = noise[i] - lastError;
      noise[i] = noise[i] + error * shapingAmount;
      lastError = error;
    }
  }

  // Helper functions
  private sinc(x: number): number {
    if (x === 0) return 1;
    return Math.sin(Math.PI * x) / (Math.PI * x);
  }

  private blackmanWindow(x: number): number {
    const a0 = 0.42;
    const a1 = 0.5;
    const a2 = 0.08;
    return a0 - a1 * Math.cos(2 * Math.PI * x) + a2 * Math.cos(4 * Math.PI * x);
  }

  // Get full config
  getConfig(): AudioQualityConfig {
    return { ...this.config };
  }

  // Set full config
  setConfig(config: Partial<AudioQualityConfig>): void {
    this.config = { ...this.config, ...config };
  }
}
