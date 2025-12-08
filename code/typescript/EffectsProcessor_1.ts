/**
 * Effects Processor - DSP Effects and Plugin System
 * Implements Part 6 of the DAW specification (Items 381-517)
 */

// Effect parameter interface
export interface EffectParameter {
  name: string;
  value: number;
  min: number;
  max: number;
  default: number;
  unit: string;
  curve: 'linear' | 'exponential' | 'logarithmic';
}

// Base effect interface
export interface Effect {
  id: string;
  name: string;
  type: string;
  category: 'eq' | 'dynamics' | 'timebased' | 'modulation' | 'distortion' | 'pitch' | 'utility' | 'restoration';
  bypassed: boolean;
  parameters: Record<string, EffectParameter>;
  inputNode: AudioNode | null;
  outputNode: AudioNode | null;
  process(input: AudioNode, output: AudioNode, context: AudioContext): void;
  setParameter(name: string, value: number): void;
  getParameter(name: string): number;
  destroy(): void;
}

// EQ Band
interface EQBand {
  type: 'lowshelf' | 'highshelf' | 'peaking' | 'lowpass' | 'highpass' | 'bandpass' | 'notch' | 'allpass';
  frequency: number;
  gain: number;
  q: number;
  filter: BiquadFilterNode | null;
}

// Parametric EQ (Items 381-398)
export class ParametricEQ implements Effect {
  id: string;
  name: string = 'Parametric EQ';
  type: string = 'parametricEQ';
  category: 'eq' = 'eq';
  bypassed: boolean = false;
  parameters: Record<string, EffectParameter> = {};
  inputNode: GainNode | null = null;
  outputNode: GainNode | null = null;

  private bands: EQBand[] = [];
  protected context: AudioContext | null = null;

  constructor(id: string, numBands: number = 5) {
    this.id = id;

    // Initialize bands
    const defaultFreqs = [80, 250, 1000, 4000, 12000];
    for (let i = 0; i < numBands; i++) {
      this.bands.push({
        type: i === 0 ? 'lowshelf' : i === numBands - 1 ? 'highshelf' : 'peaking',
        frequency: defaultFreqs[i] || 1000,
        gain: 0,
        q: 1,
        filter: null,
      });

      // Create parameters
      this.parameters[`band${i}_freq`] = {
        name: `Band ${i + 1} Frequency`,
        value: defaultFreqs[i] || 1000,
        min: 20,
        max: 20000,
        default: defaultFreqs[i] || 1000,
        unit: 'Hz',
        curve: 'logarithmic',
      };
      this.parameters[`band${i}_gain`] = {
        name: `Band ${i + 1} Gain`,
        value: 0,
        min: -24,
        max: 24,
        default: 0,
        unit: 'dB',
        curve: 'linear',
      };
      this.parameters[`band${i}_q`] = {
        name: `Band ${i + 1} Q`,
        value: 1,
        min: 0.1,
        max: 18,
        default: 1,
        unit: '',
        curve: 'logarithmic',
      };
    }
  }

  process(input: AudioNode, output: AudioNode, context: AudioContext): void {
    this.context = context;
    this.inputNode = context.createGain();
    this.outputNode = context.createGain();

    input.connect(this.inputNode);

    let lastNode: AudioNode = this.inputNode;

    // Create filter chain
    this.bands.forEach((band, _i) => {
      band.filter = context.createBiquadFilter();
      band.filter.type = band.type;
      band.filter.frequency.value = band.frequency;
      band.filter.gain.value = band.gain;
      band.filter.Q.value = band.q;

      lastNode.connect(band.filter);
      lastNode = band.filter;
    });

    lastNode.connect(this.outputNode);
    this.outputNode.connect(output);
  }

  setParameter(name: string, value: number): void {
    if (this.parameters[name]) {
      this.parameters[name].value = value;

      // Apply to filter
      const match = name.match(/band(\d+)_(\w+)/);
      if (match) {
        const bandIndex = parseInt(match[1]);
        const param = match[2];
        const band = this.bands[bandIndex];
        if (band?.filter) {
          switch (param) {
            case 'freq':
              band.frequency = value;
              band.filter.frequency.value = value;
              break;
            case 'gain':
              band.gain = value;
              band.filter.gain.value = value;
              break;
            case 'q':
              band.q = value;
              band.filter.Q.value = value;
              break;
          }
        }
      }
    }
  }

  getParameter(name: string): number {
    return this.parameters[name]?.value ?? 0;
  }

  destroy(): void {
    this.bands.forEach(band => band.filter?.disconnect());
    this.inputNode?.disconnect();
    this.outputNode?.disconnect();
  }
}

// Compressor (Items 399-420)
export class Compressor implements Effect {
  id: string;
  name: string = 'Compressor';
  type: string = 'compressor';
  category: 'dynamics' = 'dynamics';
  bypassed: boolean = false;
  parameters: Record<string, EffectParameter>;
  inputNode: GainNode | null = null;
  outputNode: GainNode | null = null;

  private compressor: DynamicsCompressorNode | null = null;
  private makeupGain: GainNode | null = null;

  constructor(id: string) {
    this.id = id;
    this.parameters = {
      threshold: { name: 'Threshold', value: -24, min: -60, max: 0, default: -24, unit: 'dB', curve: 'linear' },
      ratio: { name: 'Ratio', value: 4, min: 1, max: 20, default: 4, unit: ':1', curve: 'logarithmic' },
      attack: { name: 'Attack', value: 10, min: 0, max: 200, default: 10, unit: 'ms', curve: 'logarithmic' },
      release: { name: 'Release', value: 100, min: 10, max: 1000, default: 100, unit: 'ms', curve: 'logarithmic' },
      knee: { name: 'Knee', value: 6, min: 0, max: 40, default: 6, unit: 'dB', curve: 'linear' },
      makeup: { name: 'Makeup Gain', value: 0, min: -12, max: 24, default: 0, unit: 'dB', curve: 'linear' },
    };
  }

  process(input: AudioNode, output: AudioNode, context: AudioContext): void {
    this.inputNode = context.createGain();
    this.compressor = context.createDynamicsCompressor();
    this.makeupGain = context.createGain();
    this.outputNode = context.createGain();

    this.compressor.threshold.value = this.parameters.threshold.value;
    this.compressor.ratio.value = this.parameters.ratio.value;
    this.compressor.attack.value = this.parameters.attack.value / 1000;
    this.compressor.release.value = this.parameters.release.value / 1000;
    this.compressor.knee.value = this.parameters.knee.value;
    this.makeupGain.gain.value = Math.pow(10, this.parameters.makeup.value / 20);

    input.connect(this.inputNode);
    this.inputNode.connect(this.compressor);
    this.compressor.connect(this.makeupGain);
    this.makeupGain.connect(this.outputNode);
    this.outputNode.connect(output);
  }

  setParameter(name: string, value: number): void {
    if (this.parameters[name]) {
      this.parameters[name].value = value;

      if (this.compressor) {
        switch (name) {
          case 'threshold':
            this.compressor.threshold.value = value;
            break;
          case 'ratio':
            this.compressor.ratio.value = value;
            break;
          case 'attack':
            this.compressor.attack.value = value / 1000;
            break;
          case 'release':
            this.compressor.release.value = value / 1000;
            break;
          case 'knee':
            this.compressor.knee.value = value;
            break;
          case 'makeup':
            if (this.makeupGain) {
              this.makeupGain.gain.value = Math.pow(10, value / 20);
            }
            break;
        }
      }
    }
  }

  getParameter(name: string): number {
    return this.parameters[name]?.value ?? 0;
  }

  getReduction(): number {
    return this.compressor?.reduction ?? 0;
  }

  destroy(): void {
    this.inputNode?.disconnect();
    this.compressor?.disconnect();
    this.makeupGain?.disconnect();
    this.outputNode?.disconnect();
  }
}

// Delay (Items 421-428)
export class Delay implements Effect {
  id: string;
  name: string = 'Delay';
  type: string = 'delay';
  category: 'timebased' = 'timebased';
  bypassed: boolean = false;
  parameters: Record<string, EffectParameter>;
  inputNode: GainNode | null = null;
  outputNode: GainNode | null = null;

  private delayNode: DelayNode | null = null;
  private feedbackNode: GainNode | null = null;
  private wetNode: GainNode | null = null;
  private dryNode: GainNode | null = null;
  private filterNode: BiquadFilterNode | null = null;

  constructor(id: string) {
    this.id = id;
    this.parameters = {
      time: { name: 'Delay Time', value: 375, min: 1, max: 2000, default: 375, unit: 'ms', curve: 'logarithmic' },
      feedback: { name: 'Feedback', value: 40, min: 0, max: 95, default: 40, unit: '%', curve: 'linear' },
      mix: { name: 'Mix', value: 30, min: 0, max: 100, default: 30, unit: '%', curve: 'linear' },
      lowcut: { name: 'Low Cut', value: 200, min: 20, max: 2000, default: 200, unit: 'Hz', curve: 'logarithmic' },
      highcut: { name: 'High Cut', value: 8000, min: 1000, max: 20000, default: 8000, unit: 'Hz', curve: 'logarithmic' },
      pingpong: { name: 'Ping Pong', value: 0, min: 0, max: 1, default: 0, unit: '', curve: 'linear' },
    };
  }

  process(input: AudioNode, output: AudioNode, context: AudioContext): void {
    this.inputNode = context.createGain();
    this.delayNode = context.createDelay(5);
    this.feedbackNode = context.createGain();
    this.wetNode = context.createGain();
    this.dryNode = context.createGain();
    this.filterNode = context.createBiquadFilter();
    this.outputNode = context.createGain();

    this.delayNode.delayTime.value = this.parameters.time.value / 1000;
    this.feedbackNode.gain.value = this.parameters.feedback.value / 100;
    this.wetNode.gain.value = this.parameters.mix.value / 100;
    this.dryNode.gain.value = 1 - this.parameters.mix.value / 100;
    this.filterNode.type = 'lowpass';
    this.filterNode.frequency.value = this.parameters.highcut.value;

    input.connect(this.inputNode);

    // Dry path
    this.inputNode.connect(this.dryNode);
    this.dryNode.connect(this.outputNode);

    // Wet path with feedback
    this.inputNode.connect(this.delayNode);
    this.delayNode.connect(this.filterNode);
    this.filterNode.connect(this.wetNode);
    this.filterNode.connect(this.feedbackNode);
    this.feedbackNode.connect(this.delayNode);
    this.wetNode.connect(this.outputNode);

    this.outputNode.connect(output);
  }

  setParameter(name: string, value: number): void {
    if (this.parameters[name]) {
      this.parameters[name].value = value;

      switch (name) {
        case 'time':
          if (this.delayNode) this.delayNode.delayTime.value = value / 1000;
          break;
        case 'feedback':
          if (this.feedbackNode) this.feedbackNode.gain.value = value / 100;
          break;
        case 'mix':
          if (this.wetNode) this.wetNode.gain.value = value / 100;
          if (this.dryNode) this.dryNode.gain.value = 1 - value / 100;
          break;
        case 'highcut':
          if (this.filterNode) this.filterNode.frequency.value = value;
          break;
      }
    }
  }

  getParameter(name: string): number {
    return this.parameters[name]?.value ?? 0;
  }

  destroy(): void {
    this.inputNode?.disconnect();
    this.delayNode?.disconnect();
    this.feedbackNode?.disconnect();
    this.wetNode?.disconnect();
    this.dryNode?.disconnect();
    this.filterNode?.disconnect();
    this.outputNode?.disconnect();
  }
}

// Reverb (Items 429-442)
export class Reverb implements Effect {
  id: string;
  name: string = 'Reverb';
  type: string = 'reverb';
  category: 'timebased' = 'timebased';
  bypassed: boolean = false;
  parameters: Record<string, EffectParameter>;
  inputNode: GainNode | null = null;
  outputNode: GainNode | null = null;

  private convolver: ConvolverNode | null = null;
  private wetNode: GainNode | null = null;
  private dryNode: GainNode | null = null;
  private preDelayNode: DelayNode | null = null;
  private context: AudioContext | null = null;

  constructor(id: string) {
    this.id = id;
    this.parameters = {
      size: { name: 'Room Size', value: 50, min: 0, max: 100, default: 50, unit: '%', curve: 'linear' },
      decay: { name: 'Decay Time', value: 2, min: 0.1, max: 10, default: 2, unit: 's', curve: 'logarithmic' },
      predelay: { name: 'Pre-Delay', value: 20, min: 0, max: 200, default: 20, unit: 'ms', curve: 'linear' },
      damping: { name: 'Damping', value: 50, min: 0, max: 100, default: 50, unit: '%', curve: 'linear' },
      mix: { name: 'Mix', value: 30, min: 0, max: 100, default: 30, unit: '%', curve: 'linear' },
      width: { name: 'Stereo Width', value: 100, min: 0, max: 100, default: 100, unit: '%', curve: 'linear' },
    };
  }

  process(input: AudioNode, output: AudioNode, context: AudioContext): void {
    this.context = context;
    this.inputNode = context.createGain();
    this.convolver = context.createConvolver();
    this.preDelayNode = context.createDelay(0.5);
    this.wetNode = context.createGain();
    this.dryNode = context.createGain();
    this.outputNode = context.createGain();

    // Generate impulse response
    this.generateImpulseResponse();

    this.preDelayNode.delayTime.value = this.parameters.predelay.value / 1000;
    this.wetNode.gain.value = this.parameters.mix.value / 100;
    this.dryNode.gain.value = 1 - this.parameters.mix.value / 100;

    input.connect(this.inputNode);

    // Dry path
    this.inputNode.connect(this.dryNode);
    this.dryNode.connect(this.outputNode);

    // Wet path
    this.inputNode.connect(this.preDelayNode);
    this.preDelayNode.connect(this.convolver);
    this.convolver.connect(this.wetNode);
    this.wetNode.connect(this.outputNode);

    this.outputNode.connect(output);
  }

  private generateImpulseResponse(): void {
    if (!this.context || !this.convolver) return;

    const sampleRate = this.context.sampleRate;
    const decayTime = this.parameters.decay.value;
    const size = this.parameters.size.value / 100;
    const length = sampleRate * decayTime;

    const impulse = this.context.createBuffer(2, length, sampleRate);
    const leftChannel = impulse.getChannelData(0);
    const rightChannel = impulse.getChannelData(1);

    for (let i = 0; i < length; i++) {
      const n = length - i;
      const decay = Math.pow(n / length, 1 + size * 2);
      leftChannel[i] = (Math.random() * 2 - 1) * decay;
      rightChannel[i] = (Math.random() * 2 - 1) * decay;
    }

    this.convolver.buffer = impulse;
  }

  setParameter(name: string, value: number): void {
    if (this.parameters[name]) {
      this.parameters[name].value = value;

      switch (name) {
        case 'predelay':
          if (this.preDelayNode) this.preDelayNode.delayTime.value = value / 1000;
          break;
        case 'mix':
          if (this.wetNode) this.wetNode.gain.value = value / 100;
          if (this.dryNode) this.dryNode.gain.value = 1 - value / 100;
          break;
        case 'size':
        case 'decay':
          this.generateImpulseResponse();
          break;
      }
    }
  }

  getParameter(name: string): number {
    return this.parameters[name]?.value ?? 0;
  }

  destroy(): void {
    this.inputNode?.disconnect();
    this.convolver?.disconnect();
    this.preDelayNode?.disconnect();
    this.wetNode?.disconnect();
    this.dryNode?.disconnect();
    this.outputNode?.disconnect();
  }
}

// Chorus (Items 443-457)
export class Chorus implements Effect {
  id: string;
  name: string = 'Chorus';
  type: string = 'chorus';
  category: 'modulation' = 'modulation';
  bypassed: boolean = false;
  parameters: Record<string, EffectParameter>;
  inputNode: GainNode | null = null;
  outputNode: GainNode | null = null;

  private delayNodes: DelayNode[] = [];
  private lfoNodes: OscillatorNode[] = [];
  private lfoGainNodes: GainNode[] = [];
  private wetNode: GainNode | null = null;
  private dryNode: GainNode | null = null;

  constructor(id: string) {
    this.id = id;
    this.parameters = {
      rate: { name: 'Rate', value: 1, min: 0.1, max: 10, default: 1, unit: 'Hz', curve: 'logarithmic' },
      depth: { name: 'Depth', value: 50, min: 0, max: 100, default: 50, unit: '%', curve: 'linear' },
      delay: { name: 'Delay', value: 7, min: 1, max: 30, default: 7, unit: 'ms', curve: 'linear' },
      mix: { name: 'Mix', value: 50, min: 0, max: 100, default: 50, unit: '%', curve: 'linear' },
      voices: { name: 'Voices', value: 2, min: 1, max: 4, default: 2, unit: '', curve: 'linear' },
      spread: { name: 'Stereo Spread', value: 100, min: 0, max: 100, default: 100, unit: '%', curve: 'linear' },
    };
  }

  process(input: AudioNode, output: AudioNode, context: AudioContext): void {
    this.inputNode = context.createGain();
    this.wetNode = context.createGain();
    this.dryNode = context.createGain();
    this.outputNode = context.createGain();

    const numVoices = Math.round(this.parameters.voices.value);

    for (let i = 0; i < numVoices; i++) {
      const delay = context.createDelay(0.1);
      const lfo = context.createOscillator();
      const lfoGain = context.createGain();

      delay.delayTime.value = this.parameters.delay.value / 1000;
      lfo.frequency.value = this.parameters.rate.value + (i * 0.1);
      lfo.type = 'sine';
      lfoGain.gain.value = (this.parameters.depth.value / 100) * 0.002;

      lfo.connect(lfoGain);
      lfoGain.connect(delay.delayTime);
      lfo.start();

      this.delayNodes.push(delay);
      this.lfoNodes.push(lfo);
      this.lfoGainNodes.push(lfoGain);
    }

    this.wetNode.gain.value = this.parameters.mix.value / 100;
    this.dryNode.gain.value = 1 - this.parameters.mix.value / 100;

    input.connect(this.inputNode);

    // Dry path
    this.inputNode.connect(this.dryNode);
    this.dryNode.connect(this.outputNode);

    // Wet path (chorus voices)
    this.delayNodes.forEach(delay => {
      this.inputNode!.connect(delay);
      delay.connect(this.wetNode!);
    });
    this.wetNode.connect(this.outputNode);

    this.outputNode.connect(output);
  }

  setParameter(name: string, value: number): void {
    if (this.parameters[name]) {
      this.parameters[name].value = value;

      switch (name) {
        case 'rate':
          this.lfoNodes.forEach((lfo, i) => {
            lfo.frequency.value = value + (i * 0.1);
          });
          break;
        case 'depth':
          this.lfoGainNodes.forEach(lfoGain => {
            lfoGain.gain.value = (value / 100) * 0.002;
          });
          break;
        case 'delay':
          this.delayNodes.forEach(delay => {
            delay.delayTime.value = value / 1000;
          });
          break;
        case 'mix':
          if (this.wetNode) this.wetNode.gain.value = value / 100;
          if (this.dryNode) this.dryNode.gain.value = 1 - value / 100;
          break;
      }
    }
  }

  getParameter(name: string): number {
    return this.parameters[name]?.value ?? 0;
  }

  destroy(): void {
    this.lfoNodes.forEach(lfo => lfo.stop());
    this.inputNode?.disconnect();
    this.delayNodes.forEach(d => d.disconnect());
    this.lfoGainNodes.forEach(g => g.disconnect());
    this.wetNode?.disconnect();
    this.dryNode?.disconnect();
    this.outputNode?.disconnect();
  }
}

// Distortion (Items 458-474)
export class Distortion implements Effect {
  id: string;
  name: string = 'Distortion';
  type: string = 'distortion';
  category: 'distortion' = 'distortion';
  bypassed: boolean = false;
  parameters: Record<string, EffectParameter>;
  inputNode: GainNode | null = null;
  outputNode: GainNode | null = null;

  private waveshaper: WaveShaperNode | null = null;
  private preGain: GainNode | null = null;
  private postGain: GainNode | null = null;
  private lowpassFilter: BiquadFilterNode | null = null;

  constructor(id: string) {
    this.id = id;
    this.parameters = {
      drive: { name: 'Drive', value: 50, min: 0, max: 100, default: 50, unit: '%', curve: 'linear' },
      tone: { name: 'Tone', value: 5000, min: 500, max: 15000, default: 5000, unit: 'Hz', curve: 'logarithmic' },
      output: { name: 'Output', value: 0, min: -24, max: 12, default: 0, unit: 'dB', curve: 'linear' },
      type: { name: 'Type', value: 0, min: 0, max: 3, default: 0, unit: '', curve: 'linear' }, // 0=soft, 1=hard, 2=tube, 3=fuzz
      mix: { name: 'Mix', value: 100, min: 0, max: 100, default: 100, unit: '%', curve: 'linear' },
    };
  }

  process(input: AudioNode, output: AudioNode, context: AudioContext): void {
    this.inputNode = context.createGain();
    this.waveshaper = context.createWaveShaper();
    this.preGain = context.createGain();
    this.postGain = context.createGain();
    this.lowpassFilter = context.createBiquadFilter();
    this.outputNode = context.createGain();

    this.updateDistortionCurve();
    this.preGain.gain.value = 1 + this.parameters.drive.value / 20;
    this.postGain.gain.value = Math.pow(10, this.parameters.output.value / 20);
    this.lowpassFilter.type = 'lowpass';
    this.lowpassFilter.frequency.value = this.parameters.tone.value;

    input.connect(this.inputNode);
    this.inputNode.connect(this.preGain);
    this.preGain.connect(this.waveshaper);
    this.waveshaper.connect(this.lowpassFilter);
    this.lowpassFilter.connect(this.postGain);
    this.postGain.connect(this.outputNode);
    this.outputNode.connect(output);
  }

  private updateDistortionCurve(): void {
    if (!this.waveshaper) return;

    const samples = 44100;
    const curve = new Float32Array(samples);
    const drive = this.parameters.drive.value / 100;
    const type = Math.round(this.parameters.type.value);

    for (let i = 0; i < samples; i++) {
      const x = (i * 2) / samples - 1;

      switch (type) {
        case 0: // Soft clip (tanh)
          curve[i] = Math.tanh(x * (1 + drive * 10));
          break;
        case 1: // Hard clip
          curve[i] = Math.max(-1, Math.min(1, x * (1 + drive * 10)));
          break;
        case 2: // Tube-like
          curve[i] = (3 + drive * 10) * x * 20 * (Math.PI / 180) / (Math.PI + drive * 10 * Math.abs(x));
          break;
        case 3: // Fuzz
          curve[i] = Math.sign(x) * Math.pow(Math.abs(x), 1 / (1 + drive * 2));
          break;
        default:
          curve[i] = x;
      }
    }

    this.waveshaper.curve = curve;
    this.waveshaper.oversample = '4x';
  }

  setParameter(name: string, value: number): void {
    if (this.parameters[name]) {
      this.parameters[name].value = value;

      switch (name) {
        case 'drive':
          if (this.preGain) this.preGain.gain.value = 1 + value / 20;
          this.updateDistortionCurve();
          break;
        case 'tone':
          if (this.lowpassFilter) this.lowpassFilter.frequency.value = value;
          break;
        case 'output':
          if (this.postGain) this.postGain.gain.value = Math.pow(10, value / 20);
          break;
        case 'type':
          this.updateDistortionCurve();
          break;
      }
    }
  }

  getParameter(name: string): number {
    return this.parameters[name]?.value ?? 0;
  }

  destroy(): void {
    this.inputNode?.disconnect();
    this.preGain?.disconnect();
    this.waveshaper?.disconnect();
    this.lowpassFilter?.disconnect();
    this.postGain?.disconnect();
    this.outputNode?.disconnect();
  }
}

// Gate (Items 401, 405-406)
export class Gate implements Effect {
  id: string;
  name: string = 'Gate';
  type: string = 'gate';
  category: 'dynamics' = 'dynamics';
  bypassed: boolean = false;
  parameters: Record<string, EffectParameter>;
  inputNode: GainNode | null = null;
  outputNode: GainNode | null = null;

  private analyser: AnalyserNode | null = null;
  private gateGain: GainNode | null = null;
  private animationFrame: number | null = null;

  constructor(id: string) {
    this.id = id;
    this.parameters = {
      threshold: { name: 'Threshold', value: -40, min: -80, max: 0, default: -40, unit: 'dB', curve: 'linear' },
      attack: { name: 'Attack', value: 0.5, min: 0.01, max: 50, default: 0.5, unit: 'ms', curve: 'logarithmic' },
      hold: { name: 'Hold', value: 10, min: 0, max: 500, default: 10, unit: 'ms', curve: 'logarithmic' },
      release: { name: 'Release', value: 50, min: 1, max: 1000, default: 50, unit: 'ms', curve: 'logarithmic' },
      range: { name: 'Range', value: -80, min: -80, max: 0, default: -80, unit: 'dB', curve: 'linear' },
    };
  }

  process(input: AudioNode, output: AudioNode, context: AudioContext): void {
    this.inputNode = context.createGain();
    this.analyser = context.createAnalyser();
    this.gateGain = context.createGain();
    this.outputNode = context.createGain();

    this.analyser.fftSize = 256;

    input.connect(this.inputNode);
    this.inputNode.connect(this.analyser);
    this.inputNode.connect(this.gateGain);
    this.gateGain.connect(this.outputNode);
    this.outputNode.connect(output);

    this.startGateProcessing();
  }

  private startGateProcessing(): void {
    const process = () => {
      if (!this.analyser || !this.gateGain) return;

      const dataArray = new Float32Array(this.analyser.fftSize);
      this.analyser.getFloatTimeDomainData(dataArray);

      let peak = 0;
      for (let i = 0; i < dataArray.length; i++) {
        peak = Math.max(peak, Math.abs(dataArray[i]));
      }

      const peakDb = 20 * Math.log10(peak || 0.0001);
      const threshold = this.parameters.threshold.value;
      const range = this.parameters.range.value;

      // Simple gate behavior
      const targetGain = peakDb > threshold ? 1 : Math.pow(10, range / 20);
      // currentGain reserved for future smooth transitions
      const smoothing = peakDb > threshold
        ? this.parameters.attack.value / 1000
        : this.parameters.release.value / 1000;

      this.gateGain.gain.setTargetAtTime(targetGain, this.gateGain.context.currentTime, smoothing);

      this.animationFrame = requestAnimationFrame(process);
    };

    process();
  }

  setParameter(name: string, value: number): void {
    if (this.parameters[name]) {
      this.parameters[name].value = value;
    }
  }

  getParameter(name: string): number {
    return this.parameters[name]?.value ?? 0;
  }

  destroy(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    this.inputNode?.disconnect();
    this.analyser?.disconnect();
    this.gateGain?.disconnect();
    this.outputNode?.disconnect();
  }
}

// Effects factory
export class EffectsFactory {
  static create(type: string, id?: string): Effect {
    const effectId = id || crypto.randomUUID();

    switch (type) {
      case 'parametricEQ':
        return new ParametricEQ(effectId);
      case 'compressor':
        return new Compressor(effectId);
      case 'delay':
        return new Delay(effectId);
      case 'reverb':
        return new Reverb(effectId);
      case 'chorus':
        return new Chorus(effectId);
      case 'distortion':
        return new Distortion(effectId);
      case 'gate':
        return new Gate(effectId);
      default:
        throw new Error(`Unknown effect type: ${type}`);
    }
  }

  static getAvailableEffects(): { type: string; name: string; category: string }[] {
    return [
      { type: 'parametricEQ', name: 'Parametric EQ', category: 'eq' },
      { type: 'compressor', name: 'Compressor', category: 'dynamics' },
      { type: 'gate', name: 'Gate', category: 'dynamics' },
      { type: 'delay', name: 'Delay', category: 'timebased' },
      { type: 'reverb', name: 'Reverb', category: 'timebased' },
      { type: 'chorus', name: 'Chorus', category: 'modulation' },
      { type: 'distortion', name: 'Distortion', category: 'distortion' },
    ];
  }
}

export default EffectsFactory;
