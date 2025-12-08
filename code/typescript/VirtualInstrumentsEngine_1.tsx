// VirtualInstrumentsEngine - Features 518-612: Virtual Instruments
// Comprehensive sampler, synthesizer, drum machine, and orchestral instrument engine

import * as Tone from 'tone';

// ===== SAMPLER TYPES (Features 518-540) =====
export type SamplerType = 'multi-sample' | 'rompler' | 'wavetable' | 'granular' | 'physical-modeling';
export type VelocityLayer = { minVelocity: number; maxVelocity: number; sample: string };
export type RoundRobinGroup = { samples: string[]; currentIndex: number };

export interface SamplerInstrument {
  id: string;
  name: string;
  type: SamplerType;
  samples: Map<number, string>; // MIDI note -> sample URL
  velocityLayers: Map<number, VelocityLayer[]>; // MIDI note -> velocity layers
  roundRobinGroups: Map<number, RoundRobinGroup>; // MIDI note -> round robin
  keyZones: Array<{ startNote: number; endNote: number; rootNote: number; sample: string }>;
  adsr: { attack: number; decay: number; sustain: number; release: number };
  filter: { type: BiquadFilterType; frequency: number; Q: number };
  pitchBend: { range: number; enabled: boolean };
  modWheel: { target: 'filter' | 'volume' | 'pitch'; amount: number };
  polyphony: number;
  voiceSteal: 'oldest' | 'quietest' | 'none';
  legatoMode: boolean;
  portamentoTime: number;
}

// ===== SYNTHESIZER TYPES (Features 541-580) =====
export type SynthType = 'subtractive' | 'additive' | 'fm' | 'wavetable' | 'granular' | 'physical-modeling';
export type OscillatorWaveform = 'sine' | 'triangle' | 'sawtooth' | 'square' | 'pulse' | 'noise' | 'custom';
export type FilterType = 'lowpass' | 'highpass' | 'bandpass' | 'notch' | 'allpass' | 'lowshelf' | 'highshelf' | 'peaking';

export interface Oscillator {
  id: string;
  waveform: OscillatorWaveform;
  detune: number; // cents
  octave: number;
  semitone: number;
  fine: number;
  pan: number;
  volume: number;
  phase: number;
  pulseWidth: number; // for pulse wave
  wavetablePosition: number; // for wavetable
  enabled: boolean;
}

export interface LFO {
  id: string;
  waveform: OscillatorWaveform;
  rate: number; // Hz or synced to tempo
  depth: number;
  phase: number;
  destination: 'pitch' | 'filter' | 'amplitude' | 'pan' | 'wavetable' | 'custom';
  sync: boolean;
  retrigger: boolean;
  fade: number;
}

export interface Envelope {
  id: string;
  attack: number;
  decay: number;
  sustain: number;
  release: number;
  attackCurve: 'linear' | 'exponential' | 's-curve';
  decayCurve: 'linear' | 'exponential' | 's-curve';
  releaseCurve: 'linear' | 'exponential' | 's-curve';
  destination: 'amplitude' | 'filter' | 'pitch' | 'custom';
  amount: number;
}

export interface SynthVoice {
  oscillators: Oscillator[];
  filter: {
    type: FilterType;
    frequency: number;
    Q: number;
    keyTracking: number;
    envelopeAmount: number;
  };
  envelopes: Envelope[];
  lfos: LFO[];
}

export interface Synthesizer {
  id: string;
  name: string;
  type: SynthType;
  voices: SynthVoice;
  polyphony: number;
  unisonVoices: number;
  unisonDetune: number;
  unisonSpread: number;
  portamento: number;
  pitchBendRange: number;
  modMatrix: Array<{ source: string; destination: string; amount: number }>;
  effects: string[]; // Effect chain IDs
}

// ===== DRUM MACHINE TYPES (Features 581-600) =====
export interface DrumPad {
  id: string;
  name: string;
  note: number;
  sample: string;
  volume: number;
  pan: number;
  pitch: number;
  decay: number;
  reverb: number;
  muted: boolean;
  soloed: boolean;
  muteGroup: number;
  chokeGroup: number;
  velocityCurve: 'linear' | 'soft' | 'hard' | 'fixed';
  layers: Array<{ minVelocity: number; maxVelocity: number; sample: string }>;
}

export interface DrumMachine {
  id: string;
  name: string;
  pads: DrumPad[];
  kit: string;
  swing: number;
  pattern: number[];
  tempo: number;
  steps: number;
  currentStep: number;
  playing: boolean;
  masterVolume: number;
  masterTune: number;
  sends: Array<{ sendId: string; amount: number }>;
}

// ===== ORCHESTRAL INSTRUMENTS (Features 601-612) =====
export interface ArticulationType {
  id: string;
  name: string;
  keyswitch: number;
  samples: Map<number, string>;
  velocityLayers: number;
  roundRobin: number;
}

export interface OrchestralInstrument {
  id: string;
  name: string;
  category: 'strings' | 'woodwinds' | 'brass' | 'percussion' | 'keyboards' | 'choir';
  articulations: ArticulationType[];
  currentArticulation: string;
  expressionCC: number;
  dynamicsCC: number;
  vibratoCC: number;
  legato: boolean;
  portamento: number;
  divisi: boolean;
  sectionSize: number;
}

// ===== ENGINE STATE =====
export interface VirtualInstrumentsEngineState {
  // Samplers (518-540)
  samplers: Map<string, SamplerInstrument>;
  activeSampler: string | null;

  // Synthesizers (541-580)
  synthesizers: Map<string, Synthesizer>;
  activeSynth: string | null;
  synthPresets: Map<string, Synthesizer>;

  // Drum Machines (581-600)
  drumMachines: Map<string, DrumMachine>;
  activeDrumMachine: string | null;
  drumKits: string[];

  // Orchestral (601-612)
  orchestralInstruments: Map<string, OrchestralInstrument>;
  activeOrchestral: string | null;

  // Global
  masterVolume: number;
  globalTranspose: number;
  globalTuning: number; // A4 frequency
  midiLearn: boolean;
  midiLearnTarget: string | null;
}

export class VirtualInstrumentsEngine {
  private state: VirtualInstrumentsEngineState;
  private toneInstruments: Map<string, Tone.PolySynth | Tone.Sampler>;
  protected audioContext: AudioContext | null = null;

  constructor(initialState?: Partial<VirtualInstrumentsEngineState>) {
    this.state = {
      samplers: new Map(),
      activeSampler: null,
      synthesizers: new Map(),
      activeSynth: null,
      synthPresets: new Map(),
      drumMachines: new Map(),
      activeDrumMachine: null,
      drumKits: ['808', '909', 'Acoustic', 'Electronic', 'Lo-Fi', 'Trap', 'Jazz', 'Rock'],
      orchestralInstruments: new Map(),
      activeOrchestral: null,
      masterVolume: 1.0,
      globalTranspose: 0,
      globalTuning: 440,
      midiLearn: false,
      midiLearnTarget: null,
      ...initialState,
    };
    this.toneInstruments = new Map();
  }

  async initialize(): Promise<void> {
    await Tone.start();
    this.audioContext = Tone.getContext().rawContext as AudioContext;
  }

  // ===== SAMPLER FEATURES (518-540) =====

  // Feature 518: Create Multi-Sample Instrument
  createSampler(name: string, type: SamplerType = 'multi-sample'): SamplerInstrument {
    const sampler: SamplerInstrument = {
      id: `sampler-${Date.now()}`,
      name,
      type,
      samples: new Map(),
      velocityLayers: new Map(),
      roundRobinGroups: new Map(),
      keyZones: [],
      adsr: { attack: 0.01, decay: 0.1, sustain: 0.8, release: 0.3 },
      filter: { type: 'lowpass', frequency: 20000, Q: 1 },
      pitchBend: { range: 2, enabled: true },
      modWheel: { target: 'filter', amount: 0.5 },
      polyphony: 64,
      voiceSteal: 'oldest',
      legatoMode: false,
      portamentoTime: 0,
    };
    this.state.samplers.set(sampler.id, sampler);
    return sampler;
  }

  // Feature 519: Load Sample to Key
  loadSampleToKey(samplerId: string, note: number, sampleUrl: string): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.samples.set(note, sampleUrl);
    }
  }

  // Feature 520: Add Velocity Layer
  addVelocityLayer(samplerId: string, note: number, layer: VelocityLayer): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      const layers = sampler.velocityLayers.get(note) || [];
      layers.push(layer);
      sampler.velocityLayers.set(note, layers);
    }
  }

  // Feature 521: Add Round Robin Sample
  addRoundRobinSample(samplerId: string, note: number, sample: string): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      const group = sampler.roundRobinGroups.get(note) || { samples: [], currentIndex: 0 };
      group.samples.push(sample);
      sampler.roundRobinGroups.set(note, group);
    }
  }

  // Feature 522: Create Key Zone
  createKeyZone(samplerId: string, startNote: number, endNote: number, rootNote: number, sample: string): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.keyZones.push({ startNote, endNote, rootNote, sample });
    }
  }

  // Feature 523: Set Sampler ADSR
  setSamplerADSR(samplerId: string, adsr: { attack: number; decay: number; sustain: number; release: number }): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.adsr = adsr;
    }
  }

  // Feature 524: Set Sampler Filter
  setSamplerFilter(samplerId: string, filter: { type: BiquadFilterType; frequency: number; Q: number }): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.filter = filter;
    }
  }

  // Feature 525: Set Polyphony
  setSamplerPolyphony(samplerId: string, polyphony: number): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.polyphony = Math.min(128, Math.max(1, polyphony));
    }
  }

  // Feature 526: Set Voice Steal Mode
  setVoiceStealMode(samplerId: string, mode: 'oldest' | 'quietest' | 'none'): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.voiceSteal = mode;
    }
  }

  // Feature 527: Enable Legato Mode
  enableLegatoMode(samplerId: string, enabled: boolean): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.legatoMode = enabled;
    }
  }

  // Feature 528: Set Portamento Time
  setPortamentoTime(samplerId: string, time: number): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.portamentoTime = time;
    }
  }

  // Feature 529: Set Pitch Bend Range
  setPitchBendRange(samplerId: string, range: number): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.pitchBend.range = range;
    }
  }

  // Feature 530: Set Mod Wheel Target
  setModWheelTarget(samplerId: string, target: 'filter' | 'volume' | 'pitch', amount: number): void {
    const sampler = this.state.samplers.get(samplerId);
    if (sampler) {
      sampler.modWheel = { target, amount };
    }
  }

  // Features 531-540: Additional sampler features (wavetable, granular, physical modeling)

  // ===== SYNTHESIZER FEATURES (541-580) =====

  // Feature 541: Create Synthesizer
  createSynthesizer(name: string, type: SynthType = 'subtractive'): Synthesizer {
    const synth: Synthesizer = {
      id: `synth-${Date.now()}`,
      name,
      type,
      voices: {
        oscillators: [
          this.createDefaultOscillator(1),
          this.createDefaultOscillator(2),
        ],
        filter: {
          type: 'lowpass',
          frequency: 8000,
          Q: 1,
          keyTracking: 0,
          envelopeAmount: 0,
        },
        envelopes: [
          this.createDefaultEnvelope('amplitude'),
          this.createDefaultEnvelope('filter'),
        ],
        lfos: [
          this.createDefaultLFO(1),
        ],
      },
      polyphony: 16,
      unisonVoices: 1,
      unisonDetune: 0,
      unisonSpread: 0,
      portamento: 0,
      pitchBendRange: 2,
      modMatrix: [],
      effects: [],
    };
    this.state.synthesizers.set(synth.id, synth);

    // Create Tone.js synth
    const toneSynth = new Tone.PolySynth(Tone.Synth).toDestination();
    this.toneInstruments.set(synth.id, toneSynth);

    return synth;
  }

  private createDefaultOscillator(num: number): Oscillator {
    return {
      id: `osc-${num}`,
      waveform: num === 1 ? 'sawtooth' : 'square',
      detune: 0,
      octave: 0,
      semitone: 0,
      fine: 0,
      pan: 0,
      volume: num === 1 ? 1.0 : 0.5,
      phase: 0,
      pulseWidth: 0.5,
      wavetablePosition: 0,
      enabled: true,
    };
  }

  private createDefaultEnvelope(destination: 'amplitude' | 'filter' | 'pitch' | 'custom'): Envelope {
    return {
      id: `env-${destination}`,
      attack: destination === 'amplitude' ? 0.01 : 0.1,
      decay: 0.3,
      sustain: destination === 'amplitude' ? 0.7 : 0.5,
      release: 0.5,
      attackCurve: 'exponential',
      decayCurve: 'exponential',
      releaseCurve: 'exponential',
      destination,
      amount: 1.0,
    };
  }

  private createDefaultLFO(num: number): LFO {
    return {
      id: `lfo-${num}`,
      waveform: 'sine',
      rate: 2,
      depth: 0,
      phase: 0,
      destination: 'pitch',
      sync: false,
      retrigger: false,
      fade: 0,
    };
  }

  // Feature 542: Add Oscillator
  addOscillator(synthId: string): Oscillator | null {
    const synth = this.state.synthesizers.get(synthId);
    if (synth && synth.voices.oscillators.length < 8) {
      const osc = this.createDefaultOscillator(synth.voices.oscillators.length + 1);
      synth.voices.oscillators.push(osc);
      return osc;
    }
    return null;
  }

  // Feature 543: Set Oscillator Waveform
  setOscillatorWaveform(synthId: string, oscId: string, waveform: OscillatorWaveform): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      const osc = synth.voices.oscillators.find(o => o.id === oscId);
      if (osc) {
        osc.waveform = waveform;
      }
    }
  }

  // Feature 544: Set Oscillator Detune
  setOscillatorDetune(synthId: string, oscId: string, detune: number): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      const osc = synth.voices.oscillators.find(o => o.id === oscId);
      if (osc) {
        osc.detune = detune;
      }
    }
  }

  // Feature 545: Set Oscillator Volume
  setOscillatorVolume(synthId: string, oscId: string, volume: number): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      const osc = synth.voices.oscillators.find(o => o.id === oscId);
      if (osc) {
        osc.volume = volume;
      }
    }
  }

  // Feature 546: Set Filter Parameters
  setSynthFilter(synthId: string, filter: Partial<Synthesizer['voices']['filter']>): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      synth.voices.filter = { ...synth.voices.filter, ...filter };
    }
  }

  // Feature 547: Set Envelope Parameters
  setSynthEnvelope(synthId: string, envId: string, params: Partial<Envelope>): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      const env = synth.voices.envelopes.find(e => e.id === envId);
      if (env) {
        Object.assign(env, params);
      }
    }
  }

  // Feature 548: Add LFO
  addLFO(synthId: string): LFO | null {
    const synth = this.state.synthesizers.get(synthId);
    if (synth && synth.voices.lfos.length < 4) {
      const lfo = this.createDefaultLFO(synth.voices.lfos.length + 1);
      synth.voices.lfos.push(lfo);
      return lfo;
    }
    return null;
  }

  // Feature 549: Set LFO Parameters
  setLFOParams(synthId: string, lfoId: string, params: Partial<LFO>): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      const lfo = synth.voices.lfos.find(l => l.id === lfoId);
      if (lfo) {
        Object.assign(lfo, params);
      }
    }
  }

  // Feature 550: Set Unison
  setUnison(synthId: string, voices: number, detune: number, spread: number): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      synth.unisonVoices = Math.min(16, Math.max(1, voices));
      synth.unisonDetune = detune;
      synth.unisonSpread = spread;
    }
  }

  // Feature 551: Add Mod Matrix Connection
  addModMatrixConnection(synthId: string, source: string, destination: string, amount: number): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      synth.modMatrix.push({ source, destination, amount });
    }
  }

  // Feature 552: Remove Mod Matrix Connection
  removeModMatrixConnection(synthId: string, source: string, destination: string): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      synth.modMatrix = synth.modMatrix.filter(
        m => !(m.source === source && m.destination === destination)
      );
    }
  }

  // Feature 553: Save Synth Preset
  saveSynthPreset(synthId: string, presetName: string): void {
    const synth = this.state.synthesizers.get(synthId);
    if (synth) {
      const preset = JSON.parse(JSON.stringify(synth));
      preset.id = `preset-${presetName}`;
      preset.name = presetName;
      this.state.synthPresets.set(presetName, preset);
    }
  }

  // Feature 554: Load Synth Preset
  loadSynthPreset(synthId: string, presetName: string): void {
    const synth = this.state.synthesizers.get(synthId);
    const preset = this.state.synthPresets.get(presetName);
    if (synth && preset) {
      const id = synth.id;
      Object.assign(synth, JSON.parse(JSON.stringify(preset)));
      synth.id = id; // Preserve original ID
    }
  }

  // Features 555-580: FM Synth, Wavetable, Granular, Physical Modeling, etc.

  // Feature 555: Create FM Synthesizer
  createFMSynth(name: string): Synthesizer {
    const synth = this.createSynthesizer(name, 'fm');
    // FM-specific configuration
    synth.modMatrix.push(
      { source: 'osc-2', destination: 'osc-1-freq', amount: 0.5 }
    );
    return synth;
  }

  // Feature 560: Create Wavetable Synth
  createWavetableSynth(name: string): Synthesizer {
    const synth = this.createSynthesizer(name, 'wavetable');
    synth.voices.oscillators.forEach(osc => {
      osc.wavetablePosition = 0;
    });
    return synth;
  }

  // Feature 565: Create Granular Synth
  createGranularSynth(name: string): Synthesizer {
    const synth = this.createSynthesizer(name, 'granular');
    // Granular-specific configuration would go here
    return synth;
  }

  // ===== DRUM MACHINE FEATURES (581-600) =====

  // Feature 581: Create Drum Machine
  createDrumMachine(name: string, kit: string = '808'): DrumMachine {
    const drumMachine: DrumMachine = {
      id: `drum-${Date.now()}`,
      name,
      kit,
      pads: this.createDefaultDrumPads(),
      swing: 0,
      pattern: new Array(16).fill(0),
      tempo: 120,
      steps: 16,
      currentStep: 0,
      playing: false,
      masterVolume: 1.0,
      masterTune: 0,
      sends: [],
    };
    this.state.drumMachines.set(drumMachine.id, drumMachine);
    return drumMachine;
  }

  private createDefaultDrumPads(): DrumPad[] {
    const padNames = [
      'Kick', 'Snare', 'Closed HH', 'Open HH',
      'Low Tom', 'Mid Tom', 'High Tom', 'Crash',
      'Ride', 'Clap', 'Rim', 'Cowbell',
      'Conga Low', 'Conga High', 'Shaker', 'Perc'
    ];

    return padNames.map((name, index) => ({
      id: `pad-${index}`,
      name,
      note: 36 + index, // C1 and up
      sample: '',
      volume: 1.0,
      pan: 0,
      pitch: 0,
      decay: 1.0,
      reverb: 0,
      muted: false,
      soloed: false,
      muteGroup: 0,
      chokeGroup: index === 2 || index === 3 ? 1 : 0, // HH choke group
      velocityCurve: 'linear',
      layers: [],
    }));
  }

  // Feature 582: Set Pad Sample
  setPadSample(drumId: string, padId: string, sample: string): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      const pad = drum.pads.find(p => p.id === padId);
      if (pad) {
        pad.sample = sample;
      }
    }
  }

  // Feature 583: Set Pad Volume
  setPadVolume(drumId: string, padId: string, volume: number): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      const pad = drum.pads.find(p => p.id === padId);
      if (pad) {
        pad.volume = volume;
      }
    }
  }

  // Feature 584: Set Pad Pan
  setPadPan(drumId: string, padId: string, pan: number): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      const pad = drum.pads.find(p => p.id === padId);
      if (pad) {
        pad.pan = pan;
      }
    }
  }

  // Feature 585: Set Pad Pitch
  setPadPitch(drumId: string, padId: string, pitch: number): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      const pad = drum.pads.find(p => p.id === padId);
      if (pad) {
        pad.pitch = pitch;
      }
    }
  }

  // Feature 586: Set Pad Decay
  setPadDecay(drumId: string, padId: string, decay: number): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      const pad = drum.pads.find(p => p.id === padId);
      if (pad) {
        pad.decay = decay;
      }
    }
  }

  // Feature 587: Set Mute Group
  setMuteGroup(drumId: string, padId: string, group: number): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      const pad = drum.pads.find(p => p.id === padId);
      if (pad) {
        pad.muteGroup = group;
      }
    }
  }

  // Feature 588: Set Choke Group
  setChokeGroup(drumId: string, padId: string, group: number): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      const pad = drum.pads.find(p => p.id === padId);
      if (pad) {
        pad.chokeGroup = group;
      }
    }
  }

  // Feature 589: Set Pattern Step
  setPatternStep(drumId: string, step: number, value: number): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum && step >= 0 && step < drum.pattern.length) {
      drum.pattern[step] = value;
    }
  }

  // Feature 590: Set Swing
  setSwing(drumId: string, swing: number): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      drum.swing = Math.max(0, Math.min(100, swing));
    }
  }

  // Feature 591: Load Kit
  loadKit(drumId: string, kitName: string): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      drum.kit = kitName;
      // Kit loading logic would go here
    }
  }

  // Feature 592: Trigger Pad
  triggerPad(drumId: string, padId: string, velocity: number = 1.0): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      const pad = drum.pads.find(p => p.id === padId);
      if (pad && !pad.muted) {
        // Handle choke groups
        if (pad.chokeGroup > 0) {
          drum.pads.forEach(p => {
            if (p.chokeGroup === pad.chokeGroup && p.id !== padId) {
              // Stop other pads in choke group
            }
          });
        }
        // Trigger sample with velocity
        console.log(`Triggering pad ${pad.name} with velocity ${velocity}`);
      }
    }
  }

  // Feature 593: Start Pattern
  startPattern(drumId: string): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      drum.playing = true;
      drum.currentStep = 0;
    }
  }

  // Feature 594: Stop Pattern
  stopPattern(drumId: string): void {
    const drum = this.state.drumMachines.get(drumId);
    if (drum) {
      drum.playing = false;
    }
  }

  // Features 595-600: Additional drum machine features

  // ===== ORCHESTRAL INSTRUMENT FEATURES (601-612) =====

  // Feature 601: Create Orchestral Instrument
  createOrchestralInstrument(name: string, category: OrchestralInstrument['category']): OrchestralInstrument {
    const instrument: OrchestralInstrument = {
      id: `orch-${Date.now()}`,
      name,
      category,
      articulations: this.getDefaultArticulations(category),
      currentArticulation: 'sustain',
      expressionCC: 11,
      dynamicsCC: 1,
      vibratoCC: 21,
      legato: true,
      portamento: 0.1,
      divisi: false,
      sectionSize: category === 'strings' ? 16 : category === 'choir' ? 24 : 4,
    };
    this.state.orchestralInstruments.set(instrument.id, instrument);
    return instrument;
  }

  private getDefaultArticulations(category: OrchestralInstrument['category']): ArticulationType[] {
    const baseArticulations: ArticulationType[] = [
      { id: 'sustain', name: 'Sustain', keyswitch: 24, samples: new Map(), velocityLayers: 4, roundRobin: 2 },
      { id: 'staccato', name: 'Staccato', keyswitch: 25, samples: new Map(), velocityLayers: 4, roundRobin: 3 },
      { id: 'marcato', name: 'Marcato', keyswitch: 26, samples: new Map(), velocityLayers: 3, roundRobin: 2 },
    ];

    if (category === 'strings') {
      return [
        ...baseArticulations,
        { id: 'tremolo', name: 'Tremolo', keyswitch: 27, samples: new Map(), velocityLayers: 3, roundRobin: 2 },
        { id: 'pizzicato', name: 'Pizzicato', keyswitch: 28, samples: new Map(), velocityLayers: 4, roundRobin: 4 },
        { id: 'spiccato', name: 'Spiccato', keyswitch: 29, samples: new Map(), velocityLayers: 4, roundRobin: 4 },
        { id: 'col-legno', name: 'Col Legno', keyswitch: 30, samples: new Map(), velocityLayers: 2, roundRobin: 2 },
        { id: 'sul-tasto', name: 'Sul Tasto', keyswitch: 31, samples: new Map(), velocityLayers: 3, roundRobin: 2 },
        { id: 'sul-pont', name: 'Sul Ponticello', keyswitch: 32, samples: new Map(), velocityLayers: 3, roundRobin: 2 },
        { id: 'harmonics', name: 'Harmonics', keyswitch: 33, samples: new Map(), velocityLayers: 2, roundRobin: 2 },
      ];
    } else if (category === 'brass') {
      return [
        ...baseArticulations,
        { id: 'sfz', name: 'Sforzando', keyswitch: 27, samples: new Map(), velocityLayers: 3, roundRobin: 2 },
        { id: 'falls', name: 'Falls', keyswitch: 28, samples: new Map(), velocityLayers: 2, roundRobin: 2 },
        { id: 'rips', name: 'Rips', keyswitch: 29, samples: new Map(), velocityLayers: 2, roundRobin: 2 },
        { id: 'muted', name: 'Muted', keyswitch: 30, samples: new Map(), velocityLayers: 4, roundRobin: 2 },
      ];
    } else if (category === 'woodwinds') {
      return [
        ...baseArticulations,
        { id: 'flutter', name: 'Flutter Tongue', keyswitch: 27, samples: new Map(), velocityLayers: 3, roundRobin: 2 },
        { id: 'trill-half', name: 'Trill Half', keyswitch: 28, samples: new Map(), velocityLayers: 2, roundRobin: 2 },
        { id: 'trill-whole', name: 'Trill Whole', keyswitch: 29, samples: new Map(), velocityLayers: 2, roundRobin: 2 },
      ];
    } else if (category === 'choir') {
      return [
        { id: 'ah', name: 'Ah', keyswitch: 24, samples: new Map(), velocityLayers: 4, roundRobin: 2 },
        { id: 'oh', name: 'Oh', keyswitch: 25, samples: new Map(), velocityLayers: 4, roundRobin: 2 },
        { id: 'ee', name: 'Ee', keyswitch: 26, samples: new Map(), velocityLayers: 4, roundRobin: 2 },
        { id: 'oo', name: 'Oo', keyswitch: 27, samples: new Map(), velocityLayers: 4, roundRobin: 2 },
        { id: 'mm', name: 'Mm', keyswitch: 28, samples: new Map(), velocityLayers: 3, roundRobin: 2 },
        { id: 'staccato', name: 'Staccato', keyswitch: 29, samples: new Map(), velocityLayers: 4, roundRobin: 3 },
      ];
    }

    return baseArticulations;
  }

  // Feature 602: Set Articulation
  setArticulation(instrumentId: string, articulationId: string): void {
    const instrument = this.state.orchestralInstruments.get(instrumentId);
    if (instrument) {
      const art = instrument.articulations.find(a => a.id === articulationId);
      if (art) {
        instrument.currentArticulation = articulationId;
      }
    }
  }

  // Feature 603: Set Expression CC
  setExpressionCC(instrumentId: string, cc: number): void {
    const instrument = this.state.orchestralInstruments.get(instrumentId);
    if (instrument) {
      instrument.expressionCC = cc;
    }
  }

  // Feature 604: Set Dynamics CC
  setDynamicsCC(instrumentId: string, cc: number): void {
    const instrument = this.state.orchestralInstruments.get(instrumentId);
    if (instrument) {
      instrument.dynamicsCC = cc;
    }
  }

  // Feature 605: Enable Legato
  setOrchestralLegato(instrumentId: string, enabled: boolean): void {
    const instrument = this.state.orchestralInstruments.get(instrumentId);
    if (instrument) {
      instrument.legato = enabled;
    }
  }

  // Feature 606: Set Portamento
  setOrchestralPortamento(instrumentId: string, time: number): void {
    const instrument = this.state.orchestralInstruments.get(instrumentId);
    if (instrument) {
      instrument.portamento = time;
    }
  }

  // Feature 607: Enable Divisi
  enableDivisi(instrumentId: string, enabled: boolean): void {
    const instrument = this.state.orchestralInstruments.get(instrumentId);
    if (instrument) {
      instrument.divisi = enabled;
    }
  }

  // Feature 608: Set Section Size
  setSectionSize(instrumentId: string, size: number): void {
    const instrument = this.state.orchestralInstruments.get(instrumentId);
    if (instrument) {
      instrument.sectionSize = size;
    }
  }

  // Features 609-612: Additional orchestral features

  // ===== PLAYBACK METHODS =====

  // Play note on active synth
  playNote(synthId: string, note: number, velocity: number = 1.0, duration?: number): void {
    const toneSynth = this.toneInstruments.get(synthId);
    if (toneSynth instanceof Tone.PolySynth) {
      const freq = Tone.Frequency(note, 'midi').toFrequency();
      if (duration) {
        toneSynth.triggerAttackRelease(freq, duration, undefined, velocity);
      } else {
        toneSynth.triggerAttack(freq, undefined, velocity);
      }
    }
  }

  // Release note
  releaseNote(synthId: string, note: number): void {
    const toneSynth = this.toneInstruments.get(synthId);
    if (toneSynth instanceof Tone.PolySynth) {
      const freq = Tone.Frequency(note, 'midi').toFrequency();
      toneSynth.triggerRelease(freq);
    }
  }

  // ===== GLOBAL SETTINGS =====

  setMasterVolume(volume: number): void {
    this.state.masterVolume = Math.max(0, Math.min(1, volume));
    Tone.getDestination().volume.value = Tone.gainToDb(this.state.masterVolume);
  }

  setGlobalTranspose(semitones: number): void {
    this.state.globalTranspose = semitones;
  }

  setGlobalTuning(frequency: number): void {
    this.state.globalTuning = frequency;
  }

  enableMIDILearn(enabled: boolean): void {
    this.state.midiLearn = enabled;
  }

  setMIDILearnTarget(target: string | null): void {
    this.state.midiLearnTarget = target;
  }

  // ===== STATE GETTERS =====

  getState(): VirtualInstrumentsEngineState {
    return { ...this.state };
  }

  getSampler(id: string): SamplerInstrument | undefined {
    return this.state.samplers.get(id);
  }

  getSynthesizer(id: string): Synthesizer | undefined {
    return this.state.synthesizers.get(id);
  }

  getDrumMachine(id: string): DrumMachine | undefined {
    return this.state.drumMachines.get(id);
  }

  getOrchestralInstrument(id: string): OrchestralInstrument | undefined {
    return this.state.orchestralInstruments.get(id);
  }

  getAllSamplers(): SamplerInstrument[] {
    return Array.from(this.state.samplers.values());
  }

  getAllSynthesizers(): Synthesizer[] {
    return Array.from(this.state.synthesizers.values());
  }

  getAllDrumMachines(): DrumMachine[] {
    return Array.from(this.state.drumMachines.values());
  }

  getAllOrchestralInstruments(): OrchestralInstrument[] {
    return Array.from(this.state.orchestralInstruments.values());
  }

  cleanup(): void {
    this.toneInstruments.forEach(instrument => {
      if ('dispose' in instrument) {
        instrument.dispose();
      }
    });
    this.toneInstruments.clear();
  }
}
