/**
 * Core Audio Engine - Professional DAW Audio Processing
 * Implements Parts 1, 2, 5, 6 of the DAW specification
 */

// Types for audio engine
export interface AudioEngineConfig {
  sampleRate: 22050 | 44100 | 48000 | 88200 | 96000 | 176400 | 192000 | 384000;
  bitDepth: 16 | 24 | 32;
  bufferSize: 64 | 128 | 256 | 512 | 1024 | 2048 | 4096;
  channels: number;
}

export interface TransportState {
  isPlaying: boolean;
  isRecording: boolean;
  isPaused: boolean;
  currentTime: number;
  startTime: number;
  endTime: number;
  loopStart: number;
  loopEnd: number;
  loopEnabled: boolean;
  tempo: number;
  timeSignatureNumerator: number;
  timeSignatureDenominator: number;
  preRoll: number;
  postRoll: number;
  countIn: number;
  metronomeEnabled: boolean;
}

export interface RecordingConfig {
  mode: 'mono' | 'stereo' | 'multitrack';
  punchIn: number | null;
  punchOut: number | null;
  autoPunch: boolean;
  loopRecord: boolean;
  takeNumber: number;
  retrospectiveBufferSize: number;
  inputMonitoring: boolean;
  destructive: boolean;
}

export interface Marker {
  id: string;
  time: number;
  name: string;
  color: string;
  type: 'basic' | 'verse' | 'chorus' | 'bridge' | 'intro' | 'outro' | 'cd' | 'cycle';
}

export interface TempoChange {
  time: number;
  tempo: number;
  curve: 'linear' | 'bezier' | 'step';
}

export type TimeFormat = 'bars' | 'time' | 'timecode' | 'samples' | 'feet';

export interface MeterReading {
  peak: number;
  rms: number;
  lufsShortTerm: number;
  lufsMomentary: number;
  lufsIntegrated: number;
  truePeak: number;
  phaseCorrelation: number;
}

class AudioEngine {
  private context: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private analyser: AnalyserNode | null = null;
  private compressor: DynamicsCompressorNode | null = null;

  private config: AudioEngineConfig = {
    sampleRate: 48000,
    bitDepth: 24,
    bufferSize: 512,
    channels: 2,
  };

  private transport: TransportState = {
    isPlaying: false,
    isRecording: false,
    isPaused: false,
    currentTime: 0,
    startTime: 0,
    endTime: 300,
    loopStart: 0,
    loopEnd: 8,
    loopEnabled: false,
    tempo: 120,
    timeSignatureNumerator: 4,
    timeSignatureDenominator: 4,
    preRoll: 1,
    postRoll: 1,
    countIn: 0,
    metronomeEnabled: true,
  };

  private recording: RecordingConfig = {
    mode: 'stereo',
    punchIn: null,
    punchOut: null,
    autoPunch: false,
    loopRecord: false,
    takeNumber: 1,
    retrospectiveBufferSize: 30, // seconds
    inputMonitoring: true,
    destructive: false,
  };

  private markers: Marker[] = [];
  private tempoMap: TempoChange[] = [{ time: 0, tempo: 120, curve: 'step' }];
  private timeFormat: TimeFormat = 'bars';
  private animationFrame: number | null = null;
  private startTimestamp: number = 0;
  private listeners: Map<string, Set<Function>> = new Map();

  // Initialize audio context
  async initialize(): Promise<void> {
    try {
      this.context = new AudioContext({
        sampleRate: this.config.sampleRate,
        latencyHint: 'interactive',
      });

      // Master chain: Source -> Compressor -> Gain -> Analyser -> Destination
      this.compressor = this.context.createDynamicsCompressor();
      this.masterGain = this.context.createGain();
      this.analyser = this.context.createAnalyser();

      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.8;

      this.compressor.connect(this.masterGain);
      this.masterGain.connect(this.analyser);
      this.analyser.connect(this.context.destination);

      this.emit('initialized', { sampleRate: this.context.sampleRate });
    } catch (error) {
      console.error('Failed to initialize audio engine:', error);
      throw error;
    }
  }

  // Event system
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: Function): void {
    this.listeners.get(event)?.delete(callback);
  }

  private emit(event: string, data?: unknown): void {
    this.listeners.get(event)?.forEach(cb => cb(data));
  }

  // Transport controls (Items 26-48, 58-73)
  play(): void {
    if (!this.context) return;
    if (this.context.state === 'suspended') {
      this.context.resume();
    }

    this.transport.isPlaying = true;
    this.transport.isPaused = false;
    this.startTimestamp = performance.now() - (this.transport.currentTime * 1000);
    this.startPlaybackLoop();
    this.emit('play', this.transport);
  }

  pause(): void {
    this.transport.isPlaying = false;
    this.transport.isPaused = true;
    this.stopPlaybackLoop();
    this.emit('pause', this.transport);
  }

  stop(): void {
    this.transport.isPlaying = false;
    this.transport.isPaused = false;
    this.transport.isRecording = false;
    this.transport.currentTime = this.transport.startTime;
    this.stopPlaybackLoop();
    this.emit('stop', this.transport);
  }

  record(): void {
    if (!this.context) return;
    this.transport.isRecording = true;
    this.play();
    this.emit('record', this.transport);
  }

  toggleLoop(): void {
    this.transport.loopEnabled = !this.transport.loopEnabled;
    this.emit('loopToggle', this.transport.loopEnabled);
  }

  setLoopPoints(start: number, end: number): void {
    this.transport.loopStart = start;
    this.transport.loopEnd = end;
    this.emit('loopPoints', { start, end });
  }

  goToStart(): void {
    this.transport.currentTime = this.transport.startTime;
    this.emit('seek', this.transport.currentTime);
  }

  goToEnd(): void {
    this.transport.currentTime = this.transport.endTime;
    this.emit('seek', this.transport.currentTime);
  }

  goToTime(time: number): void {
    this.transport.currentTime = Math.max(0, Math.min(time, this.transport.endTime));
    if (this.transport.isPlaying) {
      this.startTimestamp = performance.now() - (this.transport.currentTime * 1000);
    }
    this.emit('seek', this.transport.currentTime);
  }

  goToMarker(markerId: string): void {
    const marker = this.markers.find(m => m.id === markerId);
    if (marker) {
      this.goToTime(marker.time);
    }
  }

  goToBar(bar: number): void {
    const time = this.barToTime(bar);
    this.goToTime(time);
  }

  rewind(seconds: number = 5): void {
    this.goToTime(this.transport.currentTime - seconds);
  }

  fastForward(seconds: number = 5): void {
    this.goToTime(this.transport.currentTime + seconds);
  }

  // Variable speed playback (Items 35-38)
  private playbackRate: number = 1;

  setPlaybackRate(rate: number): void {
    this.playbackRate = Math.max(0.25, Math.min(4, rate));
    this.emit('playbackRateChange', this.playbackRate);
  }

  halfSpeed(): void {
    this.setPlaybackRate(0.5);
  }

  doubleSpeed(): void {
    this.setPlaybackRate(2);
  }

  private startPlaybackLoop(): void {
    const tick = () => {
      if (!this.transport.isPlaying) return;

      const elapsed = (performance.now() - this.startTimestamp) / 1000 * this.playbackRate;
      this.transport.currentTime = this.transport.startTime + elapsed;

      // Loop handling
      if (this.transport.loopEnabled && this.transport.currentTime >= this.transport.loopEnd) {
        this.transport.currentTime = this.transport.loopStart;
        this.startTimestamp = performance.now() - (this.transport.loopStart * 1000 / this.playbackRate);
      }

      // End handling
      if (this.transport.currentTime >= this.transport.endTime) {
        this.stop();
        return;
      }

      this.emit('timeUpdate', this.transport.currentTime);
      this.animationFrame = requestAnimationFrame(tick);
    };

    this.animationFrame = requestAnimationFrame(tick);
  }

  private stopPlaybackLoop(): void {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
  }

  // Tempo & Time Signature (Items 81-94)
  setTempo(tempo: number): void {
    this.transport.tempo = Math.max(20, Math.min(999, tempo));
    this.tempoMap[0].tempo = this.transport.tempo;
    this.emit('tempoChange', this.transport.tempo);
  }

  tapTempo(tapTimes: number[]): number {
    if (tapTimes.length < 2) return this.transport.tempo;

    const intervals = [];
    for (let i = 1; i < tapTimes.length; i++) {
      intervals.push(tapTimes[i] - tapTimes[i - 1]);
    }

    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const tempo = 60000 / avgInterval;
    this.setTempo(tempo);
    return tempo;
  }

  setTimeSignature(numerator: number, denominator: number): void {
    this.transport.timeSignatureNumerator = numerator;
    this.transport.timeSignatureDenominator = denominator;
    this.emit('timeSignatureChange', { numerator, denominator });
  }

  addTempoChange(time: number, tempo: number, curve: 'linear' | 'bezier' | 'step' = 'step'): void {
    this.tempoMap.push({ time, tempo, curve });
    this.tempoMap.sort((a, b) => a.time - b.time);
    this.emit('tempoMapChange', this.tempoMap);
  }

  getTempoAtTime(time: number): number {
    let tempo = this.tempoMap[0].tempo;
    for (const change of this.tempoMap) {
      if (change.time <= time) {
        tempo = change.tempo;
      } else {
        break;
      }
    }
    return tempo;
  }

  // Time format conversion (Items 74-80)
  setTimeFormat(format: TimeFormat): void {
    this.timeFormat = format;
    this.emit('timeFormatChange', format);
  }

  formatTime(seconds: number): string {
    switch (this.timeFormat) {
      case 'bars': {
        const { bar, beat, tick } = this.timeToBars(seconds);
        return `${bar}.${beat}.${tick.toString().padStart(3, '0')}`;
      }
      case 'time': {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
      }
      case 'timecode': {
        const fps = 30;
        const totalFrames = Math.floor(seconds * fps);
        const frames = totalFrames % fps;
        const totalSecs = Math.floor(totalFrames / fps);
        const secs = totalSecs % 60;
        const mins = Math.floor(totalSecs / 60) % 60;
        const hours = Math.floor(totalSecs / 3600);
        return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`;
      }
      case 'samples': {
        const samples = Math.floor(seconds * this.config.sampleRate);
        return samples.toLocaleString();
      }
      case 'feet': {
        const fps = 24;
        const framesPerFoot = 16;
        const totalFrames = Math.floor(seconds * fps);
        const feet = Math.floor(totalFrames / framesPerFoot);
        const frames = totalFrames % framesPerFoot;
        return `${feet}+${frames.toString().padStart(2, '0')}`;
      }
      default:
        return seconds.toFixed(3);
    }
  }

  timeToBars(seconds: number): { bar: number; beat: number; tick: number } {
    const beatsPerSecond = this.transport.tempo / 60;
    const totalBeats = seconds * beatsPerSecond;
    const beatsPerBar = this.transport.timeSignatureNumerator;

    const bar = Math.floor(totalBeats / beatsPerBar) + 1;
    const beat = Math.floor(totalBeats % beatsPerBar) + 1;
    const tick = Math.floor((totalBeats % 1) * 960); // 960 PPQN

    return { bar, beat, tick };
  }

  barToTime(bar: number, beat: number = 1, tick: number = 0): number {
    const beatsPerBar = this.transport.timeSignatureNumerator;
    const beatsPerSecond = this.transport.tempo / 60;
    const totalBeats = (bar - 1) * beatsPerBar + (beat - 1) + tick / 960;
    return totalBeats / beatsPerSecond;
  }

  // Markers (Items 95-107)
  addMarker(time: number, name: string, type: Marker['type'] = 'basic', color: string = '#6366f1'): Marker {
    const marker: Marker = {
      id: crypto.randomUUID(),
      time,
      name,
      color,
      type,
    };
    this.markers.push(marker);
    this.markers.sort((a, b) => a.time - b.time);
    this.emit('markerAdd', marker);
    return marker;
  }

  removeMarker(id: string): void {
    this.markers = this.markers.filter(m => m.id !== id);
    this.emit('markerRemove', id);
  }

  updateMarker(id: string, updates: Partial<Omit<Marker, 'id'>>): void {
    const marker = this.markers.find(m => m.id === id);
    if (marker) {
      Object.assign(marker, updates);
      this.emit('markerUpdate', marker);
    }
  }

  getMarkers(): Marker[] {
    return [...this.markers];
  }

  getNextMarker(): Marker | null {
    return this.markers.find(m => m.time > this.transport.currentTime) || null;
  }

  getPreviousMarker(): Marker | null {
    const previous = this.markers.filter(m => m.time < this.transport.currentTime);
    return previous[previous.length - 1] || null;
  }

  // Metering (Items 315-332)
  getMeterReading(): MeterReading {
    if (!this.analyser) {
      return {
        peak: -Infinity,
        rms: -Infinity,
        lufsShortTerm: -Infinity,
        lufsMomentary: -Infinity,
        lufsIntegrated: -Infinity,
        truePeak: -Infinity,
        phaseCorrelation: 1,
      };
    }

    const dataArray = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(dataArray);

    // Calculate peak
    let peak = 0;
    let sumSquares = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const abs = Math.abs(dataArray[i]);
      if (abs > peak) peak = abs;
      sumSquares += dataArray[i] * dataArray[i];
    }

    // Calculate RMS
    const rms = Math.sqrt(sumSquares / dataArray.length);

    // Convert to dB
    const peakDb = 20 * Math.log10(peak || 0.0001);
    const rmsDb = 20 * Math.log10(rms || 0.0001);

    // Simplified LUFS (actual LUFS requires K-weighting filter)
    const lufs = rmsDb - 0.691; // Approximate

    return {
      peak: peakDb,
      rms: rmsDb,
      lufsShortTerm: lufs,
      lufsMomentary: lufs,
      lufsIntegrated: lufs,
      truePeak: peakDb + 0.5, // Approximate true peak
      phaseCorrelation: 1, // Would need stereo analysis
    };
  }

  getFrequencyData(): Uint8Array {
    if (!this.analyser) return new Uint8Array(0);
    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    this.analyser.getByteFrequencyData(dataArray);
    return dataArray;
  }

  getWaveformData(): Float32Array {
    if (!this.analyser) return new Float32Array(0);
    const dataArray = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(dataArray);
    return dataArray;
  }

  // Audio quality settings (Items 49-57)
  setSampleRate(rate: AudioEngineConfig['sampleRate']): void {
    this.config.sampleRate = rate;
    // Would need to reinitialize context
    this.emit('sampleRateChange', rate);
  }

  setBitDepth(depth: AudioEngineConfig['bitDepth']): void {
    this.config.bitDepth = depth;
    this.emit('bitDepthChange', depth);
  }

  setBufferSize(size: AudioEngineConfig['bufferSize']): void {
    this.config.bufferSize = size;
    this.emit('bufferSizeChange', size);
  }

  // Recording (Items 1-25)
  setRecordingMode(mode: RecordingConfig['mode']): void {
    this.recording.mode = mode;
    this.emit('recordingModeChange', mode);
  }

  setPunchPoints(punchIn: number | null, punchOut: number | null): void {
    this.recording.punchIn = punchIn;
    this.recording.punchOut = punchOut;
    this.recording.autoPunch = punchIn !== null && punchOut !== null;
    this.emit('punchPointsChange', { punchIn, punchOut });
  }

  setLoopRecording(enabled: boolean): void {
    this.recording.loopRecord = enabled;
    this.emit('loopRecordingChange', enabled);
  }

  incrementTake(): void {
    this.recording.takeNumber++;
    this.emit('takeChange', this.recording.takeNumber);
  }

  // Master output controls
  setMasterVolume(volume: number): void {
    if (this.masterGain) {
      this.masterGain.gain.value = Math.max(0, Math.min(2, volume));
      this.emit('masterVolumeChange', volume);
    }
  }

  getMasterVolume(): number {
    return this.masterGain?.gain.value || 1;
  }

  // Getters
  getContext(): AudioContext | null {
    return this.context;
  }

  getAnalyser(): AnalyserNode | null {
    return this.analyser;
  }

  getTransport(): TransportState {
    return { ...this.transport };
  }

  getConfig(): AudioEngineConfig {
    return { ...this.config };
  }

  getRecordingConfig(): RecordingConfig {
    return { ...this.recording };
  }

  // Cleanup
  destroy(): void {
    this.stop();
    this.context?.close();
    this.context = null;
    this.listeners.clear();
  }
}

// Singleton instance
export const audioEngine = new AudioEngine();
export default AudioEngine;
