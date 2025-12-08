// TransportEngine - Advanced transport and time management (Features 58-100)

export type TimeFormat = 'bars-beats' | 'time' | 'samples' | 'feet-frames' | 'seconds' | 'minutes-seconds' | 'custom';
export type SyncSource = 'internal' | 'midi-clock' | 'word-clock' | 'ltc' | 'video' | 'none';

export interface TimePosition {
  bars: number;
  beats: number;
  ticks: number;
  samples: number;
  seconds: number;
  minutes: number;
  totalSeconds: number;
}

export interface TempoEvent {
  time: number; // in bars.beats.ticks
  tempo: number; // BPM
  timeSignature: [number, number]; // [numerator, denominator]
}

export interface Marker {
  id: string;
  name: string;
  position: TimePosition;
  color: string;
  locked: boolean;
}

export interface Locator {
  id: string;
  name: string;
  position: TimePosition;
  type: 'start' | 'end' | 'loop-start' | 'loop-end' | 'punch-in' | 'punch-out';
}

export interface TransportEngineState {
  // Transport (58-73)
  isPlaying: boolean;
  isRecording: boolean;
  isPaused: boolean;
  position: TimePosition;
  loopStart: TimePosition | null;
  loopEnd: TimePosition | null;
  cycleEnabled: boolean;
  shuttleSpeed: number; // -1.0 to 1.0
  scrubEnabled: boolean;
  scrubPosition: TimePosition | null;
  variableSpeed: number; // 0.25 to 4.0
  halfSpeed: boolean;
  doubleSpeed: boolean;
  reverse: boolean;
  frameByFrame: boolean;
  frameAdvance: number;
  syncSource: SyncSource;
  syncEnabled: boolean;
  
  // Time Formats (74-80)
  primaryTimeFormat: TimeFormat;
  secondaryTimeFormat: TimeFormat | null;
  displayFormat: TimeFormat;
  sampleRate: number;
  ticksPerQuarter: number;
  
  // Tempo & Time Signature (81-94)
  tempo: number; // BPM
  timeSignature: [number, number];
  tempoTrack: TempoEvent[];
  tempoMapEnabled: boolean;
  tempoRampEnabled: boolean;
  timeSignatureTrack: TempoEvent[];
  globalTempo: number;
  masterTempo: boolean;
  
  // Markers & Locators (95-107)
  markers: Marker[];
  locators: Locator[];
  selectedMarker: string | null;
  selectedLocator: string | null;
}

export class TransportEngine {
  private state: TransportEngineState;
  private transport: any; // Tone.Transport
  private updateInterval: ReturnType<typeof setInterval> | null = null;
  private tempoMap: Map<number, TempoEvent> = new Map();
  
  constructor(initialState?: Partial<TransportEngineState>) {
    this.state = {
      isPlaying: false,
      isRecording: false,
      isPaused: false,
      position: this.createTimePosition(0),
      loopStart: null,
      loopEnd: null,
      cycleEnabled: false,
      shuttleSpeed: 0,
      scrubEnabled: false,
      scrubPosition: null,
      variableSpeed: 1.0,
      halfSpeed: false,
      doubleSpeed: false,
      reverse: false,
      frameByFrame: false,
      frameAdvance: 0,
      syncSource: 'internal',
      syncEnabled: false,
      primaryTimeFormat: 'bars-beats',
      secondaryTimeFormat: null,
      displayFormat: 'bars-beats',
      sampleRate: 44100,
      ticksPerQuarter: 960,
      tempo: 120,
      timeSignature: [4, 4],
      tempoTrack: [],
      tempoMapEnabled: true,
      tempoRampEnabled: false,
      timeSignatureTrack: [],
      globalTempo: 120,
      masterTempo: true,
      markers: [],
      locators: [],
      selectedMarker: null,
      selectedLocator: null,
      ...initialState,
    };
    
    this.buildTempoMap();
  }

  // Initialize with Tone.Transport
  async initialize(transport: any): Promise<void> {
    this.transport = transport;
    // AudioContext not needed for transport, but kept for future use
    // this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    // Start position update loop
    this.startUpdateLoop();
  }

  private startUpdateLoop(): void {
    if (this.updateInterval) clearInterval(this.updateInterval);
    
    this.updateInterval = setInterval(() => {
      if (this.transport && this.state.isPlaying) {
        const seconds = this.transport.seconds;
        this.state.position = this.secondsToTimePosition(seconds);
      }
    }, 10); // Update every 10ms
  }

  // ===== TRANSPORT CONTROLS (Features 58-73) =====

  // Feature 58: Play
  play(): void {
    if (this.transport) {
      this.transport.start();
      this.state.isPlaying = true;
      this.state.isPaused = false;
    }
  }

  // Feature 59: Stop
  stop(): void {
    if (this.transport) {
      this.transport.stop();
      this.state.isPlaying = false;
      this.state.isPaused = false;
      this.state.position = this.createTimePosition(0);
    }
  }

  // Feature 60: Record
  record(): void {
    this.state.isRecording = true;
    this.play();
  }

  // Feature 61: Return to zero
  returnToZero(): void {
    this.stop();
    this.state.position = this.createTimePosition(0);
    if (this.transport) {
      this.transport.cancel();
      this.transport.seconds = 0;
    }
  }

  // Feature 62: Return to start marker
  returnToStartMarker(): void {
    const startLocator = this.state.locators.find(l => l.type === 'start');
    if (startLocator) {
      this.seekToPosition(startLocator.position);
    } else {
      this.returnToZero();
    }
  }

  // Feature 63: Play from cursor
  playFromCursor(position?: TimePosition): void {
    if (position) {
      this.seekToPosition(position);
    }
    this.play();
  }

  // Feature 64: Play from selection
  playFromSelection(start: TimePosition, end: TimePosition): void {
    this.seekToPosition(start);
    this.setLoopPoints(start, end);
    this.state.cycleEnabled = true;
    this.play();
  }

  // Feature 65: Loop playback
  toggleLoop(): void {
    this.state.cycleEnabled = !this.state.cycleEnabled;
    if (this.transport && this.state.loopStart && this.state.loopEnd) {
      if (this.state.cycleEnabled) {
        const startSeconds = this.timePositionToSeconds(this.state.loopStart);
        const endSeconds = this.timePositionToSeconds(this.state.loopEnd);
        this.transport.loopStart = startSeconds;
        this.transport.loopEnd = endSeconds;
        this.transport.loop = true;
      } else {
        this.transport.loop = false;
      }
    }
  }

  // Feature 66: Shuttle (variable speed forward/backward)
  setShuttleSpeed(speed: number): void {
    this.state.shuttleSpeed = Math.max(-1.0, Math.min(1.0, speed));
    if (this.transport) {
      this.transport.playbackRate = 1.0 + this.state.shuttleSpeed;
    }
  }

  // Feature 67: Scrub (scrub through audio)
  enableScrub(enabled: boolean): void {
    this.state.scrubEnabled = enabled;
    if (!enabled) {
      this.state.scrubPosition = null;
    }
  }

  scrubToPosition(position: TimePosition): void {
    this.state.scrubPosition = position;
    const seconds = this.timePositionToSeconds(position);
    if (this.transport) {
      this.transport.seconds = seconds;
    }
  }

  // Feature 68: Variable speed playback
  setVariableSpeed(speed: number): void {
    this.state.variableSpeed = Math.max(0.25, Math.min(4.0, speed));
    if (this.transport) {
      this.transport.playbackRate = this.state.variableSpeed;
    }
  }

  // Feature 69: Half speed playback
  setHalfSpeed(enabled: boolean): void {
    this.state.halfSpeed = enabled;
    if (enabled) {
      this.state.doubleSpeed = false;
      this.setVariableSpeed(0.5);
    } else if (!this.state.doubleSpeed) {
      this.setVariableSpeed(1.0);
    }
  }

  // Feature 70: Double speed playback
  setDoubleSpeed(enabled: boolean): void {
    this.state.doubleSpeed = enabled;
    if (enabled) {
      this.state.halfSpeed = false;
      this.setVariableSpeed(2.0);
    } else if (!this.state.halfSpeed) {
      this.setVariableSpeed(1.0);
    }
  }

  // Feature 71: Reverse playback
  setReverse(enabled: boolean): void {
    this.state.reverse = enabled;
    if (this.transport) {
      this.transport.playbackRate = enabled ? -this.state.variableSpeed : this.state.variableSpeed;
    }
  }

  // Feature 72: Frame-by-frame advance
  setFrameByFrame(enabled: boolean): void {
    this.state.frameByFrame = enabled;
  }

  advanceFrame(): void {
    if (this.state.frameByFrame) {
      const frameDuration = 1.0 / 30.0; // Assuming 30fps
      const currentSeconds = this.timePositionToSeconds(this.state.position);
      const newSeconds = currentSeconds + frameDuration;
      this.seekToPosition(this.secondsToTimePosition(newSeconds));
    }
  }

  // Feature 73: Sync to external clock
  setSyncSource(source: SyncSource): void {
    this.state.syncSource = source;
    this.state.syncEnabled = source !== 'none';
  }

  // ===== TIME FORMATS (Features 74-80) =====

  // Feature 74: Bars:Beats display
  setPrimaryTimeFormat(format: TimeFormat): void {
    this.state.primaryTimeFormat = format;
    this.state.displayFormat = format;
  }

  // Feature 75: Time display (HH:MM:SS:FF)
  setTimeDisplay(format: TimeFormat): void {
    this.state.displayFormat = format;
  }

  // Feature 76: Samples display
  setSamplesDisplay(): void {
    this.state.displayFormat = 'samples';
  }

  // Feature 77: Feet+Frames display
  setFeetFramesDisplay(): void {
    this.state.displayFormat = 'feet-frames';
  }

  // Feature 78: Secondary time format
  setSecondaryTimeFormat(format: TimeFormat | null): void {
    this.state.secondaryTimeFormat = format;
  }

  // Feature 79: Custom time format
  setCustomTimeFormat(_format: string): void {
    // Custom format parsing would go here
    this.state.displayFormat = 'custom';
  }

  // Feature 80: Time format preferences
  getTimeFormatString(position: TimePosition, format: TimeFormat): string {
    switch (format) {
      case 'bars-beats':
        return `${position.bars}:${position.beats}:${position.ticks}`;
      case 'time':
        return this.formatTimeCode(position);
      case 'samples':
        return position.samples.toString();
      case 'feet-frames':
        return this.formatFeetFrames(position);
      case 'seconds':
        return position.totalSeconds.toFixed(3) + 's';
      case 'minutes-seconds':
        return `${Math.floor(position.totalSeconds / 60)}:${(position.totalSeconds % 60).toFixed(2)}`;
      default:
        return this.formatTimeCode(position);
    }
  }

  // ===== TEMPO & TIME SIGNATURE (Features 81-94) =====

  // Feature 81: Global tempo
  setTempo(bpm: number): void {
    this.state.tempo = bpm;
    this.state.globalTempo = bpm;
    if (this.transport) {
      this.transport.bpm.value = bpm;
    }
  }

  // Feature 82: Tempo track
  setTempoTrack(events: TempoEvent[]): void {
    this.state.tempoTrack = events;
    this.buildTempoMap();
  }

  // Feature 83: Tempo map
  buildTempoMap(): void {
    this.tempoMap.clear();
    this.state.tempoTrack.forEach(event => {
      const timeKey = this.timePositionToSeconds(event.time as any);
      this.tempoMap.set(timeKey, event);
    });
  }

  // Feature 84: Time signature track
  setTimeSignatureTrack(events: TempoEvent[]): void {
    this.state.timeSignatureTrack = events;
  }

  // Feature 85: Tempo ramp
  setTempoRamp(enabled: boolean): void {
    this.state.tempoRampEnabled = enabled;
  }

  // Feature 86: Master tempo
  setMasterTempo(enabled: boolean): void {
    this.state.masterTempo = enabled;
  }

  // Feature 87: Tempo tap
  tapTempo(): void {
    // Implementation would track tap times and calculate BPM
  }

  // Feature 88: Tempo nudge
  nudgeTempo(amount: number): void {
    this.setTempo(this.state.tempo + amount);
  }

  // Feature 89: Time signature change
  setTimeSignature(numerator: number, denominator: number): void {
    this.state.timeSignature = [numerator, denominator];
  }

  // Feature 90-94: Additional tempo features
  getTempoAtPosition(position: TimePosition): number {
    if (!this.state.tempoMapEnabled) {
      return this.state.globalTempo;
    }
    
    const seconds = this.timePositionToSeconds(position);
    let currentTempo = this.state.globalTempo;
    
    for (const [time, event] of this.tempoMap.entries()) {
      if (time <= seconds) {
        currentTempo = event.tempo;
      } else {
        break;
      }
    }
    
    return currentTempo;
  }

  // ===== MARKERS & LOCATORS (Features 95-107) =====

  // Feature 95: Create marker
  createMarker(name: string, position: TimePosition, color?: string): Marker {
    const marker: Marker = {
      id: `marker-${Date.now()}-${Math.random()}`,
      name,
      position,
      color: color || '#ff6b6b',
      locked: false,
    };
    this.state.markers.push(marker);
    this.sortMarkers();
    return marker;
  }

  // Feature 96: Delete marker
  deleteMarker(id: string): void {
    this.state.markers = this.state.markers.filter(m => m.id !== id);
    if (this.state.selectedMarker === id) {
      this.state.selectedMarker = null;
    }
  }

  // Feature 97: Go to marker
  goToMarker(id: string): void {
    const marker = this.state.markers.find(m => m.id === id);
    if (marker) {
      this.seekToPosition(marker.position);
      this.state.selectedMarker = id;
    }
  }

  // Feature 98: Previous marker
  previousMarker(): void {
    const currentSeconds = this.timePositionToSeconds(this.state.position);
    const previous = this.state.markers
      .filter(m => this.timePositionToSeconds(m.position) < currentSeconds)
      .sort((a, b) => this.timePositionToSeconds(b.position) - this.timePositionToSeconds(a.position))[0];
    
    if (previous) {
      this.goToMarker(previous.id);
    }
  }

  // Feature 99: Next marker
  nextMarker(): void {
    const currentSeconds = this.timePositionToSeconds(this.state.position);
    const next = this.state.markers
      .filter(m => this.timePositionToSeconds(m.position) > currentSeconds)
      .sort((a, b) => this.timePositionToSeconds(a.position) - this.timePositionToSeconds(b.position))[0];
    
    if (next) {
      this.goToMarker(next.id);
    }
  }

  // Feature 100: Create locator
  createLocator(name: string, position: TimePosition, type: Locator['type']): Locator {
    const locator: Locator = {
      id: `locator-${Date.now()}-${Math.random()}`,
      name,
      position,
      type,
    };
    this.state.locators.push(locator);
    return locator;
  }

  // Additional locator features (101-107)
  deleteLocator(id: string): void {
    this.state.locators = this.state.locators.filter(l => l.id !== id);
    if (this.state.selectedLocator === id) {
      this.state.selectedLocator = null;
    }
  }

  goToLocator(id: string): void {
    const locator = this.state.locators.find(l => l.id === id);
    if (locator) {
      this.seekToPosition(locator.position);
      this.state.selectedLocator = id;
    }
  }

  setLoopPoints(start: TimePosition, end: TimePosition): void {
    this.state.loopStart = start;
    this.state.loopEnd = end;
    const loopStartLocator = this.state.locators.find(l => l.type === 'loop-start');
    const loopEndLocator = this.state.locators.find(l => l.type === 'loop-end');
    
    if (loopStartLocator) {
      loopStartLocator.position = start;
    } else {
      this.createLocator('Loop Start', start, 'loop-start');
    }
    
    if (loopEndLocator) {
      loopEndLocator.position = end;
    } else {
      this.createLocator('Loop End', end, 'loop-end');
    }
  }

  // ===== UTILITY METHODS =====

  private createTimePosition(seconds: number): TimePosition {
    const tempo = this.getTempoAtPosition(this.state.position);
    const beatsPerSecond = tempo / 60.0;
    const totalBeats = seconds * beatsPerSecond;
    const bars = Math.floor(totalBeats / this.state.timeSignature[0]);
    const beats = Math.floor(totalBeats % this.state.timeSignature[0]);
    const ticks = Math.floor((totalBeats % 1) * this.state.ticksPerQuarter);
    
    return {
      bars,
      beats,
      ticks,
      samples: Math.floor(seconds * this.state.sampleRate),
      seconds: seconds % 60,
      minutes: Math.floor(seconds / 60),
      totalSeconds: seconds,
    };
  }

  private secondsToTimePosition(seconds: number): TimePosition {
    return this.createTimePosition(seconds);
  }

  private timePositionToSeconds(position: TimePosition): number {
    return position.totalSeconds;
  }

  private seekToPosition(position: TimePosition): void {
    this.state.position = position;
    const seconds = this.timePositionToSeconds(position);
    if (this.transport) {
      this.transport.seconds = seconds;
    }
  }

  private formatTimeCode(position: TimePosition): string {
    const hours = Math.floor(position.totalSeconds / 3600);
    const minutes = Math.floor((position.totalSeconds % 3600) / 60);
    const secs = Math.floor(position.totalSeconds % 60);
    const frames = Math.floor((position.totalSeconds % 1) * 30); // 30fps
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`;
  }

  private formatFeetFrames(position: TimePosition): string {
    // 35mm film: 1 foot = 16 frames
    const totalFrames = Math.floor(position.totalSeconds * 30);
    const feet = Math.floor(totalFrames / 16);
    const frames = totalFrames % 16;
    return `${feet}+${frames}`;
  }

  private sortMarkers(): void {
    this.state.markers.sort((a, b) => 
      this.timePositionToSeconds(a.position) - this.timePositionToSeconds(b.position)
    );
  }

  // Getters
  getState(): TransportEngineState {
    return { ...this.state };
  }

  getPosition(): TimePosition {
    return { ...this.state.position };
  }

  getMarkers(): Marker[] {
    return [...this.state.markers];
  }

  getLocators(): Locator[] {
    return [...this.state.locators];
  }

  getTempo(): number {
    return this.state.tempo;
  }

  getTimeSignature(): [number, number] {
    return [...this.state.timeSignature];
  }
}
