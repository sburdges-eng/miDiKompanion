// PlaybackEngine - Comprehensive playback functionality

import * as Tone from 'tone';

export interface PlaybackState {
  isPlaying: boolean;
  isPaused: boolean;
  currentTime: number;
  playheadPosition: number; // in bars
  playbackSpeed: number; // 1.0 = normal, 0.5 = half, 2.0 = double, -1.0 = reverse
  isLooping: boolean;
  loopStart: number;
  loopEnd: number;
  selectionStart: number | null;
  selectionEnd: number | null;
  startMarker: number;
  cursorPosition: number;
  isReversed: boolean;
  frameRate: number; // for frame-by-frame
}

export interface PlaybackConfig {
  tempo: number;
  timeSignature: [number, number];
  sampleRate: number;
}

export class PlaybackEngine {
  private state: PlaybackState;
  private config: PlaybackConfig;
  private transport: typeof Tone.Transport | null = null;
  private scheduledEvents: number[] = [];
  private audioContext: AudioContext | null = null;
  private playbackSource: AudioBufferSourceNode | null = null;
  private playbackRateNode: GainNode | null = null;
  private soloStates: Map<string, boolean> = new Map();
  private muteStates: Map<string, boolean> = new Map();
  private soloSafeTracks: Set<string> = new Set();
  private cueSends: Map<string, number> = new Map(); // trackId -> send level

  constructor(config: PlaybackConfig) {
    this.config = config;
    this.state = {
      isPlaying: false,
      isPaused: false,
      currentTime: 0,
      playheadPosition: 0,
      playbackSpeed: 1.0,
      isLooping: false,
      loopStart: 0,
      loopEnd: 16,
      selectionStart: null,
      selectionEnd: null,
      startMarker: 0,
      cursorPosition: 0,
      isReversed: false,
      frameRate: 30, // 30 fps for video sync
    };
  }

  async initialize(): Promise<void> {
    if (Tone.context.state !== 'running') {
      await Tone.start();
    }
    this.audioContext = Tone.context.rawContext as AudioContext;
    this.transport = Tone.Transport;
  }

  // 26. Play/pause toggle
  async togglePlayPause(): Promise<void> {
    if (this.state.isPlaying) {
      await this.pause();
    } else {
      await this.play();
    }
  }

  async play(): Promise<void> {
    if (!this.audioContext) {
      await this.initialize();
    }

    if (this.state.isPaused) {
      // Resume from pause
      this.transport?.start();
      this.state.isPaused = false;
    } else {
      // Start new playback
      this.transport?.start();
    }

    this.state.isPlaying = true;
    this.startPlaybackLoop();
  }

  async pause(): Promise<void> {
    this.transport?.pause();
    this.state.isPlaying = false;
    this.state.isPaused = true;
  }

  // 27. Stop
  stop(): void {
    this.transport?.stop();
    this.transport?.cancel();
    this.scheduledEvents.forEach((id) => this.transport?.clear(id));
    this.scheduledEvents = [];
    this.state.isPlaying = false;
    this.state.isPaused = false;
    this.state.currentTime = 0;
    this.state.playheadPosition = 0;
  }

  // 28. Return to zero
  returnToZero(): void {
    this.stop();
    this.state.playheadPosition = 0;
    this.state.cursorPosition = 0;
    this.state.currentTime = 0;
  }

  // 29. Return to start marker
  returnToStartMarker(): void {
    this.stop();
    this.state.playheadPosition = this.state.startMarker;
    this.state.cursorPosition = this.state.startMarker;
    this.state.currentTime = this.barsToSeconds(this.state.startMarker);
  }

  // 30. Play from cursor
  async playFromCursor(): Promise<void> {
    this.state.playheadPosition = this.state.cursorPosition;
    this.state.currentTime = this.barsToSeconds(this.state.cursorPosition);
    await this.play();
  }

  // 31. Play from selection
  async playFromSelection(): Promise<void> {
    if (this.state.selectionStart !== null) {
      this.state.playheadPosition = this.state.selectionStart;
      this.state.currentTime = this.barsToSeconds(this.state.selectionStart);
      await this.play();
    }
  }

  // 32. Play selection only
  async playSelectionOnly(): Promise<void> {
    if (this.state.selectionStart !== null && this.state.selectionEnd !== null) {
      this.state.isLooping = true;
      this.state.loopStart = this.state.selectionStart;
      this.state.loopEnd = this.state.selectionEnd;
      this.state.playheadPosition = this.state.selectionStart;
      this.state.currentTime = this.barsToSeconds(this.state.selectionStart);
      await this.play();
    }
  }

  // 33. Loop playback
  setLooping(enabled: boolean, start?: number, end?: number): void {
    this.state.isLooping = enabled;
    if (start !== undefined) this.state.loopStart = start;
    if (end !== undefined) this.state.loopEnd = end;
  }

  // 34. Shuttle/scrub playback
  startScrub(position: number): void {
    this.state.playheadPosition = position;
    this.state.currentTime = this.barsToSeconds(position);
    // Scrub audio (would need audio buffer)
    this.scrubAudio(position);
  }

  stopScrub(): void {
    // Scrub stopped
  }

  scrubAudio(_position: number): void {
    // Implementation would scrub through audio buffer
    // This is a placeholder for actual audio scrubbing
  }

  // 35. Variable speed playback
  setPlaybackSpeed(speed: number): void {
    this.state.playbackSpeed = Math.max(-2.0, Math.min(2.0, speed));
    if (this.playbackRateNode) {
      this.playbackRateNode.gain.value = this.state.playbackSpeed;
    }
    if (this.transport) {
      this.transport.bpm.value = this.config.tempo * Math.abs(this.state.playbackSpeed);
    }
  }

  // 36. Half-speed playback
  setHalfSpeed(): void {
    this.setPlaybackSpeed(0.5);
  }

  // 37. Double-speed playback
  setDoubleSpeed(): void {
    this.setPlaybackSpeed(2.0);
  }

  // 38. Reverse playback
  setReverse(enabled: boolean): void {
    this.state.isReversed = enabled;
    if (enabled) {
      this.setPlaybackSpeed(-1.0);
    } else {
      this.setPlaybackSpeed(1.0);
    }
  }

  // 39. Frame-by-frame advance
  advanceFrame(): void {
    const barDuration = this.barsToSeconds(1);
    const framesPerBar = barDuration * this.state.frameRate;
    const barIncrement = 1.0 / framesPerBar;

    if (this.state.isReversed) {
      this.state.playheadPosition = Math.max(0, this.state.playheadPosition - barIncrement);
    } else {
      this.state.playheadPosition += barIncrement;
    }
    this.state.currentTime = this.barsToSeconds(this.state.playheadPosition);
  }

  // 40. Pre-listen/audition
  async preListen(trackId: string, duration: number = 2.0): Promise<void> {
    // Play a short preview of the track
    // Implementation would load and play track audio
    console.log(`Pre-listening to track ${trackId} for ${duration} seconds`);
  }

  // 41. Solo in place
  soloInPlace(trackId: string, enabled: boolean): void {
    this.soloStates.set(trackId, enabled);
    this.updateSoloStates();
  }

  // 42. Solo defeat
  soloDefeat(): void {
    this.soloStates.clear();
    this.updateSoloStates();
  }

  // 43. Mute
  mute(trackId: string, enabled: boolean): void {
    this.muteStates.set(trackId, enabled);
  }

  // 44. Exclusive solo
  setExclusiveSolo(trackId: string): void {
    // Mute all others, solo only this one
    this.soloStates.clear();
    this.soloStates.set(trackId, true);
    this.updateSoloStates();
  }

  // 45. X-OR solo (cancel others)
  setXORSolo(trackId: string): void {
    // If this track is already soloed, unsolo it; otherwise solo only this one
    const isCurrentlySoloed = this.soloStates.get(trackId);
    this.soloStates.clear();
    if (!isCurrentlySoloed) {
      this.soloStates.set(trackId, true);
    }
    this.updateSoloStates();
  }

  // 46. Solo-safe
  setSoloSafe(trackId: string, safe: boolean): void {
    if (safe) {
      this.soloSafeTracks.add(trackId);
    } else {
      this.soloSafeTracks.delete(trackId);
    }
  }

  // 47. Listen bus/AFL/PFL
  setListenBus(_mode: 'AFL' | 'PFL' | 'off', _trackId?: string): void {
    // Implementation would route audio to listen bus
    // This is a placeholder for actual audio routing
  }

  // 48. Cue mix sends
  setCueSend(trackId: string, level: number): void {
    this.cueSends.set(trackId, Math.max(0, Math.min(1, level)));
  }

  getCueSend(trackId: string): number {
    return this.cueSends.get(trackId) || 0;
  }

  // Helper methods
  private updateSoloStates(): void {
    // Update audio routing based on solo states
    // This would control actual audio routing in the mixer
    const hasSoloed = Array.from(this.soloStates.values()).some((v) => v);

    if (hasSoloed) {
      // Mute all non-soloed tracks (except solo-safe)
      // Implementation would mute tracks in the mixer
    } else {
      // Unmute all tracks
      // Implementation would unmute tracks in the mixer
    }
  }

  private startPlaybackLoop(): void {
    const updateLoop = () => {
      if (!this.state.isPlaying) return;

      if (this.state.isReversed) {
        this.state.playheadPosition -= 0.01;
        if (this.state.playheadPosition < 0) {
          this.state.playheadPosition = 0;
          if (this.state.isLooping) {
            this.state.playheadPosition = this.state.loopEnd;
          } else {
            this.stop();
            return;
          }
        }
      } else {
        this.state.playheadPosition += 0.01 * this.state.playbackSpeed;
        if (this.state.isLooping) {
          if (this.state.playheadPosition >= this.state.loopEnd) {
            this.state.playheadPosition = this.state.loopStart;
          }
        }
      }

      this.state.currentTime = this.barsToSeconds(this.state.playheadPosition);

      if (this.state.isPlaying) {
        requestAnimationFrame(updateLoop);
      }
    };

    requestAnimationFrame(updateLoop);
  }

  private barsToSeconds(bars: number): number {
    const beatsPerBar = this.config.timeSignature[0];
    const beatsPerMinute = this.config.tempo;
    const secondsPerBeat = 60.0 / beatsPerMinute;
    return bars * beatsPerBar * secondsPerBeat;
  }

  // Getters
  getState(): PlaybackState {
    return { ...this.state };
  }

  getPlaybackSpeed(): number {
    return this.state.playbackSpeed;
  }

  isTrackSoloed(trackId: string): boolean {
    return this.soloStates.get(trackId) || false;
  }

  isTrackMuted(trackId: string): boolean {
    return this.muteStates.get(trackId) || false;
  }

  isTrackSoloSafe(trackId: string): boolean {
    return this.soloSafeTracks.has(trackId);
  }

  // Setters
  setCursorPosition(position: number): void {
    this.state.cursorPosition = position;
  }

  setSelection(start: number | null, end: number | null): void {
    this.state.selectionStart = start;
    this.state.selectionEnd = end;
  }

  setStartMarker(position: number): void {
    this.state.startMarker = position;
  }

  setTempo(tempo: number): void {
    this.config.tempo = tempo;
    if (this.transport) {
      this.transport.bpm.value = tempo;
    }
  }

  cleanup(): void {
    this.stop();
    if (this.playbackSource) {
      this.playbackSource.disconnect();
    }
    if (this.audioContext) {
      this.audioContext.close();
    }
  }
}
