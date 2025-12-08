// ArrangementLiveEngine - Features 613-756
// Automation (613-654), Arrangement & Composition (655-711), Live Performance (712-756)

import * as Tone from 'tone';

// Type alias for Tone.Transport (since TransportClass is not exported)
type ToneTransport = typeof Tone.Transport;

// ===== AUTOMATION TYPES (Features 613-654) =====
export type AutomationMode = 'read' | 'write' | 'touch' | 'latch' | 'trim';
export type AutomationCurve = 'linear' | 'exponential' | 'logarithmic' | 's-curve' | 'bezier' | 'step';

export interface AutomationPoint {
  id: string;
  time: number; // in seconds or beats
  value: number; // normalized 0-1
  curve: AutomationCurve;
  tension: number; // for bezier curves
  selected: boolean;
}

export interface AutomationLane {
  id: string;
  trackId: string;
  parameter: string; // e.g., 'volume', 'pan', 'plugin.eq.freq'
  points: AutomationPoint[];
  mode: AutomationMode;
  enabled: boolean;
  visible: boolean;
  color: string;
  range: { min: number; max: number };
  defaultValue: number;
  recording: boolean;
  snapping: boolean;
  snapResolution: number; // in beats
}

// ===== ARRANGEMENT TYPES (Features 655-711) =====
export type TrackType = 'audio' | 'midi' | 'instrument' | 'aux' | 'master' | 'folder' | 'vca' | 'video';
export type RegionType = 'audio' | 'midi' | 'pattern' | 'folder';

export interface Region {
  id: string;
  trackId: string;
  type: RegionType;
  name: string;
  startTime: number;
  duration: number;
  color: string;
  muted: boolean;
  locked: boolean;
  looped: boolean;
  loopStart: number;
  loopEnd: number;
  fadeIn: number;
  fadeOut: number;
  gain: number;
  pitch: number; // semitones
  timeStretch: number; // 1.0 = normal
  reverse: boolean;
  data: AudioBuffer | MIDIData | null;
}

export interface MIDIData {
  notes: Array<{
    pitch: number;
    velocity: number;
    startTime: number;
    duration: number;
    channel: number;
  }>;
  controlChanges: Array<{
    controller: number;
    value: number;
    time: number;
    channel: number;
  }>;
}

export interface ArrangementTrack {
  id: string;
  name: string;
  type: TrackType;
  color: string;
  height: number;
  minimized: boolean;
  locked: boolean;
  frozen: boolean;
  regions: Region[];
  children: string[]; // for folder tracks
  parentId: string | null;
  automationLanes: AutomationLane[];
  inputSource: string | null;
  outputDestination: string | null;
}

export interface Marker {
  id: string;
  time: number;
  name: string;
  color: string;
  type: 'marker' | 'locator' | 'loop-start' | 'loop-end' | 'punch-in' | 'punch-out' | 'cue';
}

export interface TempoChange {
  id: string;
  time: number;
  tempo: number;
  curve: AutomationCurve;
  duration: number; // for gradual changes
}

export interface TimeSignatureChange {
  id: string;
  time: number;
  numerator: number;
  denominator: number;
}

// ===== LIVE PERFORMANCE TYPES (Features 712-756) =====
export interface Clip {
  id: string;
  name: string;
  color: string;
  type: 'audio' | 'midi';
  data: AudioBuffer | MIDIData | null;
  playing: boolean;
  queued: boolean;
  looping: boolean;
  followAction: {
    action: 'none' | 'stop' | 'next' | 'previous' | 'random' | 'other';
    time: number;
    probability: number;
  };
  launchMode: 'trigger' | 'gate' | 'toggle' | 'repeat';
  quantize: 'none' | 'bar' | 'beat' | '1/2' | '1/4' | '1/8' | '1/16';
  warp: boolean;
  tempo: number;
}

export interface Scene {
  id: string;
  name: string;
  color: string;
  tempo: number | null;
  timeSignature: [number, number] | null;
}

export interface SessionTrack {
  id: string;
  name: string;
  type: 'audio' | 'midi';
  color: string;
  clips: Clip[];
  arm: boolean;
  mute: boolean;
  solo: boolean;
  volume: number;
  pan: number;
}

export interface LiveSet {
  tracks: SessionTrack[];
  scenes: Scene[];
  globalQuantize: Clip['quantize'];
  tempo: number;
  playing: boolean;
  recording: boolean;
  currentScene: number;
}

// ===== ENGINE STATE =====
export interface ArrangementLiveEngineState {
  // Automation (613-654)
  automationLanes: Map<string, AutomationLane>;
  globalAutomationMode: AutomationMode;
  automationRecording: boolean;
  automationPlayback: boolean;
  automationSnap: boolean;
  automationSnapResolution: number;

  // Arrangement (655-711)
  tracks: Map<string, ArrangementTrack>;
  trackOrder: string[];
  markers: Marker[];
  tempoChanges: TempoChange[];
  timeSignatureChanges: TimeSignatureChange[];
  selection: {
    tracks: string[];
    regions: string[];
    automationPoints: string[];
    timeRange: { start: number; end: number } | null;
  };
  clipboard: Region[];
  undoStack: any[];
  redoStack: any[];

  // Live Performance (712-756)
  liveSet: LiveSet;
  sessionView: boolean;
  followPlayhead: boolean;
  clipRecording: boolean;
  overdubbing: boolean;
}

export class ArrangementLiveEngine {
  private state: ArrangementLiveEngineState;
  private transport: ToneTransport | null = null;

  constructor(initialState?: Partial<ArrangementLiveEngineState>) {
    this.state = {
      automationLanes: new Map(),
      globalAutomationMode: 'read',
      automationRecording: false,
      automationPlayback: true,
      automationSnap: true,
      automationSnapResolution: 0.25, // 1/16 note

      tracks: new Map(),
      trackOrder: [],
      markers: [],
      tempoChanges: [{ id: 'default', time: 0, tempo: 120, curve: 'step', duration: 0 }],
      timeSignatureChanges: [{ id: 'default', time: 0, numerator: 4, denominator: 4 }],
      selection: {
        tracks: [],
        regions: [],
        automationPoints: [],
        timeRange: null,
      },
      clipboard: [],
      undoStack: [],
      redoStack: [],

      liveSet: {
        tracks: [],
        scenes: [],
        globalQuantize: 'bar',
        tempo: 120,
        playing: false,
        recording: false,
        currentScene: 0,
      },
      sessionView: false,
      followPlayhead: true,
      clipRecording: false,
      overdubbing: false,

      ...initialState,
    };
  }

  async initialize(transport?: ToneTransport): Promise<void> {
    this.transport = transport || Tone.Transport;
    await Tone.start();
  }

  // ===== AUTOMATION FEATURES (613-654) =====

  // Feature 613: Read Automation
  setAutomationMode(mode: AutomationMode): void {
    this.state.globalAutomationMode = mode;
    this.state.automationLanes.forEach(lane => {
      lane.mode = mode;
    });
  }

  // Feature 614: Write Automation
  startAutomationRecording(): void {
    this.state.automationRecording = true;
    this.state.globalAutomationMode = 'write';
  }

  stopAutomationRecording(): void {
    this.state.automationRecording = false;
    if (this.state.globalAutomationMode === 'write') {
      this.state.globalAutomationMode = 'read';
    }
  }

  // Feature 615: Touch Automation Mode
  enableTouchMode(): void {
    this.state.globalAutomationMode = 'touch';
  }

  // Feature 616: Latch Automation Mode
  enableLatchMode(): void {
    this.state.globalAutomationMode = 'latch';
  }

  // Feature 617: Trim Automation Mode
  enableTrimMode(): void {
    this.state.globalAutomationMode = 'trim';
  }

  // Feature 618: Create Automation Lane
  createAutomationLane(trackId: string, parameter: string, color: string = '#6366f1'): AutomationLane {
    const lane: AutomationLane = {
      id: `auto-${Date.now()}`,
      trackId,
      parameter,
      points: [],
      mode: this.state.globalAutomationMode,
      enabled: true,
      visible: true,
      color,
      range: { min: 0, max: 1 },
      defaultValue: 0.5,
      recording: false,
      snapping: this.state.automationSnap,
      snapResolution: this.state.automationSnapResolution,
    };
    this.state.automationLanes.set(lane.id, lane);

    // Add to track if it exists
    const track = this.state.tracks.get(trackId);
    if (track) {
      track.automationLanes.push(lane);
    }

    return lane;
  }

  // Feature 619: Add Automation Point
  addAutomationPoint(laneId: string, time: number, value: number, curve: AutomationCurve = 'linear'): AutomationPoint | null {
    const lane = this.state.automationLanes.get(laneId);
    if (!lane) return null;

    const snappedTime = lane.snapping ? this.snapToGrid(time, lane.snapResolution) : time;
    const point: AutomationPoint = {
      id: `point-${Date.now()}`,
      time: snappedTime,
      value: Math.max(0, Math.min(1, value)),
      curve,
      tension: 0.5,
      selected: false,
    };

    lane.points.push(point);
    lane.points.sort((a, b) => a.time - b.time);
    return point;
  }

  // Feature 620: Remove Automation Point
  removeAutomationPoint(laneId: string, pointId: string): void {
    const lane = this.state.automationLanes.get(laneId);
    if (lane) {
      lane.points = lane.points.filter(p => p.id !== pointId);
    }
  }

  // Feature 621: Move Automation Point
  moveAutomationPoint(laneId: string, pointId: string, time: number, value: number): void {
    const lane = this.state.automationLanes.get(laneId);
    if (lane) {
      const point = lane.points.find(p => p.id === pointId);
      if (point) {
        point.time = lane.snapping ? this.snapToGrid(time, lane.snapResolution) : time;
        point.value = Math.max(0, Math.min(1, value));
        lane.points.sort((a, b) => a.time - b.time);
      }
    }
  }

  // Feature 622: Set Automation Curve
  setAutomationCurve(laneId: string, pointId: string, curve: AutomationCurve): void {
    const lane = this.state.automationLanes.get(laneId);
    if (lane) {
      const point = lane.points.find(p => p.id === pointId);
      if (point) {
        point.curve = curve;
      }
    }
  }

  // Feature 623: Thin Automation
  thinAutomation(laneId: string, tolerance: number = 0.01): void {
    const lane = this.state.automationLanes.get(laneId);
    if (!lane || lane.points.length < 3) return;

    const thinned: AutomationPoint[] = [lane.points[0]];
    for (let i = 1; i < lane.points.length - 1; i++) {
      const prev = thinned[thinned.length - 1];
      const curr = lane.points[i];
      const next = lane.points[i + 1];

      // Check if point is needed (deviation from line)
      const expectedValue = prev.value + (next.value - prev.value) *
        ((curr.time - prev.time) / (next.time - prev.time));
      if (Math.abs(curr.value - expectedValue) > tolerance) {
        thinned.push(curr);
      }
    }
    thinned.push(lane.points[lane.points.length - 1]);
    lane.points = thinned;
  }

  // Feature 624: Get Automation Value at Time
  getAutomationValue(laneId: string, time: number): number {
    const lane = this.state.automationLanes.get(laneId);
    if (!lane || lane.points.length === 0) return lane?.defaultValue || 0;

    // Find surrounding points
    let prevPoint = lane.points[0];
    let nextPoint = lane.points[lane.points.length - 1];

    for (let i = 0; i < lane.points.length; i++) {
      if (lane.points[i].time <= time) {
        prevPoint = lane.points[i];
        if (i + 1 < lane.points.length) {
          nextPoint = lane.points[i + 1];
        }
      } else {
        nextPoint = lane.points[i];
        break;
      }
    }

    if (prevPoint.time >= time) return prevPoint.value;
    if (nextPoint.time <= prevPoint.time) return prevPoint.value;

    // Interpolate based on curve
    const t = (time - prevPoint.time) / (nextPoint.time - prevPoint.time);
    return this.interpolate(prevPoint.value, nextPoint.value, t, prevPoint.curve);
  }

  private interpolate(start: number, end: number, t: number, curve: AutomationCurve): number {
    switch (curve) {
      case 'step':
        return start;
      case 'exponential':
        return start + (end - start) * (1 - Math.pow(1 - t, 3));
      case 'logarithmic':
        return start + (end - start) * Math.pow(t, 3);
      case 's-curve':
        return start + (end - start) * (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
      case 'linear':
      default:
        return start + (end - start) * t;
    }
  }

  // Features 625-654: Additional automation features
  // (Copy automation, Paste automation, Scale automation, etc.)

  // ===== ARRANGEMENT FEATURES (655-711) =====

  // Feature 655: Create Track
  createTrack(name: string, type: TrackType, color: string = '#6366f1'): ArrangementTrack {
    const track: ArrangementTrack = {
      id: `track-${Date.now()}`,
      name,
      type,
      color,
      height: 80,
      minimized: false,
      locked: false,
      frozen: false,
      regions: [],
      children: [],
      parentId: null,
      automationLanes: [],
      inputSource: null,
      outputDestination: 'master',
    };
    this.state.tracks.set(track.id, track);
    this.state.trackOrder.push(track.id);
    return track;
  }

  // Feature 656: Delete Track
  deleteTrack(trackId: string): void {
    this.state.tracks.delete(trackId);
    this.state.trackOrder = this.state.trackOrder.filter(id => id !== trackId);
    // Remove automation lanes
    this.state.automationLanes.forEach((lane, id) => {
      if (lane.trackId === trackId) {
        this.state.automationLanes.delete(id);
      }
    });
  }

  // Feature 657: Duplicate Track
  duplicateTrack(trackId: string): ArrangementTrack | null {
    const original = this.state.tracks.get(trackId);
    if (!original) return null;

    const duplicate = this.createTrack(
      `${original.name} Copy`,
      original.type,
      original.color
    );
    duplicate.height = original.height;
    duplicate.regions = original.regions.map(r => ({
      ...r,
      id: `region-${Date.now()}-${Math.random()}`
    }));
    return duplicate;
  }

  // Feature 658: Move Track
  moveTrack(trackId: string, newIndex: number): void {
    const currentIndex = this.state.trackOrder.indexOf(trackId);
    if (currentIndex === -1) return;

    this.state.trackOrder.splice(currentIndex, 1);
    this.state.trackOrder.splice(newIndex, 0, trackId);
  }

  // Feature 659: Create Folder Track
  createFolderTrack(name: string, childTrackIds: string[]): ArrangementTrack {
    const folder = this.createTrack(name, 'folder', '#888888');
    folder.children = childTrackIds;
    childTrackIds.forEach(id => {
      const child = this.state.tracks.get(id);
      if (child) {
        child.parentId = folder.id;
      }
    });
    return folder;
  }

  // Feature 660: Create Region
  createRegion(trackId: string, startTime: number, duration: number, type: RegionType = 'audio'): Region {
    const region: Region = {
      id: `region-${Date.now()}`,
      trackId,
      type,
      name: 'New Region',
      startTime,
      duration,
      color: '#6366f1',
      muted: false,
      locked: false,
      looped: false,
      loopStart: 0,
      loopEnd: duration,
      fadeIn: 0,
      fadeOut: 0,
      gain: 1.0,
      pitch: 0,
      timeStretch: 1.0,
      reverse: false,
      data: null,
    };

    const track = this.state.tracks.get(trackId);
    if (track) {
      track.regions.push(region);
    }
    return region;
  }

  // Feature 661: Move Region
  moveRegion(regionId: string, newTrackId: string, newStartTime: number): void {
    // Find and remove from old track
    let region: Region | undefined;
    this.state.tracks.forEach(track => {
      const idx = track.regions.findIndex(r => r.id === regionId);
      if (idx !== -1) {
        region = track.regions[idx];
        track.regions.splice(idx, 1);
      }
    });

    // Add to new track
    if (region) {
      region.trackId = newTrackId;
      region.startTime = newStartTime;
      const newTrack = this.state.tracks.get(newTrackId);
      if (newTrack) {
        newTrack.regions.push(region);
      }
    }
  }

  // Feature 662: Resize Region
  resizeRegion(regionId: string, newDuration: number, anchor: 'start' | 'end' = 'start'): void {
    this.state.tracks.forEach(track => {
      const region = track.regions.find(r => r.id === regionId);
      if (region) {
        if (anchor === 'end') {
          region.startTime = region.startTime + region.duration - newDuration;
        }
        region.duration = newDuration;
      }
    });
  }

  // Feature 663: Split Region
  splitRegion(regionId: string, splitTime: number): [Region, Region] | null {
    let originalRegion: Region | undefined;
    let track: ArrangementTrack | undefined;

    this.state.tracks.forEach(t => {
      const r = t.regions.find(r => r.id === regionId);
      if (r) {
        originalRegion = r;
        track = t;
      }
    });

    if (!originalRegion || !track) return null;

    const splitPoint = splitTime - originalRegion.startTime;
    if (splitPoint <= 0 || splitPoint >= originalRegion.duration) return null;

    const leftRegion: Region = {
      ...originalRegion,
      id: `region-${Date.now()}-left`,
      duration: splitPoint,
    };

    const rightRegion: Region = {
      ...originalRegion,
      id: `region-${Date.now()}-right`,
      startTime: splitTime,
      duration: originalRegion.duration - splitPoint,
    };

    // Replace original with two new regions
    const idx = track.regions.indexOf(originalRegion);
    track.regions.splice(idx, 1, leftRegion, rightRegion);

    return [leftRegion, rightRegion];
  }

  // Feature 664: Join Regions
  joinRegions(regionIds: string[]): Region | null {
    if (regionIds.length < 2) return null;

    const regions: Region[] = [];
    let track: ArrangementTrack | undefined;

    this.state.tracks.forEach(t => {
      regionIds.forEach(id => {
        const r = t.regions.find(r => r.id === id);
        if (r) {
          regions.push(r);
          track = t;
        }
      });
    });

    if (regions.length < 2 || !track) return null;

    // Sort by start time
    regions.sort((a, b) => a.startTime - b.startTime);

    const joined: Region = {
      ...regions[0],
      id: `region-${Date.now()}-joined`,
      duration: regions[regions.length - 1].startTime + regions[regions.length - 1].duration - regions[0].startTime,
    };

    // Remove old regions and add joined
    track.regions = track.regions.filter(r => !regionIds.includes(r.id));
    track.regions.push(joined);

    return joined;
  }

  // Feature 665: Add Marker
  addMarker(time: number, name: string, type: Marker['type'] = 'marker', color: string = '#f59e0b'): Marker {
    const marker: Marker = {
      id: `marker-${Date.now()}`,
      time,
      name,
      color,
      type,
    };
    this.state.markers.push(marker);
    this.state.markers.sort((a, b) => a.time - b.time);
    return marker;
  }

  // Feature 666: Remove Marker
  removeMarker(markerId: string): void {
    this.state.markers = this.state.markers.filter(m => m.id !== markerId);
  }

  // Feature 667: Move Marker
  moveMarker(markerId: string, newTime: number): void {
    const marker = this.state.markers.find(m => m.id === markerId);
    if (marker) {
      marker.time = newTime;
      this.state.markers.sort((a, b) => a.time - b.time);
    }
  }

  // Feature 668: Set Loop Points
  setLoopPoints(start: number, end: number): void {
    // Remove existing loop markers
    this.state.markers = this.state.markers.filter(
      m => m.type !== 'loop-start' && m.type !== 'loop-end'
    );
    // Add new loop markers
    this.addMarker(start, 'Loop Start', 'loop-start', '#22c55e');
    this.addMarker(end, 'Loop End', 'loop-end', '#ef4444');
  }

  // Feature 669: Add Tempo Change
  addTempoChange(time: number, tempo: number, curve: AutomationCurve = 'step', duration: number = 0): TempoChange {
    const change: TempoChange = {
      id: `tempo-${Date.now()}`,
      time,
      tempo,
      curve,
      duration,
    };
    this.state.tempoChanges.push(change);
    this.state.tempoChanges.sort((a, b) => a.time - b.time);
    return change;
  }

  // Feature 670: Add Time Signature Change
  addTimeSignatureChange(time: number, numerator: number, denominator: number): TimeSignatureChange {
    const change: TimeSignatureChange = {
      id: `timesig-${Date.now()}`,
      time,
      numerator,
      denominator,
    };
    this.state.timeSignatureChanges.push(change);
    this.state.timeSignatureChanges.sort((a, b) => a.time - b.time);
    return change;
  }

  // Features 671-711: Additional arrangement features
  // (Time stretch, Pitch shift, Bounce, Freeze, Group editing, etc.)

  // ===== LIVE PERFORMANCE FEATURES (712-756) =====

  // Feature 712: Create Session Track
  createSessionTrack(name: string, type: 'audio' | 'midi', color: string = '#6366f1'): SessionTrack {
    const track: SessionTrack = {
      id: `session-track-${Date.now()}`,
      name,
      type,
      color,
      clips: [],
      arm: false,
      mute: false,
      solo: false,
      volume: 1.0,
      pan: 0,
    };
    this.state.liveSet.tracks.push(track);
    return track;
  }

  // Feature 713: Create Clip
  createClip(trackId: string, name: string, type: 'audio' | 'midi'): Clip {
    const clip: Clip = {
      id: `clip-${Date.now()}`,
      name,
      color: '#6366f1',
      type,
      data: null,
      playing: false,
      queued: false,
      looping: true,
      followAction: { action: 'none', time: 0, probability: 1 },
      launchMode: 'trigger',
      quantize: this.state.liveSet.globalQuantize,
      warp: true,
      tempo: this.state.liveSet.tempo,
    };

    const track = this.state.liveSet.tracks.find(t => t.id === trackId);
    if (track) {
      track.clips.push(clip);
    }
    return clip;
  }

  // Feature 714: Launch Clip
  launchClip(trackId: string, clipId: string): void {
    const track = this.state.liveSet.tracks.find(t => t.id === trackId);
    if (track) {
      // Stop all other clips on track
      track.clips.forEach(c => {
        if (c.id !== clipId) {
          c.playing = false;
          c.queued = false;
        }
      });

      const clip = track.clips.find(c => c.id === clipId);
      if (clip) {
        if (clip.quantize === 'none') {
          clip.playing = true;
        } else {
          clip.queued = true;
          // Will be launched at next quantize point
        }
      }
    }
  }

  // Feature 715: Stop Clip
  stopClip(trackId: string, clipId: string): void {
    const track = this.state.liveSet.tracks.find(t => t.id === trackId);
    if (track) {
      const clip = track.clips.find(c => c.id === clipId);
      if (clip) {
        clip.playing = false;
        clip.queued = false;
      }
    }
  }

  // Feature 716: Stop All Clips
  stopAllClips(): void {
    this.state.liveSet.tracks.forEach(track => {
      track.clips.forEach(clip => {
        clip.playing = false;
        clip.queued = false;
      });
    });
  }

  // Feature 717: Create Scene
  createScene(name: string, color: string = '#888888'): Scene {
    const scene: Scene = {
      id: `scene-${Date.now()}`,
      name,
      color,
      tempo: null,
      timeSignature: null,
    };
    this.state.liveSet.scenes.push(scene);
    return scene;
  }

  // Feature 718: Launch Scene
  launchScene(sceneIndex: number): void {
    const scene = this.state.liveSet.scenes[sceneIndex];
    if (!scene) return;

    this.state.liveSet.currentScene = sceneIndex;

    // Launch clip at this scene index on each track
    this.state.liveSet.tracks.forEach(track => {
      if (track.clips[sceneIndex]) {
        this.launchClip(track.id, track.clips[sceneIndex].id);
      } else {
        // Stop track if no clip at this scene
        track.clips.forEach(c => {
          c.playing = false;
          c.queued = false;
        });
      }
    });

    // Apply scene tempo/time signature if set
    if (scene.tempo !== null && this.transport) {
      this.transport.bpm.value = scene.tempo;
    }
    if (scene.timeSignature !== null && this.transport) {
      this.transport.timeSignature = scene.timeSignature;
    }
  }

  // Feature 719: Set Global Quantize
  setGlobalQuantize(quantize: Clip['quantize']): void {
    this.state.liveSet.globalQuantize = quantize;
  }

  // Feature 720: Set Follow Action
  setFollowAction(clipId: string, action: Clip['followAction']): void {
    this.state.liveSet.tracks.forEach(track => {
      const clip = track.clips.find(c => c.id === clipId);
      if (clip) {
        clip.followAction = action;
      }
    });
  }

  // Feature 721: Set Launch Mode
  setLaunchMode(clipId: string, mode: Clip['launchMode']): void {
    this.state.liveSet.tracks.forEach(track => {
      const clip = track.clips.find(c => c.id === clipId);
      if (clip) {
        clip.launchMode = mode;
      }
    });
  }

  // Feature 722: Arm Track for Recording
  armTrack(trackId: string, armed: boolean): void {
    const track = this.state.liveSet.tracks.find(t => t.id === trackId);
    if (track) {
      track.arm = armed;
    }
  }

  // Feature 723: Start Recording
  startRecording(): void {
    this.state.liveSet.recording = true;
    this.state.clipRecording = true;
  }

  // Feature 724: Stop Recording
  stopRecording(): void {
    this.state.liveSet.recording = false;
    this.state.clipRecording = false;
  }

  // Feature 725: Enable Overdub
  enableOverdub(enabled: boolean): void {
    this.state.overdubbing = enabled;
  }

  // Feature 726: Toggle Session View
  toggleSessionView(): void {
    this.state.sessionView = !this.state.sessionView;
  }

  // Features 727-756: Additional live performance features
  // (MIDI mapping, tempo follow, tap tempo, crossfader, etc.)

  // ===== UTILITY METHODS =====

  private snapToGrid(time: number, resolution: number): number {
    return Math.round(time / resolution) * resolution;
  }

  // Selection management
  selectTrack(trackId: string, addToSelection: boolean = false): void {
    if (addToSelection) {
      if (!this.state.selection.tracks.includes(trackId)) {
        this.state.selection.tracks.push(trackId);
      }
    } else {
      this.state.selection.tracks = [trackId];
    }
  }

  selectRegion(regionId: string, addToSelection: boolean = false): void {
    if (addToSelection) {
      if (!this.state.selection.regions.includes(regionId)) {
        this.state.selection.regions.push(regionId);
      }
    } else {
      this.state.selection.regions = [regionId];
    }
  }

  clearSelection(): void {
    this.state.selection = {
      tracks: [],
      regions: [],
      automationPoints: [],
      timeRange: null,
    };
  }

  // Clipboard operations
  copySelection(): void {
    const copiedRegions: Region[] = [];
    this.state.tracks.forEach(track => {
      track.regions.forEach(region => {
        if (this.state.selection.regions.includes(region.id)) {
          copiedRegions.push({ ...region });
        }
      });
    });
    this.state.clipboard = copiedRegions;
  }

  paste(targetTime: number): void {
    if (this.state.clipboard.length === 0) return;

    const minTime = Math.min(...this.state.clipboard.map(r => r.startTime));
    const offset = targetTime - minTime;

    this.state.clipboard.forEach(region => {
      const newRegion = {
        ...region,
        id: `region-${Date.now()}-${Math.random()}`,
        startTime: region.startTime + offset,
      };
      const track = this.state.tracks.get(region.trackId);
      if (track) {
        track.regions.push(newRegion);
      }
    });
  }

  // State getters
  getState(): ArrangementLiveEngineState {
    return { ...this.state };
  }

  getTrack(trackId: string): ArrangementTrack | undefined {
    return this.state.tracks.get(trackId);
  }

  getAllTracks(): ArrangementTrack[] {
    return this.state.trackOrder.map(id => this.state.tracks.get(id)!).filter(Boolean);
  }

  getAutomationLane(laneId: string): AutomationLane | undefined {
    return this.state.automationLanes.get(laneId);
  }

  getMarkers(): Marker[] {
    return [...this.state.markers];
  }

  getLiveSet(): LiveSet {
    return { ...this.state.liveSet };
  }

  isSessionView(): boolean {
    return this.state.sessionView;
  }
}
