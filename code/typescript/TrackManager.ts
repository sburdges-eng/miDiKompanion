/**
 * Track Manager - Multi-track audio and MIDI management
 * Implements Parts 3, 4, 5, 9 of the DAW specification
 */

import { audioEngine } from './AudioEngine';

// Track types (Items 655-669)
export type TrackType = 'audio' | 'midi' | 'instrument' | 'aux' | 'master' | 'folder' | 'vca' | 'marker' | 'tempo' | 'chord' | 'video';

// Audio region/clip
export interface AudioRegion {
  id: string;
  trackId: string;
  startTime: number;
  duration: number;
  offset: number; // offset within the source file
  sourceFile: string | AudioBuffer;
  name: string;
  color: string;
  gain: number;
  fadeIn: number;
  fadeOut: number;
  fadeInCurve: 'linear' | 'exponential' | 'scurve';
  fadeOutCurve: 'linear' | 'exponential' | 'scurve';
  muted: boolean;
  locked: boolean;
  pitch: number;
  tempo: number | null;
  warpEnabled: boolean;
  warpMarkers: { time: number; position: number }[];
}

// MIDI note
export interface MidiNote {
  id: string;
  pitch: number; // 0-127
  velocity: number; // 0-127
  startTime: number;
  duration: number;
  channel: number;
  muted: boolean;
}

// MIDI CC event
export interface MidiCC {
  id: string;
  controller: number;
  value: number;
  time: number;
  channel: number;
}

// MIDI region
export interface MidiRegion {
  id: string;
  trackId: string;
  startTime: number;
  duration: number;
  name: string;
  color: string;
  notes: MidiNote[];
  ccEvents: MidiCC[];
  muted: boolean;
  locked: boolean;
  looped: boolean;
  loopLength: number;
}

// Automation point
export interface AutomationPoint {
  time: number;
  value: number;
  curve: 'linear' | 'bezier' | 'step';
  bezierControlX?: number;
  bezierControlY?: number;
}

// Automation lane
export interface AutomationLane {
  id: string;
  trackId: string;
  parameter: string;
  points: AutomationPoint[];
  enabled: boolean;
  mode: 'read' | 'write' | 'touch' | 'latch' | 'trim';
}

// Send configuration
export interface Send {
  id: string;
  targetId: string;
  level: number;
  pan: number;
  preFader: boolean;
  muted: boolean;
}

// Insert plugin
export interface Insert {
  id: string;
  name: string;
  type: string;
  bypassed: boolean;
  parameters: Record<string, number>;
  preset: string | null;
}

// Track interface
export interface Track {
  id: string;
  name: string;
  type: TrackType;
  color: string;
  height: number;
  visible: boolean;
  expanded: boolean;

  // Audio settings
  volume: number;
  pan: number;
  muted: boolean;
  solo: boolean;
  armed: boolean;
  inputMonitoring: boolean;
  phaseInvert: boolean;

  // Routing
  input: string;
  output: string;
  sends: Send[];
  inserts: Insert[];

  // Regions
  audioRegions: AudioRegion[];
  midiRegions: MidiRegion[];

  // Automation
  automationLanes: AutomationLane[];
  automationVisible: boolean;

  // Grouping
  parentId: string | null;
  childIds: string[];
  groupId: string | null;
  vcaId: string | null;

  // Channel strip
  eqEnabled: boolean;
  eqBands: { frequency: number; gain: number; q: number; type: string }[];
  compressorEnabled: boolean;
  compressorSettings: {
    threshold: number;
    ratio: number;
    attack: number;
    release: number;
    makeupGain: number;
  };
  gateEnabled: boolean;
  gateSettings: {
    threshold: number;
    attack: number;
    hold: number;
    release: number;
    range: number;
  };
  hpfEnabled: boolean;
  hpfFrequency: number;
  lpfEnabled: boolean;
  lpfFrequency: number;

  // Delay compensation
  delayCompensation: number;
  manualDelay: number;
}

// Track group
export interface TrackGroup {
  id: string;
  name: string;
  trackIds: string[];
  linkedParams: ('volume' | 'pan' | 'mute' | 'solo' | 'edit')[];
}

// Clipboard for copy/paste
interface Clipboard {
  type: 'audio' | 'midi' | 'mixed';
  audioRegions: AudioRegion[];
  midiRegions: MidiRegion[];
  notes: MidiNote[];
  sourceTime: number;
}

class TrackManager {
  private tracks: Map<string, Track> = new Map();
  private groups: Map<string, TrackGroup> = new Map();
  private clipboard: Clipboard | null = null;
  private undoStack: { action: string; data: unknown }[] = [];
  private redoStack: { action: string; data: unknown }[] = [];
  private listeners: Map<string, Set<Function>> = new Map();
  private audioNodes: Map<string, {
    gain: GainNode;
    pan: StereoPannerNode;
    analyser: AnalyserNode;
  }> = new Map();

  constructor() {
    this.createDefaultTracks();
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

  private createDefaultTracks(): void {
    // Create master track
    this.createTrack('master', 'Master');
  }

  // Track CRUD operations
  createTrack(type: TrackType, name?: string): Track {
    const id = crypto.randomUUID();
    const trackCount = Array.from(this.tracks.values()).filter(t => t.type === type).length + 1;

    const track: Track = {
      id,
      name: name || `${type.charAt(0).toUpperCase() + type.slice(1)} ${trackCount}`,
      type,
      color: this.getDefaultColor(type),
      height: 80,
      visible: true,
      expanded: true,

      volume: 1,
      pan: 0,
      muted: false,
      solo: false,
      armed: false,
      inputMonitoring: false,
      phaseInvert: false,

      input: 'default',
      output: type === 'master' ? 'hardware' : 'master',
      sends: [],
      inserts: [],

      audioRegions: [],
      midiRegions: [],

      automationLanes: [],
      automationVisible: false,

      parentId: null,
      childIds: [],
      groupId: null,
      vcaId: null,

      eqEnabled: false,
      eqBands: [
        { frequency: 80, gain: 0, q: 1, type: 'lowshelf' },
        { frequency: 250, gain: 0, q: 1, type: 'peaking' },
        { frequency: 1000, gain: 0, q: 1, type: 'peaking' },
        { frequency: 4000, gain: 0, q: 1, type: 'peaking' },
        { frequency: 12000, gain: 0, q: 1, type: 'highshelf' },
      ],
      compressorEnabled: false,
      compressorSettings: {
        threshold: -24,
        ratio: 4,
        attack: 10,
        release: 100,
        makeupGain: 0,
      },
      gateEnabled: false,
      gateSettings: {
        threshold: -40,
        attack: 0.5,
        hold: 10,
        release: 50,
        range: -80,
      },
      hpfEnabled: false,
      hpfFrequency: 80,
      lpfEnabled: false,
      lpfFrequency: 18000,
      delayCompensation: 0,
      manualDelay: 0,
    };

    this.tracks.set(id, track);
    this.createAudioNodesForTrack(track);
    this.emit('trackCreated', track);
    return track;
  }

  private createAudioNodesForTrack(track: Track): void {
    const context = audioEngine.getContext();
    if (!context) return;

    const gain = context.createGain();
    const pan = context.createStereoPanner();
    const analyser = context.createAnalyser();

    analyser.fftSize = 256;

    gain.connect(pan);
    pan.connect(analyser);

    // Connect to output
    if (track.type !== 'master') {
      const masterNodes = this.audioNodes.get(
        Array.from(this.tracks.values()).find(t => t.type === 'master')?.id || ''
      );
      if (masterNodes) {
        analyser.connect(masterNodes.gain);
      }
    } else {
      const engineAnalyser = audioEngine.getAnalyser();
      if (engineAnalyser) {
        // Master connects through the engine
        analyser.connect(context.destination);
      }
    }

    this.audioNodes.set(track.id, { gain, pan, analyser });
  }

  private getDefaultColor(type: TrackType): string {
    const colors: Record<TrackType, string> = {
      audio: '#6366f1',
      midi: '#22c55e',
      instrument: '#f59e0b',
      aux: '#8b5cf6',
      master: '#ef4444',
      folder: '#64748b',
      vca: '#06b6d4',
      marker: '#ec4899',
      tempo: '#f97316',
      chord: '#14b8a6',
      video: '#a855f7',
    };
    return colors[type];
  }

  deleteTrack(id: string): void {
    const track = this.tracks.get(id);
    if (!track || track.type === 'master') return;

    this.pushUndo('deleteTrack', track);
    this.tracks.delete(id);
    this.audioNodes.get(id)?.gain.disconnect();
    this.audioNodes.delete(id);
    this.emit('trackDeleted', id);
  }

  duplicateTrack(id: string): Track | null {
    const source = this.tracks.get(id);
    if (!source) return null;

    const newTrack = this.createTrack(source.type, `${source.name} Copy`);
    Object.assign(newTrack, {
      ...source,
      id: newTrack.id,
      name: `${source.name} Copy`,
      audioRegions: source.audioRegions.map(r => ({ ...r, id: crypto.randomUUID(), trackId: newTrack.id })),
      midiRegions: source.midiRegions.map(r => ({ ...r, id: crypto.randomUUID(), trackId: newTrack.id })),
    });

    this.tracks.set(newTrack.id, newTrack);
    this.emit('trackDuplicated', newTrack);
    return newTrack;
  }

  getTrack(id: string): Track | undefined {
    return this.tracks.get(id);
  }

  getAllTracks(): Track[] {
    return Array.from(this.tracks.values());
  }

  getTracksByType(type: TrackType): Track[] {
    return Array.from(this.tracks.values()).filter(t => t.type === type);
  }

  // Track parameter updates (Items 267-290)
  setVolume(trackId: string, volume: number): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    track.volume = Math.max(0, Math.min(2, volume));
    const nodes = this.audioNodes.get(trackId);
    if (nodes) {
      nodes.gain.gain.value = track.volume;
    }

    this.handleGroupLink(trackId, 'volume', volume);
    this.emit('volumeChange', { trackId, volume });
  }

  setPan(trackId: string, pan: number): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    track.pan = Math.max(-1, Math.min(1, pan));
    const nodes = this.audioNodes.get(trackId);
    if (nodes) {
      nodes.pan.pan.value = track.pan;
    }

    this.handleGroupLink(trackId, 'pan', pan);
    this.emit('panChange', { trackId, pan });
  }

  setMute(trackId: string, muted: boolean): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    track.muted = muted;
    this.handleGroupLink(trackId, 'mute', muted);
    this.updateMuteState(trackId);
    this.emit('muteChange', { trackId, muted });
  }

  setSolo(trackId: string, solo: boolean, exclusive: boolean = false): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    if (exclusive && solo) {
      // X-OR solo - unsolo all others
      this.tracks.forEach((t, id) => {
        if (id !== trackId) {
          t.solo = false;
        }
      });
    }

    track.solo = solo;
    this.handleGroupLink(trackId, 'solo', solo);
    this.updateSoloState();
    this.emit('soloChange', { trackId, solo });
  }

  setArmed(trackId: string, armed: boolean): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    track.armed = armed;
    this.emit('armChange', { trackId, armed });
  }

  setInputMonitoring(trackId: string, enabled: boolean): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    track.inputMonitoring = enabled;
    this.emit('inputMonitoringChange', { trackId, enabled });
  }

  private updateMuteState(trackId: string): void {
    const track = this.tracks.get(trackId);
    const nodes = this.audioNodes.get(trackId);
    if (!track || !nodes) return;

    // Check if any track is soloed
    const hasSolo = Array.from(this.tracks.values()).some(t => t.solo);

    if (hasSolo) {
      // If soloed tracks exist, mute non-soloed tracks
      nodes.gain.gain.value = track.solo ? track.volume : 0;
    } else {
      // Normal mute behavior
      nodes.gain.gain.value = track.muted ? 0 : track.volume;
    }
  }

  private updateSoloState(): void {
    this.tracks.forEach((_, id) => this.updateMuteState(id));
  }

  // Routing (Items 291-314)
  setOutput(trackId: string, output: string): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    track.output = output;
    // Reconnect audio nodes
    this.emit('outputChange', { trackId, output });
  }

  setInput(trackId: string, input: string): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    track.input = input;
    this.emit('inputChange', { trackId, input });
  }

  addSend(trackId: string, targetId: string, preFader: boolean = false): Send {
    const track = this.tracks.get(trackId);
    if (!track) throw new Error('Track not found');

    const send: Send = {
      id: crypto.randomUUID(),
      targetId,
      level: 0,
      pan: 0,
      preFader,
      muted: false,
    };

    track.sends.push(send);
    this.emit('sendAdded', { trackId, send });
    return send;
  }

  removeSend(trackId: string, sendId: string): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    track.sends = track.sends.filter(s => s.id !== sendId);
    this.emit('sendRemoved', { trackId, sendId });
  }

  updateSend(trackId: string, sendId: string, updates: Partial<Omit<Send, 'id'>>): void {
    const track = this.tracks.get(trackId);
    const send = track?.sends.find(s => s.id === sendId);
    if (!send) return;

    Object.assign(send, updates);
    this.emit('sendUpdated', { trackId, send });
  }

  // Insert plugins (Items 287, 360-380)
  addInsert(trackId: string, pluginType: string, position?: number): Insert {
    const track = this.tracks.get(trackId);
    if (!track) throw new Error('Track not found');

    const insert: Insert = {
      id: crypto.randomUUID(),
      name: pluginType,
      type: pluginType,
      bypassed: false,
      parameters: {},
      preset: null,
    };

    if (position !== undefined) {
      track.inserts.splice(position, 0, insert);
    } else {
      track.inserts.push(insert);
    }

    this.emit('insertAdded', { trackId, insert, position });
    return insert;
  }

  removeInsert(trackId: string, insertId: string): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    track.inserts = track.inserts.filter(i => i.id !== insertId);
    this.emit('insertRemoved', { trackId, insertId });
  }

  bypassInsert(trackId: string, insertId: string, bypassed: boolean): void {
    const track = this.tracks.get(trackId);
    const insert = track?.inserts.find(i => i.id === insertId);
    if (!insert) return;

    insert.bypassed = bypassed;
    this.emit('insertBypassed', { trackId, insertId, bypassed });
  }

  // Track groups (Items 301-304)
  createGroup(name: string, trackIds: string[]): TrackGroup {
    const group: TrackGroup = {
      id: crypto.randomUUID(),
      name,
      trackIds,
      linkedParams: ['volume', 'mute', 'solo'],
    };

    trackIds.forEach(id => {
      const track = this.tracks.get(id);
      if (track) track.groupId = group.id;
    });

    this.groups.set(group.id, group);
    this.emit('groupCreated', group);
    return group;
  }

  deleteGroup(groupId: string): void {
    const group = this.groups.get(groupId);
    if (!group) return;

    group.trackIds.forEach(id => {
      const track = this.tracks.get(id);
      if (track) track.groupId = null;
    });

    this.groups.delete(groupId);
    this.emit('groupDeleted', groupId);
  }

  private handleGroupLink(trackId: string, param: 'volume' | 'pan' | 'mute' | 'solo', value: unknown): void {
    const track = this.tracks.get(trackId);
    if (!track?.groupId) return;

    const group = this.groups.get(track.groupId);
    if (!group?.linkedParams.includes(param)) return;

    group.trackIds.forEach(id => {
      if (id === trackId) return;
      const linkedTrack = this.tracks.get(id);
      if (!linkedTrack) return;

      switch (param) {
        case 'volume':
          linkedTrack.volume = value as number;
          break;
        case 'pan':
          linkedTrack.pan = value as number;
          break;
        case 'mute':
          linkedTrack.muted = value as boolean;
          break;
        case 'solo':
          linkedTrack.solo = value as boolean;
          break;
      }
    });
  }

  // Audio regions (Items 108-182)
  addAudioRegion(trackId: string, region: Omit<AudioRegion, 'id' | 'trackId'>): AudioRegion {
    const track = this.tracks.get(trackId);
    if (!track || track.type !== 'audio') throw new Error('Invalid track');

    const newRegion: AudioRegion = {
      ...region,
      id: crypto.randomUUID(),
      trackId,
    };

    this.pushUndo('addAudioRegion', { trackId, region: newRegion });
    track.audioRegions.push(newRegion);
    track.audioRegions.sort((a, b) => a.startTime - b.startTime);
    this.emit('regionAdded', newRegion);
    return newRegion;
  }

  removeAudioRegion(trackId: string, regionId: string): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    const region = track.audioRegions.find(r => r.id === regionId);
    if (region) {
      this.pushUndo('removeAudioRegion', { trackId, region });
      track.audioRegions = track.audioRegions.filter(r => r.id !== regionId);
      this.emit('regionRemoved', { trackId, regionId });
    }
  }

  moveRegion(trackId: string, regionId: string, newStartTime: number): void {
    const track = this.tracks.get(trackId);
    const region = track?.audioRegions.find(r => r.id === regionId);
    if (!region) return;

    this.pushUndo('moveRegion', { trackId, regionId, oldTime: region.startTime });
    region.startTime = Math.max(0, newStartTime);
    this.emit('regionMoved', { trackId, regionId, startTime: region.startTime });
  }

  resizeRegion(trackId: string, regionId: string, duration: number, fromStart: boolean = false): void {
    const track = this.tracks.get(trackId);
    const region = track?.audioRegions.find(r => r.id === regionId);
    if (!region) return;

    this.pushUndo('resizeRegion', { trackId, regionId, oldDuration: region.duration });
    region.duration = Math.max(0.01, duration);
    if (fromStart) {
      region.startTime = region.startTime + (region.duration - duration);
    }
    this.emit('regionResized', { trackId, regionId, duration: region.duration });
  }

  splitRegion(trackId: string, regionId: string, splitTime: number): AudioRegion[] {
    const track = this.tracks.get(trackId);
    const region = track?.audioRegions.find(r => r.id === regionId);
    if (!region || splitTime <= region.startTime || splitTime >= region.startTime + region.duration) {
      return [];
    }

    this.pushUndo('splitRegion', { trackId, region: { ...region } });

    const splitPoint = splitTime - region.startTime;
    const secondRegion: AudioRegion = {
      ...region,
      id: crypto.randomUUID(),
      startTime: splitTime,
      duration: region.duration - splitPoint,
      offset: region.offset + splitPoint,
    };

    region.duration = splitPoint;
    track!.audioRegions.push(secondRegion);
    track!.audioRegions.sort((a, b) => a.startTime - b.startTime);

    this.emit('regionSplit', { trackId, originalId: regionId, newRegion: secondRegion });
    return [region, secondRegion];
  }

  // MIDI regions (Items 183-266)
  addMidiRegion(trackId: string, region: Omit<MidiRegion, 'id' | 'trackId'>): MidiRegion {
    const track = this.tracks.get(trackId);
    if (!track || (track.type !== 'midi' && track.type !== 'instrument')) {
      throw new Error('Invalid track');
    }

    const newRegion: MidiRegion = {
      ...region,
      id: crypto.randomUUID(),
      trackId,
    };

    this.pushUndo('addMidiRegion', { trackId, region: newRegion });
    track.midiRegions.push(newRegion);
    track.midiRegions.sort((a, b) => a.startTime - b.startTime);
    this.emit('midiRegionAdded', newRegion);
    return newRegion;
  }

  addNote(trackId: string, regionId: string, note: Omit<MidiNote, 'id'>): MidiNote {
    const track = this.tracks.get(trackId);
    const region = track?.midiRegions.find(r => r.id === regionId);
    if (!region) throw new Error('Region not found');

    const newNote: MidiNote = {
      ...note,
      id: crypto.randomUUID(),
    };

    this.pushUndo('addNote', { trackId, regionId, note: newNote });
    region.notes.push(newNote);
    region.notes.sort((a, b) => a.startTime - b.startTime);
    this.emit('noteAdded', { trackId, regionId, note: newNote });
    return newNote;
  }

  removeNote(trackId: string, regionId: string, noteId: string): void {
    const track = this.tracks.get(trackId);
    const region = track?.midiRegions.find(r => r.id === regionId);
    if (!region) return;

    const note = region.notes.find(n => n.id === noteId);
    if (note) {
      this.pushUndo('removeNote', { trackId, regionId, note });
      region.notes = region.notes.filter(n => n.id !== noteId);
      this.emit('noteRemoved', { trackId, regionId, noteId });
    }
  }

  updateNote(trackId: string, regionId: string, noteId: string, updates: Partial<Omit<MidiNote, 'id'>>): void {
    const track = this.tracks.get(trackId);
    const region = track?.midiRegions.find(r => r.id === regionId);
    const note = region?.notes.find(n => n.id === noteId);
    if (!note) return;

    this.pushUndo('updateNote', { trackId, regionId, noteId, oldNote: { ...note } });
    Object.assign(note, updates);
    this.emit('noteUpdated', { trackId, regionId, note });
  }

  // MIDI quantize (Items 202-227)
  quantizeNotes(trackId: string, regionId: string, noteIds: string[], gridValue: number, strength: number = 1): void {
    const track = this.tracks.get(trackId);
    const region = track?.midiRegions.find(r => r.id === regionId);
    if (!region) return;

    const notes = region.notes.filter(n => noteIds.includes(n.id));
    this.pushUndo('quantize', { trackId, regionId, notes: notes.map(n => ({ ...n })) });

    notes.forEach(note => {
      const quantizedTime = Math.round(note.startTime / gridValue) * gridValue;
      note.startTime = note.startTime + (quantizedTime - note.startTime) * strength;
    });

    this.emit('notesQuantized', { trackId, regionId, noteIds });
  }

  transposeNotes(trackId: string, regionId: string, noteIds: string[], semitones: number): void {
    const track = this.tracks.get(trackId);
    const region = track?.midiRegions.find(r => r.id === regionId);
    if (!region) return;

    const notes = region.notes.filter(n => noteIds.includes(n.id));
    this.pushUndo('transpose', { trackId, regionId, notes: notes.map(n => ({ ...n })) });

    notes.forEach(note => {
      note.pitch = Math.max(0, Math.min(127, note.pitch + semitones));
    });

    this.emit('notesTransposed', { trackId, regionId, noteIds, semitones });
  }

  // Automation (Items 613-654)
  addAutomationLane(trackId: string, parameter: string): AutomationLane {
    const track = this.tracks.get(trackId);
    if (!track) throw new Error('Track not found');

    const lane: AutomationLane = {
      id: crypto.randomUUID(),
      trackId,
      parameter,
      points: [{ time: 0, value: 0.5, curve: 'linear' }],
      enabled: true,
      mode: 'read',
    };

    track.automationLanes.push(lane);
    this.emit('automationLaneAdded', lane);
    return lane;
  }

  addAutomationPoint(trackId: string, laneId: string, point: AutomationPoint): void {
    const track = this.tracks.get(trackId);
    const lane = track?.automationLanes.find(l => l.id === laneId);
    if (!lane) return;

    lane.points.push(point);
    lane.points.sort((a, b) => a.time - b.time);
    this.emit('automationPointAdded', { trackId, laneId, point });
  }

  getAutomationValue(trackId: string, parameter: string, time: number): number {
    const track = this.tracks.get(trackId);
    const lane = track?.automationLanes.find(l => l.parameter === parameter && l.enabled);
    if (!lane || lane.points.length === 0) return 0.5;

    // Find surrounding points
    let prevPoint = lane.points[0];
    let nextPoint = lane.points[lane.points.length - 1];

    for (let i = 0; i < lane.points.length; i++) {
      if (lane.points[i].time <= time) {
        prevPoint = lane.points[i];
      }
      if (lane.points[i].time >= time && i > 0) {
        nextPoint = lane.points[i];
        break;
      }
    }

    if (prevPoint === nextPoint || prevPoint.curve === 'step') {
      return prevPoint.value;
    }

    // Linear interpolation
    const t = (time - prevPoint.time) / (nextPoint.time - prevPoint.time);
    return prevPoint.value + (nextPoint.value - prevPoint.value) * t;
  }

  // Clipboard operations (Items 108-115, 192-193)
  copyRegions(trackId: string, regionIds: string[]): void {
    const track = this.tracks.get(trackId);
    if (!track) return;

    const currentTime = audioEngine.getTransport().currentTime;

    this.clipboard = {
      type: track.type === 'audio' ? 'audio' : 'midi',
      audioRegions: track.audioRegions.filter(r => regionIds.includes(r.id)).map(r => ({ ...r })),
      midiRegions: track.midiRegions.filter(r => regionIds.includes(r.id)).map(r => ({ ...r })),
      notes: [],
      sourceTime: currentTime,
    };

    this.emit('copied', { type: this.clipboard.type, count: regionIds.length });
  }

  paste(trackId: string, time?: number): void {
    if (!this.clipboard) return;

    const track = this.tracks.get(trackId);
    if (!track) return;

    const pasteTime = time ?? audioEngine.getTransport().currentTime;
    const offset = pasteTime - this.clipboard.sourceTime;

    if (this.clipboard.type === 'audio' && track.type === 'audio') {
      this.clipboard.audioRegions.forEach(region => {
        this.addAudioRegion(trackId, {
          ...region,
          startTime: region.startTime + offset,
        });
      });
    } else if (this.clipboard.type === 'midi' && (track.type === 'midi' || track.type === 'instrument')) {
      this.clipboard.midiRegions.forEach(region => {
        this.addMidiRegion(trackId, {
          ...region,
          startTime: region.startTime + offset,
        });
      });
    }

    this.emit('pasted', { trackId, time: pasteTime });
  }

  // Undo/Redo
  private pushUndo(action: string, data: unknown): void {
    this.undoStack.push({ action, data });
    this.redoStack = [];
    if (this.undoStack.length > 100) {
      this.undoStack.shift();
    }
  }

  undo(): void {
    const item = this.undoStack.pop();
    if (!item) return;

    // Would implement undo logic here
    this.redoStack.push(item);
    this.emit('undo', item);
  }

  redo(): void {
    const item = this.redoStack.pop();
    if (!item) return;

    // Would implement redo logic here
    this.undoStack.push(item);
    this.emit('redo', item);
  }

  // Get track meter readings
  getTrackMeter(trackId: string): { left: number; right: number } {
    const nodes = this.audioNodes.get(trackId);
    if (!nodes) return { left: -Infinity, right: -Infinity };

    const dataArray = new Float32Array(nodes.analyser.fftSize);
    nodes.analyser.getFloatTimeDomainData(dataArray);

    let peak = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const abs = Math.abs(dataArray[i]);
      if (abs > peak) peak = abs;
    }

    const db = 20 * Math.log10(peak || 0.0001);
    return { left: db, right: db };
  }

  // Cleanup
  destroy(): void {
    this.audioNodes.forEach(nodes => {
      nodes.gain.disconnect();
    });
    this.audioNodes.clear();
    this.tracks.clear();
    this.groups.clear();
    this.listeners.clear();
  }
}

// Singleton instance
export const trackManager = new TrackManager();
export default TrackManager;
