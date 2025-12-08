// MIDIEngine - Features 183-200: Basic MIDI Editing

export interface MIDINote {
  id: string;
  pitch: number; // MIDI note number (0-127)
  velocity: number; // 0-127
  startTime: number; // in beats
  duration: number; // in beats
  channel: number; // 0-15
  selected: boolean;
}

export interface MIDIClip {
  id: string;
  trackId: string;
  name: string;
  startTime: number; // in beats
  endTime: number; // in beats
  notes: MIDINote[];
  color: string;
  muted: boolean;
  soloed: boolean;
  looped: boolean;
  loopLength: number; // in beats
}

export interface MIDITrack {
  id: string;
  name: string;
  channel: number;
  instrument: string;
  volume: number; // 0-127
  pan: number; // -64 to +63
  clips: MIDIClip[];
  muted: boolean;
  soloed: boolean;
  recordEnabled: boolean;
}

export interface MIDIEditingEngineState {
  // Basic MIDI Editing (183-201)
  tracks: MIDITrack[];
  selectedTracks: Set<string>;
  selectedClips: Set<string>;
  selectedNotes: Set<string>;
  clipboard: { notes: MIDINote[]; clips: MIDIClip[] };
  editHistory: Array<{ type: string; data: any; timestamp: number }>;
  currentHistoryIndex: number;
  
  // Piano Roll
  pianoRollVisible: boolean;
  pianoRollZoom: number; // 1.0 = 100%
  pianoRollTimeRange: [number, number]; // [start, end] in beats
  pianoRollNoteRange: [number, number]; // [low, high] MIDI notes
  
  // Quantization
  quantizeEnabled: boolean;
  quantizeGrid: number; // in beats (1/4, 1/8, 1/16, etc.)
  quantizeStrength: number; // 0.0 to 1.0
  quantizeSwing: number; // 0.0 to 1.0
  
  // Note Editing
  noteSnap: boolean;
  snapGrid: number; // in beats
  noteLength: number; // default note length in beats
  velocityRange: [number, number]; // [min, max] 0-127
}

export class MIDIEngine {
  private state: MIDIEditingEngineState;
  // private maxHistorySize: number = 100; // Reserved for future use

  constructor(initialState?: Partial<MIDIEditingEngineState>) {
    this.state = {
      tracks: [],
      selectedTracks: new Set(),
      selectedClips: new Set(),
      selectedNotes: new Set(),
      clipboard: { notes: [], clips: [] },
      editHistory: [],
      currentHistoryIndex: -1,
      pianoRollVisible: true,
      pianoRollZoom: 1.0,
      pianoRollTimeRange: [0, 16], // 16 beats default
      pianoRollNoteRange: [36, 96], // C2 to C7
      quantizeEnabled: false,
      quantizeGrid: 1 / 16, // 16th notes
      quantizeStrength: 1.0,
      quantizeSwing: 0.0,
      noteSnap: true,
      snapGrid: 1 / 16,
      noteLength: 1 / 4, // quarter note
      velocityRange: [0, 127],
      ...initialState,
    };
  }

  // ===== BASIC MIDI EDITING (Features 183-201) =====

  // Feature 183: Create MIDI Track
  createTrack(name: string, channel: number = 0): MIDITrack {
    const track: MIDITrack = {
      id: `track-${Date.now()}`,
      name,
      channel,
      instrument: 'Acoustic Grand Piano',
      volume: 100,
      pan: 0,
      clips: [],
      muted: false,
      soloed: false,
      recordEnabled: false,
    };
    this.state.tracks.push(track);
    return track;
  }

  // Feature 184: Delete MIDI Track
  deleteTrack(trackId: string): void {
    this.state.tracks = this.state.tracks.filter(t => t.id !== trackId);
    this.state.selectedTracks.delete(trackId);
  }

  // Feature 185: Create MIDI Clip
  createClip(trackId: string, name: string, startTime: number, length: number): MIDIClip {
    const track = this.state.tracks.find(t => t.id === trackId);
    if (!track) throw new Error('Track not found');

    const clip: MIDIClip = {
      id: `clip-${Date.now()}`,
      trackId,
      name,
      startTime,
      endTime: startTime + length,
      notes: [],
      color: '#6366f1',
      muted: false,
      soloed: false,
      looped: false,
      loopLength: length,
    };
    track.clips.push(clip);
    return clip;
  }

  // Feature 186: Delete MIDI Clip
  deleteClip(clipId: string): void {
    this.state.tracks.forEach(track => {
      track.clips = track.clips.filter(c => c.id !== clipId);
    });
    this.state.selectedClips.delete(clipId);
  }

  // Feature 187: Add MIDI Note
  addNote(clipId: string, pitch: number, startTime: number, duration: number, velocity: number = 100): MIDINote {
    const clip = this.findClip(clipId);
    if (!clip) throw new Error('Clip not found');

    const note: MIDINote = {
      id: `note-${Date.now()}-${pitch}`,
      pitch,
      velocity,
      startTime,
      duration,
      channel: 0,
      selected: false,
    };
    clip.notes.push(note);
    return note;
  }

  // Feature 188: Delete MIDI Note
  deleteNote(noteId: string): void {
    this.state.tracks.forEach(track => {
      track.clips.forEach(clip => {
        clip.notes = clip.notes.filter(n => n.id !== noteId);
      });
    });
    this.state.selectedNotes.delete(noteId);
  }

  // Feature 189: Select Note
  selectNote(noteId: string, multiSelect: boolean = false): void {
    if (!multiSelect) {
      this.deselectAllNotes();
    }
    this.state.selectedNotes.add(noteId);
    this.updateNoteSelection(noteId, true);
  }

  // Feature 190: Deselect Note
  deselectNote(noteId: string): void {
    this.state.selectedNotes.delete(noteId);
    this.updateNoteSelection(noteId, false);
  }

  // Feature 191: Select All Notes
  selectAllNotes(): void {
    this.state.tracks.forEach(track => {
      track.clips.forEach(clip => {
        clip.notes.forEach(note => {
          this.state.selectedNotes.add(note.id);
          note.selected = true;
        });
      });
    });
  }

  // Feature 192: Deselect All Notes
  deselectAllNotes(): void {
    this.state.selectedNotes.clear();
    this.state.tracks.forEach(track => {
      track.clips.forEach(clip => {
        clip.notes.forEach(note => {
          note.selected = false;
        });
      });
    });
  }

  // Feature 193: Move Note
  moveNote(noteId: string, newStartTime: number, newPitch?: number): void {
    const note = this.findNote(noteId);
    if (!note) return;

    note.startTime = this.snapToGrid(newStartTime);
    if (newPitch !== undefined) {
      note.pitch = Math.max(0, Math.min(127, newPitch));
    }
  }

  // Feature 194: Resize Note
  resizeNote(noteId: string, newDuration: number): void {
    const note = this.findNote(noteId);
    if (note) {
      note.duration = Math.max(this.state.snapGrid, this.snapToGrid(newDuration));
    }
  }

  // Feature 195: Change Note Velocity
  setNoteVelocity(noteId: string, velocity: number): void {
    const note = this.findNote(noteId);
    if (note) {
      note.velocity = Math.max(0, Math.min(127, velocity));
    }
  }

  // Feature 196: Change Note Pitch
  setNotePitch(noteId: string, pitch: number): void {
    const note = this.findNote(noteId);
    if (note) {
      note.pitch = Math.max(0, Math.min(127, pitch));
    }
  }

  // Feature 197: Duplicate Note
  duplicateNote(noteId: string, offset: number = 0): MIDINote {
    const note = this.findNote(noteId);
    if (!note) throw new Error('Note not found');

    const clip = this.findClipByNote(noteId);
    if (!clip) throw new Error('Clip not found');

    const newNote: MIDINote = {
      ...note,
      id: `note-${Date.now()}-${note.pitch}`,
      startTime: note.startTime + offset,
      selected: false,
    };
    clip.notes.push(newNote);
    return newNote;
  }

  // Feature 198: Quantize Notes
  quantizeNotes(noteIds: string[]): void {
    noteIds.forEach(noteId => {
      const note = this.findNote(noteId);
      if (note) {
        const quantizedTime = Math.round(note.startTime / this.state.quantizeGrid) * this.state.quantizeGrid;
        const difference = quantizedTime - note.startTime;
        note.startTime = note.startTime + (difference * this.state.quantizeStrength);
      }
    });
  }

  // Feature 199: Transpose Notes
  transposeNotes(noteIds: string[], semitones: number): void {
    noteIds.forEach(noteId => {
      const note = this.findNote(noteId);
      if (note) {
        note.pitch = Math.max(0, Math.min(127, note.pitch + semitones));
      }
    });
  }

  // Feature 200: Copy Notes
  copyNotes(noteIds: string[]): void {
    const notes = noteIds.map(id => this.findNote(id)).filter(n => n !== null) as MIDINote[];
    this.state.clipboard.notes = notes.map(n => ({ ...n }));
  }

  // Feature 201: Paste Notes
  pasteNotes(clipId: string, position: number): MIDINote[] {
    const clip = this.findClip(clipId);
    if (!clip) throw new Error('Clip not found');

    const newNotes: MIDINote[] = [];
    this.state.clipboard.notes.forEach((note, index) => {
      const newNote: MIDINote = {
        ...note,
        id: `note-${Date.now()}-${index}`,
        startTime: position + (note.startTime - this.getClipboardStartTime()),
        selected: false,
      };
      clip.notes.push(newNote);
      newNotes.push(newNote);
    });
    return newNotes;
  }

  // ===== UTILITY METHODS =====

  private findClip(clipId: string): MIDIClip | null {
    for (const track of this.state.tracks) {
      const clip = track.clips.find(c => c.id === clipId);
      if (clip) return clip;
    }
    return null;
  }

  private findNote(noteId: string): MIDINote | null {
    for (const track of this.state.tracks) {
      for (const clip of track.clips) {
        const note = clip.notes.find(n => n.id === noteId);
        if (note) return note;
      }
    }
    return null;
  }

  private findClipByNote(noteId: string): MIDIClip | null {
    for (const track of this.state.tracks) {
      for (const clip of track.clips) {
        if (clip.notes.find(n => n.id === noteId)) {
          return clip;
        }
      }
    }
    return null;
  }

  private updateNoteSelection(noteId: string, selected: boolean): void {
    const note = this.findNote(noteId);
    if (note) {
      note.selected = selected;
    }
  }

  private snapToGrid(time: number): number {
    if (!this.state.noteSnap) return time;
    return Math.round(time / this.state.snapGrid) * this.state.snapGrid;
  }

  private getClipboardStartTime(): number {
    if (this.state.clipboard.notes.length === 0) return 0;
    return Math.min(...this.state.clipboard.notes.map(n => n.startTime));
  }

  // Getters
  getState(): MIDIEditingEngineState {
    return { ...this.state };
  }

  getTracks(): MIDITrack[] {
    return [...this.state.tracks];
  }

  getSelectedNotes(): string[] {
    return Array.from(this.state.selectedNotes);
  }

  getSelectedClips(): string[] {
    return Array.from(this.state.selectedClips);
  }

  // Setters
  setQuantizeGrid(grid: number): void {
    this.state.quantizeGrid = grid;
  }

  setQuantizeStrength(strength: number): void {
    this.state.quantizeStrength = Math.max(0, Math.min(1, strength));
  }

  setSnapGrid(grid: number): void {
    this.state.snapGrid = grid;
  }

  setNoteSnap(enabled: boolean): void {
    this.state.noteSnap = enabled;
  }

  // ===== ADVANCED MIDI EDITING (Features 201-227) =====

  // Feature 201: Paste Notes (already implemented above)
  
  // Feature 202: MIDI CC Editing
  setCCValue(_trackId: string, _ccNumber: number, _time: number, _value: number): void {
    // CC editing would be implemented here
  }

  // Feature 203: MIDI Program Change
  setProgramChange(_trackId: string, _program: number, _time: number): void {
    // Program change implementation
  }

  // Feature 204: MIDI Aftertouch
  setAftertouch(_trackId: string, _time: number, _pressure: number): void {
    // Aftertouch implementation
  }

  // Feature 205: MIDI Pitch Bend
  setPitchBend(_trackId: string, _time: number, _bend: number): void {
    // Pitch bend implementation
  }

  // Features 206-227: Additional advanced MIDI editing
  // (MIDI filters, MIDI transforms, etc.)

  // ===== MIDI TOOLS (Features 248-266) =====

  // Feature 248: MIDI Arpeggiator
  createArpeggiator(_trackId: string): void {
    // Arpeggiator implementation
  }

  // Feature 249: MIDI Chord Generator
  generateChord(_root: number, _type: string): number[] {
    // Chord generation logic
    return [];
  }

  // Feature 250: MIDI Scale Quantize
  scaleQuantize(noteIds: string[], scale: number[]): void {
    noteIds.forEach(noteId => {
      const note = this.findNote(noteId);
      if (note) {
        const noteInScale = scale.find(s => (note.pitch % 12) === (s % 12));
        if (noteInScale !== undefined) {
          note.pitch = Math.floor(note.pitch / 12) * 12 + (noteInScale % 12);
        }
      }
    });
  }

  // Features 251-266: Additional MIDI tools
  // (MIDI humanize, MIDI velocity tools, etc.)
}
