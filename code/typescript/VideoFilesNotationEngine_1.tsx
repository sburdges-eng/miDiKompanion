// VideoFilesNotationEngine - Features 757-961
// Video & Sync (757-808), File Management (809-871),
// Notation & Scoring (872-911), Browser & Library (912-941),
// Collaboration (942-961)

// ===== VIDEO & SYNC TYPES (Features 757-808) =====
export interface VideoTrack {
  id: string;
  name: string;
  path: string;
  duration: number;
  frameRate: number;
  width: number;
  height: number;
  position: number; // Start time in seconds
  volume: number;
  muted: boolean;
  opacity: number;
  visible: boolean;
}

export interface TimecodeSettings {
  format: 'smpte' | 'samples' | 'bars-beats' | 'seconds' | 'frames';
  frameRate: 24 | 25 | 29.97 | 30 | 48 | 50 | 59.94 | 60;
  dropFrame: boolean;
  offset: number; // in frames
  syncSource: 'internal' | 'external' | 'mtc' | 'ltc';
}

export interface SurroundFormat {
  name: string;
  channels: number;
  layout: string[];
  lfe: boolean;
}

// ===== FILE MANAGEMENT TYPES (Features 809-871) =====
export interface ProjectFile {
  id: string;
  name: string;
  path: string;
  type: 'audio' | 'midi' | 'video' | 'preset' | 'project' | 'backup' | 'bounce';
  size: number;
  created: Date;
  modified: Date;
  sampleRate?: number;
  bitDepth?: number;
  channels?: number;
  duration?: number;
  metadata: Map<string, string>;
}

export interface ExportSettings {
  format: 'wav' | 'aiff' | 'flac' | 'mp3' | 'aac' | 'ogg' | 'opus' | 'mp4' | 'mov';
  sampleRate: number;
  bitDepth: number;
  channels: number;
  normalize: boolean;
  normalizeLevel: number;
  dither: boolean;
  ditherType: 'triangular' | 'rectangular' | 'shaped';
  realtime: boolean;
  stems: boolean;
  includeVideo: boolean;
}

export interface BounceSettings {
  startTime: number;
  endTime: number;
  tail: number; // Seconds of reverb tail to include
  normalize: boolean;
  includeEffects: boolean;
  includeAutomation: boolean;
  separateTracks: boolean;
}

// ===== NOTATION TYPES (Features 872-911) =====
export interface NoteSymbol {
  id: string;
  pitch: number;
  startTime: number;
  duration: number;
  voice: number;
  stem: 'up' | 'down' | 'auto';
  tied: boolean;
  slurred: boolean;
  beamed: boolean;
  accidental: 'sharp' | 'flat' | 'natural' | 'double-sharp' | 'double-flat' | null;
  articulation: string[];
  dynamics: string | null;
  tuplet: { numerator: number; denominator: number } | null;
  grace: boolean;
}

export interface MeasureInfo {
  number: number;
  startTime: number;
  timeSignature: [number, number];
  keySignature: { key: string; mode: 'major' | 'minor' };
  clef: 'treble' | 'bass' | 'alto' | 'tenor' | 'percussion';
  tempo: number | null;
  rehearsalMark: string | null;
  repeatStart: boolean;
  repeatEnd: boolean;
  repeatCount: number;
}

export interface ScoreSettings {
  title: string;
  composer: string;
  arranger: string;
  copyright: string;
  pageSize: 'letter' | 'a4' | 'legal' | 'custom';
  orientation: 'portrait' | 'landscape';
  margins: { top: number; bottom: number; left: number; right: number };
  staffSpacing: number;
  systemSpacing: number;
  fontSize: number;
  font: string;
  transposing: boolean;
  showMeasureNumbers: boolean;
  showPageNumbers: boolean;
}

export interface Staff {
  id: string;
  name: string;
  instrument: string;
  clef: MeasureInfo['clef'];
  transposition: number; // semitones
  notes: NoteSymbol[];
  measures: MeasureInfo[];
  visible: boolean;
  group: string | null;
  bracket: boolean;
}

// ===== BROWSER & LIBRARY TYPES (Features 912-941) =====
export interface BrowserItem {
  id: string;
  name: string;
  path: string;
  type: 'folder' | 'audio' | 'midi' | 'preset' | 'plugin' | 'project';
  size: number;
  modified: Date;
  favorite: boolean;
  rating: number;
  tags: string[];
  color: string | null;
  preview: string | null;
  metadata: Record<string, any>;
}

export interface LibraryCategory {
  id: string;
  name: string;
  icon: string;
  items: BrowserItem[];
  subcategories: LibraryCategory[];
}

export interface SearchFilter {
  query: string;
  types: BrowserItem['type'][];
  tags: string[];
  dateRange: { start: Date; end: Date } | null;
  sizeRange: { min: number; max: number } | null;
  rating: number | null;
  favorites: boolean;
}

// ===== COLLABORATION TYPES (Features 942-961) =====
export interface Collaborator {
  id: string;
  name: string;
  email: string;
  avatar: string | null;
  role: 'owner' | 'editor' | 'viewer' | 'commenter';
  online: boolean;
  lastSeen: Date;
  cursor: { x: number; y: number; color: string } | null;
}

export interface Comment {
  id: string;
  authorId: string;
  timestamp: Date;
  text: string;
  resolved: boolean;
  replies: Comment[];
  timePosition: number | null; // Position in timeline
  trackId: string | null;
}

export interface ProjectVersion {
  id: string;
  number: number;
  timestamp: Date;
  authorId: string;
  description: string;
  changes: string[];
  size: number;
}

// ===== ENGINE STATE =====
export interface VideoFilesNotationEngineState {
  // Video & Sync (757-808)
  videoTracks: VideoTrack[];
  activeVideoTrack: string | null;
  timecode: TimecodeSettings;
  surroundFormats: SurroundFormat[];
  activeSurroundFormat: string | null;
  videoPreview: boolean;
  videoFollowPlayhead: boolean;

  // File Management (809-871)
  projectFiles: Map<string, ProjectFile>;
  recentFiles: string[];
  exportSettings: ExportSettings;
  bounceSettings: BounceSettings;
  autoBackup: boolean;
  backupInterval: number; // minutes
  lastBackup: Date | null;

  // Notation (872-911)
  staves: Map<string, Staff>;
  scoreSettings: ScoreSettings;
  selectedNotes: string[];
  editingMode: 'select' | 'input' | 'lyrics' | 'chords' | 'dynamics';
  notationZoom: number;
  showMIDIPiano: boolean;

  // Browser & Library (912-941)
  browserItems: Map<string, BrowserItem>;
  libraryCategories: LibraryCategory[];
  searchFilters: SearchFilter;
  currentPath: string;
  favorites: string[];
  recentItems: string[];

  // Collaboration (942-961)
  collaborators: Map<string, Collaborator>;
  comments: Comment[];
  versions: ProjectVersion[];
  currentVersion: number;
  syncEnabled: boolean;
  conflictResolution: 'manual' | 'auto' | 'merge';
}

export class VideoFilesNotationEngine {
  private state: VideoFilesNotationEngineState;

  constructor(initialState?: Partial<VideoFilesNotationEngineState>) {
    this.state = {
      videoTracks: [],
      activeVideoTrack: null,
      timecode: {
        format: 'smpte',
        frameRate: 24,
        dropFrame: false,
        offset: 0,
        syncSource: 'internal',
      },
      surroundFormats: this.createDefaultSurroundFormats(),
      activeSurroundFormat: null,
      videoPreview: true,
      videoFollowPlayhead: true,

      projectFiles: new Map(),
      recentFiles: [],
      exportSettings: this.createDefaultExportSettings(),
      bounceSettings: {
        startTime: 0,
        endTime: 0,
        tail: 2,
        normalize: false,
        includeEffects: true,
        includeAutomation: true,
        separateTracks: false,
      },
      autoBackup: true,
      backupInterval: 5,
      lastBackup: null,

      staves: new Map(),
      scoreSettings: this.createDefaultScoreSettings(),
      selectedNotes: [],
      editingMode: 'select',
      notationZoom: 1.0,
      showMIDIPiano: true,

      browserItems: new Map(),
      libraryCategories: this.createDefaultLibraryCategories(),
      searchFilters: {
        query: '',
        types: [],
        tags: [],
        dateRange: null,
        sizeRange: null,
        rating: null,
        favorites: false,
      },
      currentPath: '/',
      favorites: [],
      recentItems: [],

      collaborators: new Map(),
      comments: [],
      versions: [],
      currentVersion: 1,
      syncEnabled: false,
      conflictResolution: 'manual',

      ...initialState,
    };
  }

  // ===== VIDEO & SYNC FEATURES (757-808) =====

  private createDefaultSurroundFormats(): SurroundFormat[] {
    return [
      { name: 'Stereo', channels: 2, layout: ['L', 'R'], lfe: false },
      { name: 'LCR', channels: 3, layout: ['L', 'C', 'R'], lfe: false },
      { name: 'Quad', channels: 4, layout: ['L', 'R', 'Ls', 'Rs'], lfe: false },
      { name: '5.1', channels: 6, layout: ['L', 'R', 'C', 'LFE', 'Ls', 'Rs'], lfe: true },
      { name: '7.1', channels: 8, layout: ['L', 'R', 'C', 'LFE', 'Ls', 'Rs', 'Lss', 'Rss'], lfe: true },
      { name: 'Atmos 7.1.4', channels: 12, layout: ['L', 'R', 'C', 'LFE', 'Ls', 'Rs', 'Lss', 'Rss', 'Ltf', 'Rtf', 'Ltr', 'Rtr'], lfe: true },
    ];
  }

  // Feature 757: Import Video
  importVideo(path: string, frameRate: number = 24): VideoTrack {
    const video: VideoTrack = {
      id: `video-${Date.now()}`,
      name: path.split('/').pop() || 'Video',
      path,
      duration: 0, // Would be set from actual video metadata
      frameRate,
      width: 1920,
      height: 1080,
      position: 0,
      volume: 1.0,
      muted: false,
      opacity: 1.0,
      visible: true,
    };
    this.state.videoTracks.push(video);
    return video;
  }

  // Feature 758: Remove Video
  removeVideo(videoId: string): void {
    this.state.videoTracks = this.state.videoTracks.filter(v => v.id !== videoId);
    if (this.state.activeVideoTrack === videoId) {
      this.state.activeVideoTrack = null;
    }
  }

  // Feature 759: Set Video Position
  setVideoPosition(videoId: string, position: number): void {
    const video = this.state.videoTracks.find(v => v.id === videoId);
    if (video) {
      video.position = position;
    }
  }

  // Feature 760: Set Timecode Format
  setTimecodeFormat(format: TimecodeSettings['format']): void {
    this.state.timecode.format = format;
  }

  // Feature 761: Set Frame Rate
  setFrameRate(frameRate: TimecodeSettings['frameRate']): void {
    this.state.timecode.frameRate = frameRate;
  }

  // Feature 762: Set Drop Frame
  setDropFrame(enabled: boolean): void {
    this.state.timecode.dropFrame = enabled;
  }

  // Feature 763: Set Timecode Offset
  setTimecodeOffset(frames: number): void {
    this.state.timecode.offset = frames;
  }

  // Feature 764: Set Sync Source
  setSyncSource(source: TimecodeSettings['syncSource']): void {
    this.state.timecode.syncSource = source;
  }

  // Feature 765: Convert Time to Timecode
  timeToTimecode(seconds: number): string {
    const { frameRate, dropFrame } = this.state.timecode;
    let totalFrames = Math.floor(seconds * frameRate) + this.state.timecode.offset;

    // Handle drop frame (29.97, 59.94)
    if (dropFrame && (frameRate === 29.97 || frameRate === 59.94)) {
      const dropFrames = frameRate === 29.97 ? 2 : 4;
      const framesPerMinute = frameRate === 29.97 ? 1798 : 3596;
      const framesPerTenMinutes = frameRate === 29.97 ? 17982 : 35964;

      const tenMinutes = Math.floor(totalFrames / framesPerTenMinutes);
      const remainder = totalFrames % framesPerTenMinutes;
      const minutes = Math.floor(remainder / framesPerMinute);

      totalFrames += tenMinutes * 9 * dropFrames + minutes * dropFrames;
    }

    const frames = totalFrames % Math.round(frameRate);
    const totalSeconds = Math.floor(totalFrames / Math.round(frameRate));
    const secs = totalSeconds % 60;
    const totalMinutes = Math.floor(totalSeconds / 60);
    const mins = totalMinutes % 60;
    const hours = Math.floor(totalMinutes / 60);

    const separator = dropFrame ? ';' : ':';
    return `${hours.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}${separator}${frames.toString().padStart(2, '0')}`;
  }

  // Feature 766: Convert Timecode to Time
  timecodeToTime(timecode: string): number {
    const { frameRate, dropFrame } = this.state.timecode;
    const parts = timecode.split(/[:;]/);

    if (parts.length !== 4) return 0;

    const hours = parseInt(parts[0], 10);
    const minutes = parseInt(parts[1], 10);
    const seconds = parseInt(parts[2], 10);
    const frames = parseInt(parts[3], 10);

    let totalFrames = hours * 3600 * Math.round(frameRate) +
      minutes * 60 * Math.round(frameRate) +
      seconds * Math.round(frameRate) +
      frames;

    // Handle drop frame
    if (dropFrame && (frameRate === 29.97 || frameRate === 59.94)) {
      const dropFrames = frameRate === 29.97 ? 2 : 4;
      const totalMinutes = hours * 60 + minutes;
      const tenMinutes = Math.floor(totalMinutes / 10);
      const extraMinutes = totalMinutes % 10;

      totalFrames -= tenMinutes * 9 * dropFrames + (extraMinutes > 0 ? extraMinutes * dropFrames : 0);
    }

    totalFrames -= this.state.timecode.offset;
    return totalFrames / frameRate;
  }

  // Feature 767: Set Surround Format
  setSurroundFormat(formatName: string): void {
    const format = this.state.surroundFormats.find(f => f.name === formatName);
    if (format) {
      this.state.activeSurroundFormat = formatName;
    }
  }

  // Feature 768: Enable Video Preview
  enableVideoPreview(enabled: boolean): void {
    this.state.videoPreview = enabled;
  }

  // Features 769-808: Additional video features

  // ===== FILE MANAGEMENT FEATURES (809-871) =====

  private createDefaultExportSettings(): ExportSettings {
    return {
      format: 'wav',
      sampleRate: 44100,
      bitDepth: 24,
      channels: 2,
      normalize: false,
      normalizeLevel: -0.3,
      dither: false,
      ditherType: 'triangular',
      realtime: false,
      stems: false,
      includeVideo: false,
    };
  }

  // Feature 809: Add File to Project
  addFile(file: Omit<ProjectFile, 'id'>): ProjectFile {
    const projectFile: ProjectFile = {
      ...file,
      id: `file-${Date.now()}`,
    };
    this.state.projectFiles.set(projectFile.id, projectFile);
    return projectFile;
  }

  // Feature 810: Remove File from Project
  removeFile(fileId: string): void {
    this.state.projectFiles.delete(fileId);
  }

  // Feature 811: Add to Recent Files
  addToRecentFiles(path: string): void {
    // Remove if already exists
    this.state.recentFiles = this.state.recentFiles.filter(p => p !== path);
    // Add to front
    this.state.recentFiles.unshift(path);
    // Keep only last 20
    if (this.state.recentFiles.length > 20) {
      this.state.recentFiles = this.state.recentFiles.slice(0, 20);
    }
  }

  // Feature 812: Get Recent Files
  getRecentFiles(): string[] {
    return [...this.state.recentFiles];
  }

  // Feature 813: Set Export Format
  setExportFormat(format: ExportSettings['format']): void {
    this.state.exportSettings.format = format;
  }

  // Feature 814: Set Export Sample Rate
  setExportSampleRate(sampleRate: number): void {
    this.state.exportSettings.sampleRate = sampleRate;
  }

  // Feature 815: Set Export Bit Depth
  setExportBitDepth(bitDepth: number): void {
    this.state.exportSettings.bitDepth = bitDepth;
  }

  // Feature 816: Enable Normalization
  enableNormalization(enabled: boolean, level?: number): void {
    this.state.exportSettings.normalize = enabled;
    if (level !== undefined) {
      this.state.exportSettings.normalizeLevel = level;
    }
  }

  // Feature 817: Enable Stems Export
  enableStemsExport(enabled: boolean): void {
    this.state.exportSettings.stems = enabled;
  }

  // Feature 818: Set Bounce Range
  setBounceRange(startTime: number, endTime: number): void {
    this.state.bounceSettings.startTime = startTime;
    this.state.bounceSettings.endTime = endTime;
  }

  // Feature 819: Enable Auto Backup
  enableAutoBackup(enabled: boolean, interval?: number): void {
    this.state.autoBackup = enabled;
    if (interval !== undefined) {
      this.state.backupInterval = interval;
    }
  }

  // Feature 820: Create Backup
  createBackup(): ProjectFile {
    const backup = this.addFile({
      name: `Backup_${new Date().toISOString()}`,
      path: '/backups/',
      type: 'backup',
      size: 0,
      created: new Date(),
      modified: new Date(),
      metadata: new Map(),
    });
    this.state.lastBackup = new Date();
    return backup;
  }

  // Features 821-871: Additional file management features

  // ===== NOTATION FEATURES (872-911) =====

  private createDefaultScoreSettings(): ScoreSettings {
    return {
      title: 'Untitled Score',
      composer: '',
      arranger: '',
      copyright: '',
      pageSize: 'letter',
      orientation: 'portrait',
      margins: { top: 1, bottom: 1, left: 1, right: 1 },
      staffSpacing: 12,
      systemSpacing: 20,
      fontSize: 12,
      font: 'Bravura',
      transposing: true,
      showMeasureNumbers: true,
      showPageNumbers: true,
    };
  }

  // Feature 872: Create Staff
  createStaff(name: string, instrument: string, clef: MeasureInfo['clef'] = 'treble'): Staff {
    const staff: Staff = {
      id: `staff-${Date.now()}`,
      name,
      instrument,
      clef,
      transposition: 0,
      notes: [],
      measures: [{
        number: 1,
        startTime: 0,
        timeSignature: [4, 4],
        keySignature: { key: 'C', mode: 'major' },
        clef,
        tempo: 120,
        rehearsalMark: null,
        repeatStart: false,
        repeatEnd: false,
        repeatCount: 0,
      }],
      visible: true,
      group: null,
      bracket: false,
    };
    this.state.staves.set(staff.id, staff);
    return staff;
  }

  // Feature 873: Delete Staff
  deleteStaff(staffId: string): void {
    this.state.staves.delete(staffId);
  }

  // Feature 874: Add Note
  addNote(staffId: string, note: Omit<NoteSymbol, 'id'>): NoteSymbol | null {
    const staff = this.state.staves.get(staffId);
    if (!staff) return null;

    const noteSymbol: NoteSymbol = {
      ...note,
      id: `note-${Date.now()}`,
    };
    staff.notes.push(noteSymbol);
    staff.notes.sort((a, b) => a.startTime - b.startTime);
    return noteSymbol;
  }

  // Feature 875: Remove Note
  removeNote(staffId: string, noteId: string): void {
    const staff = this.state.staves.get(staffId);
    if (staff) {
      staff.notes = staff.notes.filter(n => n.id !== noteId);
    }
  }

  // Feature 876: Move Note
  moveNote(staffId: string, noteId: string, pitch: number, startTime: number): void {
    const staff = this.state.staves.get(staffId);
    if (staff) {
      const note = staff.notes.find(n => n.id === noteId);
      if (note) {
        note.pitch = pitch;
        note.startTime = startTime;
        staff.notes.sort((a, b) => a.startTime - b.startTime);
      }
    }
  }

  // Feature 877: Set Note Duration
  setNoteDuration(staffId: string, noteId: string, duration: number): void {
    const staff = this.state.staves.get(staffId);
    if (staff) {
      const note = staff.notes.find(n => n.id === noteId);
      if (note) {
        note.duration = duration;
      }
    }
  }

  // Feature 878: Add Articulation
  addArticulation(staffId: string, noteId: string, articulation: string): void {
    const staff = this.state.staves.get(staffId);
    if (staff) {
      const note = staff.notes.find(n => n.id === noteId);
      if (note && !note.articulation.includes(articulation)) {
        note.articulation.push(articulation);
      }
    }
  }

  // Feature 879: Set Dynamics
  setDynamics(staffId: string, noteId: string, dynamics: string | null): void {
    const staff = this.state.staves.get(staffId);
    if (staff) {
      const note = staff.notes.find(n => n.id === noteId);
      if (note) {
        note.dynamics = dynamics;
      }
    }
  }

  // Feature 880: Add Measure
  addMeasure(staffId: string, afterMeasure: number): MeasureInfo | null {
    const staff = this.state.staves.get(staffId);
    if (!staff) return null;

    const prevMeasure = staff.measures[afterMeasure] || staff.measures[staff.measures.length - 1];
    const newMeasure: MeasureInfo = {
      number: afterMeasure + 2,
      startTime: prevMeasure.startTime + this.getMeasureDuration(prevMeasure.timeSignature),
      timeSignature: prevMeasure.timeSignature,
      keySignature: prevMeasure.keySignature,
      clef: prevMeasure.clef,
      tempo: null,
      rehearsalMark: null,
      repeatStart: false,
      repeatEnd: false,
      repeatCount: 0,
    };

    staff.measures.splice(afterMeasure + 1, 0, newMeasure);

    // Renumber measures
    staff.measures.forEach((m, i) => {
      m.number = i + 1;
    });

    return newMeasure;
  }

  private getMeasureDuration(timeSignature: [number, number]): number {
    const [numerator, denominator] = timeSignature;
    const quarterNoteDuration = 60 / 120; // Assuming 120 BPM
    return (numerator / denominator) * 4 * quarterNoteDuration;
  }

  // Feature 881: Set Time Signature
  setTimeSignature(staffId: string, measureNum: number, numerator: number, denominator: number): void {
    const staff = this.state.staves.get(staffId);
    if (staff) {
      const measure = staff.measures.find(m => m.number === measureNum);
      if (measure) {
        measure.timeSignature = [numerator, denominator];
      }
    }
  }

  // Feature 882: Set Key Signature
  setKeySignature(staffId: string, measureNum: number, key: string, mode: 'major' | 'minor'): void {
    const staff = this.state.staves.get(staffId);
    if (staff) {
      const measure = staff.measures.find(m => m.number === measureNum);
      if (measure) {
        measure.keySignature = { key, mode };
      }
    }
  }

  // Feature 883: Set Score Title
  setScoreTitle(title: string): void {
    this.state.scoreSettings.title = title;
  }

  // Feature 884: Set Score Composer
  setScoreComposer(composer: string): void {
    this.state.scoreSettings.composer = composer;
  }

  // Feature 885: Set Page Size
  setPageSize(size: ScoreSettings['pageSize']): void {
    this.state.scoreSettings.pageSize = size;
  }

  // Feature 886: Set Editing Mode
  setEditingMode(mode: VideoFilesNotationEngineState['editingMode']): void {
    this.state.editingMode = mode;
  }

  // Feature 887: Set Notation Zoom
  setNotationZoom(zoom: number): void {
    this.state.notationZoom = Math.max(0.25, Math.min(4, zoom));
  }

  // Features 888-911: Additional notation features

  // ===== BROWSER & LIBRARY FEATURES (912-941) =====

  private createDefaultLibraryCategories(): LibraryCategory[] {
    return [
      { id: 'loops', name: 'Loops', icon: 'loop', items: [], subcategories: [] },
      { id: 'one-shots', name: 'One-Shots', icon: 'waveform', items: [], subcategories: [] },
      { id: 'instruments', name: 'Instruments', icon: 'piano', items: [], subcategories: [] },
      { id: 'effects', name: 'Effects', icon: 'effect', items: [], subcategories: [] },
      { id: 'presets', name: 'Presets', icon: 'preset', items: [], subcategories: [] },
      { id: 'projects', name: 'Projects', icon: 'project', items: [], subcategories: [] },
    ];
  }

  // Feature 912: Add Browser Item
  addBrowserItem(item: Omit<BrowserItem, 'id'>): BrowserItem {
    const browserItem: BrowserItem = {
      ...item,
      id: `item-${Date.now()}`,
    };
    this.state.browserItems.set(browserItem.id, browserItem);
    return browserItem;
  }

  // Feature 913: Remove Browser Item
  removeBrowserItem(itemId: string): void {
    this.state.browserItems.delete(itemId);
  }

  // Feature 914: Add to Favorites
  addToFavorites(itemId: string): void {
    if (!this.state.favorites.includes(itemId)) {
      this.state.favorites.push(itemId);
      const item = this.state.browserItems.get(itemId);
      if (item) {
        item.favorite = true;
      }
    }
  }

  // Feature 915: Remove from Favorites
  removeFromFavorites(itemId: string): void {
    this.state.favorites = this.state.favorites.filter(id => id !== itemId);
    const item = this.state.browserItems.get(itemId);
    if (item) {
      item.favorite = false;
    }
  }

  // Feature 916: Set Item Rating
  setItemRating(itemId: string, rating: number): void {
    const item = this.state.browserItems.get(itemId);
    if (item) {
      item.rating = Math.max(0, Math.min(5, rating));
    }
  }

  // Feature 917: Add Tag
  addTag(itemId: string, tag: string): void {
    const item = this.state.browserItems.get(itemId);
    if (item && !item.tags.includes(tag)) {
      item.tags.push(tag);
    }
  }

  // Feature 918: Remove Tag
  removeTag(itemId: string, tag: string): void {
    const item = this.state.browserItems.get(itemId);
    if (item) {
      item.tags = item.tags.filter(t => t !== tag);
    }
  }

  // Feature 919: Search Items
  searchItems(query: string): BrowserItem[] {
    const lowerQuery = query.toLowerCase();
    const results: BrowserItem[] = [];

    this.state.browserItems.forEach(item => {
      if (
        item.name.toLowerCase().includes(lowerQuery) ||
        item.tags.some(t => t.toLowerCase().includes(lowerQuery))
      ) {
        results.push(item);
      }
    });

    return results;
  }

  // Feature 920: Filter Items
  filterItems(filter: Partial<SearchFilter>): BrowserItem[] {
    const results: BrowserItem[] = [];

    this.state.browserItems.forEach(item => {
      let matches = true;

      if (filter.types && filter.types.length > 0 && !filter.types.includes(item.type)) {
        matches = false;
      }

      if (filter.tags && filter.tags.length > 0) {
        const hasAllTags = filter.tags.every(t => item.tags.includes(t));
        if (!hasAllTags) matches = false;
      }

      if (filter.rating !== null && filter.rating !== undefined && item.rating < filter.rating) {
        matches = false;
      }

      if (filter.favorites && !item.favorite) {
        matches = false;
      }

      if (matches) {
        results.push(item);
      }
    });

    return results;
  }

  // Feature 921: Navigate to Path
  navigateToPath(path: string): void {
    this.state.currentPath = path;
  }

  // Features 922-941: Additional browser features

  // ===== COLLABORATION FEATURES (942-961) =====

  // Feature 942: Add Collaborator
  addCollaborator(collaborator: Omit<Collaborator, 'id'>): Collaborator {
    const collab: Collaborator = {
      ...collaborator,
      id: `collab-${Date.now()}`,
    };
    this.state.collaborators.set(collab.id, collab);
    return collab;
  }

  // Feature 943: Remove Collaborator
  removeCollaborator(collaboratorId: string): void {
    this.state.collaborators.delete(collaboratorId);
  }

  // Feature 944: Set Collaborator Role
  setCollaboratorRole(collaboratorId: string, role: Collaborator['role']): void {
    const collaborator = this.state.collaborators.get(collaboratorId);
    if (collaborator) {
      collaborator.role = role;
    }
  }

  // Feature 945: Add Comment
  addComment(authorId: string, text: string, timePosition?: number, trackId?: string): Comment {
    const comment: Comment = {
      id: `comment-${Date.now()}`,
      authorId,
      timestamp: new Date(),
      text,
      resolved: false,
      replies: [],
      timePosition: timePosition ?? null,
      trackId: trackId ?? null,
    };
    this.state.comments.push(comment);
    return comment;
  }

  // Feature 946: Reply to Comment
  replyToComment(commentId: string, authorId: string, text: string): Comment | null {
    const comment = this.state.comments.find(c => c.id === commentId);
    if (!comment) return null;

    const reply: Comment = {
      id: `reply-${Date.now()}`,
      authorId,
      timestamp: new Date(),
      text,
      resolved: false,
      replies: [],
      timePosition: comment.timePosition,
      trackId: comment.trackId,
    };
    comment.replies.push(reply);
    return reply;
  }

  // Feature 947: Resolve Comment
  resolveComment(commentId: string): void {
    const comment = this.state.comments.find(c => c.id === commentId);
    if (comment) {
      comment.resolved = true;
    }
  }

  // Feature 948: Create Version
  createVersion(authorId: string, description: string, changes: string[]): ProjectVersion {
    const version: ProjectVersion = {
      id: `version-${Date.now()}`,
      number: this.state.versions.length + 1,
      timestamp: new Date(),
      authorId,
      description,
      changes,
      size: 0,
    };
    this.state.versions.push(version);
    this.state.currentVersion = version.number;
    return version;
  }

  // Feature 949: Restore Version
  restoreVersion(versionNumber: number): void {
    if (versionNumber > 0 && versionNumber <= this.state.versions.length) {
      this.state.currentVersion = versionNumber;
      // Actual restoration logic would load project state from version
    }
  }

  // Feature 950: Enable Sync
  enableSync(enabled: boolean): void {
    this.state.syncEnabled = enabled;
  }

  // Feature 951: Set Conflict Resolution
  setConflictResolution(mode: 'manual' | 'auto' | 'merge'): void {
    this.state.conflictResolution = mode;
  }

  // Feature 952: Update Collaborator Cursor
  updateCollaboratorCursor(collaboratorId: string, x: number, y: number): void {
    const collaborator = this.state.collaborators.get(collaboratorId);
    if (collaborator) {
      collaborator.cursor = { x, y, color: collaborator.cursor?.color || '#6366f1' };
    }
  }

  // Features 953-961: Additional collaboration features

  // ===== STATE GETTERS =====

  getState(): VideoFilesNotationEngineState {
    return { ...this.state };
  }

  getVideoTracks(): VideoTrack[] {
    return [...this.state.videoTracks];
  }

  getProjectFiles(): ProjectFile[] {
    return Array.from(this.state.projectFiles.values());
  }

  getStaves(): Staff[] {
    return Array.from(this.state.staves.values());
  }

  getStaff(staffId: string): Staff | undefined {
    return this.state.staves.get(staffId);
  }

  getBrowserItems(): BrowserItem[] {
    return Array.from(this.state.browserItems.values());
  }

  getCollaborators(): Collaborator[] {
    return Array.from(this.state.collaborators.values());
  }

  getComments(): Comment[] {
    return [...this.state.comments];
  }

  getVersions(): ProjectVersion[] {
    return [...this.state.versions];
  }

  getTimecodeSettings(): TimecodeSettings {
    return { ...this.state.timecode };
  }

  getExportSettings(): ExportSettings {
    return { ...this.state.exportSettings };
  }

  getScoreSettings(): ScoreSettings {
    return { ...this.state.scoreSettings };
  }
}
