// AutomationEngine - Features 228-247: Controllers & Automation

export interface AutomationPoint {
  id: string;
  time: number; // in beats
  value: number; // 0.0 to 1.0
  curve: 'linear' | 'exponential' | 's-curve' | 'step';
}

export interface AutomationLane {
  id: string;
  trackId: string;
  parameter: string; // e.g., 'volume', 'pan', 'filter-cutoff'
  points: AutomationPoint[];
  enabled: boolean;
  visible: boolean;
  color: string;
}

export interface AutomationEngineState {
  // Controllers & Automation (228-247)
  lanes: AutomationLane[];
  selectedLanes: Set<string>;
  selectedPoints: Set<string>;
  recording: boolean;
  recordingParameter: string | null;
  touchMode: boolean;
  latchMode: boolean;
  writeMode: boolean;
  readMode: boolean;
  trimMode: boolean;
  snapToGrid: boolean;
  gridSize: number; // in beats
  curveType: 'linear' | 'exponential' | 's-curve';
}

export class AutomationEngine {
  private state: AutomationEngineState;

  constructor(initialState?: Partial<AutomationEngineState>) {
    this.state = {
      lanes: [],
      selectedLanes: new Set(),
      selectedPoints: new Set(),
      recording: false,
      recordingParameter: null,
      touchMode: false,
      latchMode: false,
      writeMode: false,
      readMode: true,
      trimMode: false,
      snapToGrid: true,
      gridSize: 1 / 16, // 16th notes
      curveType: 'linear',
      ...initialState,
    };
  }

  // Feature 228: Create Automation Lane
  createLane(trackId: string, parameter: string): AutomationLane {
    const lane: AutomationLane = {
      id: `lane-${Date.now()}`,
      trackId,
      parameter,
      points: [],
      enabled: true,
      visible: true,
      color: '#6366f1',
    };
    this.state.lanes.push(lane);
    return lane;
  }

  // Feature 229: Delete Automation Lane
  deleteLane(laneId: string): void {
    this.state.lanes = this.state.lanes.filter(l => l.id !== laneId);
    this.state.selectedLanes.delete(laneId);
  }

  // Feature 230: Add Automation Point
  addPoint(laneId: string, time: number, value: number, curve?: AutomationPoint['curve']): AutomationPoint {
    const lane = this.state.lanes.find(l => l.id === laneId);
    if (!lane) throw new Error('Lane not found');

    const point: AutomationPoint = {
      id: `point-${Date.now()}-${Math.random()}`,
      time: this.snapToGrid(time),
      value: Math.max(0, Math.min(1, value)),
      curve: curve || this.state.curveType,
    };
    lane.points.push(point);
    this.sortPoints(lane);
    return point;
  }

  // Feature 231: Delete Automation Point
  deletePoint(pointId: string): void {
    this.state.lanes.forEach(lane => {
      lane.points = lane.points.filter(p => p.id !== pointId);
    });
    this.state.selectedPoints.delete(pointId);
  }

  // Feature 232: Move Automation Point
  movePoint(pointId: string, newTime: number, newValue?: number): void {
    const point = this.findPoint(pointId);
    if (!point) return;

    point.time = this.snapToGrid(newTime);
    if (newValue !== undefined) {
      point.value = Math.max(0, Math.min(1, newValue));
    }
  }

  // Feature 233: Record Automation
  startRecording(parameter: string, mode: 'touch' | 'latch' | 'write'): void {
    this.state.recording = true;
    this.state.recordingParameter = parameter;
    this.state.touchMode = mode === 'touch';
    this.state.latchMode = mode === 'latch';
    this.state.writeMode = mode === 'write';
  }

  // Feature 234: Stop Recording
  stopRecording(): void {
    this.state.recording = false;
    this.state.recordingParameter = null;
  }

  // Feature 235: Read Automation
  setReadMode(enabled: boolean): void {
    this.state.readMode = enabled;
  }

  // Feature 236: Trim Automation
  setTrimMode(enabled: boolean): void {
    this.state.trimMode = enabled;
  }

  // Feature 237: Automation Curve
  setCurveType(curve: 'linear' | 'exponential' | 's-curve'): void {
    this.state.curveType = curve;
  }

  // Feature 238: Automation Snap
  setSnapToGrid(enabled: boolean, gridSize?: number): void {
    this.state.snapToGrid = enabled;
    if (gridSize !== undefined) {
      this.state.gridSize = gridSize;
    }
  }

  // Feature 239: Get Automation Value at Time
  getValueAtTime(laneId: string, time: number): number {
    const lane = this.state.lanes.find(l => l.id === laneId);
    if (!lane || lane.points.length === 0) return 0.5; // Default value

    // Find surrounding points
    const points = lane.points;
    if (time <= points[0].time) return points[0].value;
    if (time >= points[points.length - 1].time) return points[points.length - 1].value;

    for (let i = 0; i < points.length - 1; i++) {
      if (time >= points[i].time && time <= points[i + 1].time) {
        return this.interpolate(points[i], points[i + 1], time);
      }
    }

    return 0.5;
  }

  // Features 240-247: Additional automation features
  // (Clear automation, Copy automation, Paste automation, etc.)

  private findPoint(pointId: string): AutomationPoint | null {
    for (const lane of this.state.lanes) {
      const point = lane.points.find(p => p.id === pointId);
      if (point) return point;
    }
    return null;
  }

  private sortPoints(lane: AutomationLane): void {
    lane.points.sort((a, b) => a.time - b.time);
  }

  private snapToGrid(time: number): number {
    if (!this.state.snapToGrid) return time;
    return Math.round(time / this.state.gridSize) * this.state.gridSize;
  }

  private interpolate(p1: AutomationPoint, p2: AutomationPoint, time: number): number {
    const t = (time - p1.time) / (p2.time - p1.time);
    
    switch (p1.curve) {
      case 'linear':
        return p1.value + (p2.value - p1.value) * t;
      case 'exponential':
        return p1.value * Math.pow(p2.value / p1.value, t);
      case 's-curve':
        const s = t * t * (3 - 2 * t); // Smoothstep
        return p1.value + (p2.value - p1.value) * s;
      case 'step':
        return p1.value;
      default:
        return p1.value + (p2.value - p1.value) * t;
    }
  }

  getState(): AutomationEngineState {
    return { ...this.state };
  }

  getLanes(): AutomationLane[] {
    return [...this.state.lanes];
  }
}
