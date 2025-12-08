// RecordingEngine - Core recording functionality

export interface RecordingConfig {
  channels: number; // 1 = mono, 2 = stereo, >2 = multi-track
  sampleRate: number;
  bitDepth: number;
  format: 'wav' | 'mp3' | 'flac';
}

export interface Take {
  id: string;
  startTime: number;
  endTime: number;
  audioBuffer: AudioBuffer | null;
  filePath?: string;
  selected: boolean;
}

export interface Track {
  id: string;
  name: string;
  armed: boolean;
  recordSafe: boolean;
  inputMonitoring: 'off' | 'software' | 'hardware' | 'direct';
  takes: Take[];
  currentTake: number;
  inputChannel: number;
  pan: number;
  volume: number;
}

export interface RecordingEngineState {
  isRecording: boolean;
  isPunching: boolean;
  punchInTime: number | null;
  punchOutTime: number | null;
  autoPunch: boolean;
  preRoll: number; // seconds
  postRoll: number; // seconds
  loopRecording: boolean;
  loopStart: number;
  loopEnd: number;
  takeIncrement: number;
  tracks: Track[];
  currentTime: number;
  retrospectiveBuffer: AudioBuffer | null;
  retrospectiveEnabled: boolean;
}

export class RecordingEngine {
  private recorder: MediaRecorder | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private state: RecordingEngineState;
  private audioChunks: Blob[] = [];
  private loopInterval: ReturnType<typeof setInterval> | null = null;
  private retrospectiveBuffer: Float32Array[] = [];
  private retrospectiveMaxDuration: number = 30; // 30 seconds buffer

  constructor(initialState?: Partial<RecordingEngineState>) {
    this.state = {
      isRecording: false,
      isPunching: false,
      punchInTime: null,
      punchOutTime: null,
      autoPunch: false,
      preRoll: 2,
      postRoll: 1,
      loopRecording: false,
      loopStart: 0,
      loopEnd: 0,
      takeIncrement: 1,
      tracks: [],
      currentTime: 0,
      retrospectiveBuffer: null,
      retrospectiveEnabled: true,
      ...initialState,
    };
  }

  // Initialize audio context and get user media
  async initialize(): Promise<void> {
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: { ideal: 2 },
          sampleRate: { ideal: 44100 },
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (error) {
      console.error('Failed to initialize recording:', error);
      throw error;
    }
  }

  // Start recording
  async startRecording(
    trackId: string,
    config: Partial<RecordingConfig> = {}
  ): Promise<void> {
    if (!this.mediaStream || !this.audioContext) {
      await this.initialize();
    }

    const track = this.state.tracks.find((t) => t.id === trackId);
    if (!track) {
      throw new Error(`Track ${trackId} not found`);
    }

    if (track.recordSafe) {
      throw new Error(`Track ${trackId} is record-safe`);
    }

    if (!track.armed) {
      throw new Error(`Track ${trackId} is not armed`);
    }

    const recordingConfig: RecordingConfig = {
      channels: config.channels || 2,
      sampleRate: config.sampleRate || 44100,
      bitDepth: config.bitDepth || 24,
      format: config.format || 'wav',
    };

    try {
      // Create MediaRecorder
      const options: MediaRecorderOptions = {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: recordingConfig.bitDepth * 44100,
      };

      this.recorder = new MediaRecorder(this.mediaStream!, options);
      this.audioChunks = [];

      this.recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.recorder.onstop = async () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const audioBuffer = await this.blobToAudioBuffer(audioBlob);
        this.finishRecording(trackId, audioBuffer);
      };

      this.recorder.start();
      this.state.isRecording = true;

      // Start retrospective buffer if enabled
      if (this.state.retrospectiveEnabled) {
        this.startRetrospectiveBuffer();
      }
    } catch (error) {
      console.error('Failed to start recording:', error);
      throw error;
    }
  }

  // Stop recording
  stopRecording(): void {
    if (this.recorder && this.state.isRecording) {
      this.recorder.stop();
      this.state.isRecording = false;
      this.stopRetrospectiveBuffer();
    }
  }

  // Punch-in recording
  async punchIn(trackId: string, time: number): Promise<void> {
    this.state.punchInTime = time;
    this.state.isPunching = true;

    if (this.state.preRoll > 0) {
      // Start pre-roll playback
      await this.startPreRoll(time - this.state.preRoll);
    }

    await this.startRecording(trackId);
  }

  // Punch-out recording
  punchOut(time: number): void {
    this.state.punchOutTime = time;
    this.stopRecording();
    this.state.isPunching = false;
  }

  // Auto punch-in/out
  setAutoPunch(inTime: number, outTime: number): void {
    this.state.autoPunch = true;
    this.state.punchInTime = inTime;
    this.state.punchOutTime = outTime;
  }

  // Loop recording
  startLoopRecording(
    trackId: string,
    loopStart: number,
    loopEnd: number,
    autoIncrement: boolean = true
  ): void {
    this.state.loopRecording = true;
    this.state.loopStart = loopStart;
    this.state.loopEnd = loopEnd;

    const recordLoop = async () => {
      if (this.state.currentTime >= loopStart && this.state.currentTime < loopEnd) {
        if (!this.state.isRecording) {
          await this.startRecording(trackId);
        }
      } else if (this.state.currentTime >= loopEnd) {
        this.stopRecording();
        if (autoIncrement) {
          this.state.takeIncrement++;
        }
        // Create new take
        this.createNewTake(trackId);
      }
    };

    this.loopInterval = setInterval(recordLoop, 100);
  }

  // Take management
  createNewTake(trackId: string): void {
    const track = this.state.tracks.find((t) => t.id === trackId);
    if (!track) return;

    const newTake: Take = {
      id: `take-${Date.now()}`,
      startTime: this.state.loopStart,
      endTime: this.state.loopEnd,
      audioBuffer: null,
      selected: false,
    };

    track.takes.push(newTake);
    track.currentTake = track.takes.length - 1;
  }

  // Comping (composite takes)
  createComp(trackId: string, _takeIds: string[], regions: Array<{ start: number; end: number; takeId: string }>): AudioBuffer {
    const track = this.state.tracks.find((t) => t.id === trackId);
    if (!track) {
      throw new Error(`Track ${trackId} not found`);
    }

    // Find longest duration
    const maxDuration = Math.max(...regions.map((r) => r.end));
    const sampleRate = this.audioContext?.sampleRate || 44100;
    const length = Math.ceil(maxDuration * sampleRate);

    // Create composite buffer
    const compBuffer = this.audioContext!.createBuffer(2, length, sampleRate);

    // Mix regions
    regions.forEach((region) => {
      const take = track.takes.find((t) => t.id === region.takeId);
      if (take && take.audioBuffer) {
        const startSample = Math.floor(region.start * sampleRate);
        const endSample = Math.floor(region.end * sampleRate);
        const regionLength = endSample - startSample;

        for (let channel = 0; channel < Math.min(take.audioBuffer.numberOfChannels, 2); channel++) {
          const sourceData = take.audioBuffer.getChannelData(channel);
          const destData = compBuffer.getChannelData(channel);

          for (let i = 0; i < regionLength && startSample + i < length; i++) {
            destData[startSample + i] += sourceData[i];
          }
        }
      }
    });

    return compBuffer;
  }

  // Retrospective recording (capture buffer)
  startRetrospectiveBuffer(): void {
    if (!this.audioContext || !this.mediaStream) return;

    const source = this.audioContext.createMediaStreamSource(this.mediaStream);
    const bufferSize = 4096;
    const sampleRate = this.audioContext.sampleRate;
    const maxSamples = this.retrospectiveMaxDuration * sampleRate;

    const scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 2, 2);
    this.retrospectiveBuffer = [];

    scriptProcessor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      const buffer = new Float32Array(inputData.length);
      buffer.set(inputData);
      this.retrospectiveBuffer.push(buffer);

      // Keep only last N seconds
      const totalSamples = this.retrospectiveBuffer.reduce((sum, buf) => sum + buf.length, 0);
      if (totalSamples > maxSamples) {
        const samplesToRemove = totalSamples - maxSamples;
        let removed = 0;
        while (removed < samplesToRemove && this.retrospectiveBuffer.length > 0) {
          removed += this.retrospectiveBuffer[0].length;
          this.retrospectiveBuffer.shift();
        }
      }
    };

    source.connect(scriptProcessor);
    scriptProcessor.connect(this.audioContext.destination);
  }

  stopRetrospectiveBuffer(): void {
    // Convert buffer to AudioBuffer
    if (this.retrospectiveBuffer.length > 0 && this.audioContext) {
      const totalLength = this.retrospectiveBuffer.reduce((sum, buf) => sum + buf.length, 0);
      const sampleRate = this.audioContext.sampleRate;
      const buffer = this.audioContext.createBuffer(1, totalLength, sampleRate);
      const channelData = buffer.getChannelData(0);

      let offset = 0;
      this.retrospectiveBuffer.forEach((chunk) => {
        channelData.set(chunk, offset);
        offset += chunk.length;
      });

      this.state.retrospectiveBuffer = buffer;
    }
  }

  // Capture retrospective recording
  captureRetrospective(duration: number): AudioBuffer | null {
    if (!this.state.retrospectiveBuffer) return null;

    const sampleRate = this.audioContext?.sampleRate || 44100;
    const samplesToCapture = Math.floor(duration * sampleRate);
    const bufferLength = this.state.retrospectiveBuffer.length;
    const startSample = Math.max(0, bufferLength - samplesToCapture);

    const capturedBuffer = this.audioContext!.createBuffer(
      this.state.retrospectiveBuffer.numberOfChannels,
      samplesToCapture,
      sampleRate
    );

    for (let channel = 0; channel < this.state.retrospectiveBuffer.numberOfChannels; channel++) {
      const sourceData = this.state.retrospectiveBuffer.getChannelData(channel);
      const destData = capturedBuffer.getChannelData(channel);
      destData.set(sourceData.subarray(startSample));
    }

    return capturedBuffer;
  }

  // Auto-record on signal detection
  async startAutoRecord(trackId: string, threshold: number = 0.01): Promise<void> {
    if (!this.audioContext || !this.mediaStream) {
      await this.initialize();
    }

    const source = this.audioContext!.createMediaStreamSource(this.mediaStream!);
    const analyser = this.audioContext!.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    const checkLevel = () => {
      analyser.getByteTimeDomainData(dataArray);
      const average = dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length;
      const normalized = (average - 128) / 128;

      if (Math.abs(normalized) > threshold && !this.state.isRecording) {
        this.startRecording(trackId);
      }
    };

    setInterval(checkLevel, 100);
  }

  // Voice-activated recording
  async startVoiceActivated(trackId: string, sensitivity: number = 0.02): Promise<void> {
    await this.startAutoRecord(trackId, sensitivity);
  }

  // Helper methods
  private async blobToAudioBuffer(blob: Blob): Promise<AudioBuffer> {
    const arrayBuffer = await blob.arrayBuffer();
    return await this.audioContext!.decodeAudioData(arrayBuffer);
  }

  private finishRecording(trackId: string, audioBuffer: AudioBuffer): void {
    const track = this.state.tracks.find((t) => t.id === trackId);
    if (!track) return;

    const currentTake = track.takes[track.currentTake];
    if (currentTake) {
      currentTake.audioBuffer = audioBuffer;
      currentTake.endTime = this.audioContext!.currentTime;
    }
  }

  private async startPreRoll(_startTime: number): Promise<void> {
    // Implementation for pre-roll playback
    // This would start playback at startTime - preRoll
  }

  // Getters
  getState(): RecordingEngineState {
    return { ...this.state };
  }

  getTracks(): Track[] {
    return [...this.state.tracks];
  }

  // Setters
  setTrackArmed(trackId: string, armed: boolean): void {
    const track = this.state.tracks.find((t) => t.id === trackId);
    if (track) {
      track.armed = armed;
    }
  }

  setRecordSafe(trackId: string, safe: boolean): void {
    const track = this.state.tracks.find((t) => t.id === trackId);
    if (track) {
      track.recordSafe = safe;
    }
  }

  setInputMonitoring(trackId: string, mode: 'off' | 'software' | 'hardware' | 'direct'): void {
    const track = this.state.tracks.find((t) => t.id === trackId);
    if (track) {
      track.inputMonitoring = mode;
    }
  }

  cleanup(): void {
    if (this.recorder) {
      this.recorder.stop();
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
    }
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
    }
    if (this.audioContext) {
      this.audioContext.close();
    }
  }
}
