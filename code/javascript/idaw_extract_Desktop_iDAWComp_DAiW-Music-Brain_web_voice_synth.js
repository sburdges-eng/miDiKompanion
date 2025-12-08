/**
 * DAiW Web Audio Voice Synthesizer
 *
 * Browser-based formant synthesis using Web Audio API and AudioWorklet.
 * Mirrors the C++ VoiceProcessor for consistent cross-platform synthesis.
 *
 * Features:
 * - Real-time formant synthesis using AudioWorklet
 * - WebSocket bridge to Python/C++ backend
 * - MIDI keyboard input support
 * - Real-time vowel/pitch control
 *
 * Usage:
 *   const synth = new DAiWVoiceSynth();
 *   await synth.initialize();
 *
 *   synth.setVowel('a');
 *   synth.noteOn(60, 0.8);
 *   synth.noteOff();
 */

// Vowel formant data (matches C++ VoiceCharacteristics)
const VOWEL_FORMANTS = {
    'a': { f1: 730, f2: 1090, f3: 2440, bw1: 50, bw2: 70, bw3: 90 },  // "ah"
    'e': { f1: 570, f2: 1980, f3: 2440, bw1: 50, bw2: 70, bw3: 90 },  // "eh"
    'i': { f1: 270, f2: 2290, f3: 3010, bw1: 50, bw2: 70, bw3: 90 },  // "ee"
    'o': { f1: 570, f2: 840,  f3: 2410, bw1: 50, bw2: 70, bw3: 90 },  // "oh"
    'u': { f1: 300, f2: 870,  f3: 2240, bw1: 50, bw2: 70, bw3: 90 },  // "oo"
    'schwa': { f1: 500, f2: 1500, f3: 2500, bw1: 60, bw2: 80, bw3: 100 }  // "uh"
};

// Voice characteristics preset
const DEFAULT_VOICE = {
    averagePitch: 200,
    pitchRangeMin: 80,
    pitchRangeMax: 400,
    vibratoRate: 5.5,
    vibratoDepth: 30,  // cents
    jitter: 0.5,       // %
    shimmer: 1.0,      // %
    breathiness: 0.1,
    attackTime: 0.01,
    releaseTime: 0.05
};


/**
 * AudioWorklet processor for formant synthesis
 */
const FORMANT_PROCESSOR_CODE = `
class FormantSynthProcessor extends AudioWorkletProcessor {
    constructor() {
        super();

        // Formant filters (3 formants)
        this.formantFilters = [
            { a1: 0, a2: 0, b0: 0, y1: 0, y2: 0 },
            { a1: 0, a2: 0, b0: 0, y1: 0, y2: 0 },
            { a1: 0, a2: 0, b0: 0, y1: 0, y2: 0 }
        ];

        // Glottal source
        this.phase = 0;
        this.frequency = 200;
        this.openQuotient = 0.5;
        this.returnQuotient = 0.1;

        // Current formants
        this.formants = { f1: 730, f2: 1090, f3: 2440 };
        this.targetFormants = { ...this.formants };
        this.formantTransitionRate = 0.001;

        // Envelope
        this.envelope = 0;
        this.envelopeTarget = 0;
        this.attackRate = 0.001;
        this.releaseRate = 0.0002;

        // Vibrato
        this.vibratoPhase = 0;
        this.vibratoRate = 5.5;
        this.vibratoDepth = 30;

        // Voice quality
        this.jitter = 0.5;
        this.shimmer = 1.0;
        this.breathiness = 0.1;

        this.active = false;

        // Handle messages from main thread
        this.port.onmessage = (event) => {
            const { type, data } = event.data;

            switch (type) {
                case 'noteOn':
                    this.envelopeTarget = data.velocity;
                    this.active = true;
                    break;
                case 'noteOff':
                    this.envelopeTarget = 0;
                    break;
                case 'setFrequency':
                    this.frequency = data.frequency;
                    break;
                case 'setFormants':
                    this.targetFormants = data;
                    break;
                case 'setVoiceParams':
                    Object.assign(this, data);
                    break;
            }
        };
    }

    setFormantFilter(filter, frequency, bandwidth, sampleRate) {
        const omega = 2 * Math.PI * frequency / sampleRate;
        const r = Math.exp(-Math.PI * bandwidth / sampleRate);

        filter.a1 = -2 * r * Math.cos(omega);
        filter.a2 = r * r;
        filter.b0 = (1 - r * r) * 0.5;
    }

    processFormantFilter(filter, input) {
        const output = filter.b0 * input - filter.a1 * filter.y1 - filter.a2 * filter.y2;
        filter.y2 = filter.y1;
        filter.y1 = output;
        return output;
    }

    generateGlottalPulse() {
        let output = 0;

        if (this.phase < this.openQuotient) {
            const t = this.phase / this.openQuotient;
            output = 0.5 * (1 - Math.cos(Math.PI * t));
        } else if (this.phase < this.openQuotient + this.returnQuotient) {
            const t = (this.phase - this.openQuotient) / this.returnQuotient;
            output = Math.exp(-5 * t);
        }

        return output;
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0];

        if (!channel) return true;

        const sampleRate = 44100;  // AudioWorklet runs at context sample rate

        for (let i = 0; i < channel.length; i++) {
            // Update envelope
            if (this.envelope < this.envelopeTarget) {
                this.envelope += this.attackRate;
                if (this.envelope > this.envelopeTarget) this.envelope = this.envelopeTarget;
            } else if (this.envelope > this.envelopeTarget) {
                this.envelope -= this.releaseRate;
                if (this.envelope < 0) {
                    this.envelope = 0;
                    this.active = false;
                }
            }

            if (!this.active && this.envelope === 0) {
                channel[i] = 0;
                continue;
            }

            // Interpolate formants
            this.formants.f1 += (this.targetFormants.f1 - this.formants.f1) * this.formantTransitionRate;
            this.formants.f2 += (this.targetFormants.f2 - this.formants.f2) * this.formantTransitionRate;
            this.formants.f3 += (this.targetFormants.f3 - this.formants.f3) * this.formantTransitionRate;

            // Update formant filters
            this.setFormantFilter(this.formantFilters[0], this.formants.f1, 50, sampleRate);
            this.setFormantFilter(this.formantFilters[1], this.formants.f2, 70, sampleRate);
            this.setFormantFilter(this.formantFilters[2], this.formants.f3, 90, sampleRate);

            // Apply vibrato
            this.vibratoPhase += this.vibratoRate / sampleRate;
            if (this.vibratoPhase >= 1) this.vibratoPhase -= 1;
            const vibratoCents = this.vibratoDepth * Math.sin(2 * Math.PI * this.vibratoPhase);
            const vibratoMod = Math.pow(2, vibratoCents / 1200);

            // Apply jitter
            const jitterMod = 1 + (Math.random() - 0.5) * this.jitter * 0.02;

            // Generate glottal pulse
            const glottalPulse = this.generateGlottalPulse();

            // Advance phase with modulation
            const modulatedFreq = this.frequency * vibratoMod * jitterMod;
            this.phase += modulatedFreq / sampleRate;
            if (this.phase >= 1) this.phase -= 1;

            // Differentiate glottal pulse
            const derivative = glottalPulse * 2;

            // Apply formant filters
            const f1Out = this.processFormantFilter(this.formantFilters[0], derivative);
            const f2Out = this.processFormantFilter(this.formantFilters[1], derivative);
            const f3Out = this.processFormantFilter(this.formantFilters[2], derivative);

            // Mix formants
            let sample = f1Out * 1.0 + f2Out * 0.7 + f3Out * 0.4;

            // Add breathiness
            if (this.breathiness > 0) {
                const noise = (Math.random() * 2 - 1) * 0.3;
                sample = sample * (1 - this.breathiness * 0.5) + noise * this.breathiness;
            }

            // Apply shimmer
            const shimmerMod = 1 + (Math.random() - 0.5) * this.shimmer * 0.02;
            sample *= shimmerMod;

            // Apply envelope
            sample *= this.envelope;

            // Soft clip
            sample = Math.tanh(sample);

            channel[i] = sample * 0.5;
        }

        // Copy to all output channels
        for (let ch = 1; ch < output.length; ch++) {
            output[ch].set(channel);
        }

        return true;
    }
}

registerProcessor('formant-synth-processor', FormantSynthProcessor);
`;


/**
 * Main DAiW Voice Synthesizer class
 */
class DAiWVoiceSynth {
    constructor(options = {}) {
        this.audioContext = null;
        this.workletNode = null;
        this.gainNode = null;

        this.wsUrl = options.wsUrl || 'ws://localhost:8765';
        this.ws = null;

        this.currentVowel = 'a';
        this.currentPitch = 200;
        this.voice = { ...DEFAULT_VOICE };

        this.isInitialized = false;
    }

    /**
     * Initialize the synthesizer
     */
    async initialize() {
        // Create audio context
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Create worklet from inline code
        const blob = new Blob([FORMANT_PROCESSOR_CODE], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);

        try {
            await this.audioContext.audioWorklet.addModule(url);
        } finally {
            URL.revokeObjectURL(url);
        }

        // Create worklet node
        this.workletNode = new AudioWorkletNode(this.audioContext, 'formant-synth-processor');

        // Create gain node for volume control
        this.gainNode = this.audioContext.createGain();
        this.gainNode.gain.value = 0.8;

        // Connect nodes
        this.workletNode.connect(this.gainNode);
        this.gainNode.connect(this.audioContext.destination);

        // Set initial voice parameters
        this.updateVoiceParams();

        this.isInitialized = true;
        console.log('DAiW Voice Synth initialized');

        return this;
    }

    /**
     * Connect to Python/C++ backend via WebSocket
     */
    async connectBackend() {
        return new Promise((resolve, reject) => {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => {
                console.log('Connected to DAiW backend');
                this.ws.send(JSON.stringify({ command: 'ready' }));
                resolve();
            };

            this.ws.onmessage = (event) => {
                this.handleBackendMessage(JSON.parse(event.data));
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };

            this.ws.onclose = () => {
                console.log('Disconnected from DAiW backend');
            };
        });
    }

    /**
     * Handle messages from backend
     */
    handleBackendMessage(message) {
        switch (message.type) {
            case 'audio':
                // Receive audio data from Python
                this.playAudioData(message.data, message.sampleRate);
                break;

            case 'command':
                this.executeCommand(message.command, message.params);
                break;

            case 'voiceModel':
                this.loadVoiceModel(message.data);
                break;
        }
    }

    /**
     * Execute a command from backend
     */
    executeCommand(command, params) {
        switch (command) {
            case 'setVowel':
                this.setVowel(params.vowel);
                break;
            case 'setPitch':
                this.setPitch(params.pitch);
                break;
            case 'noteOn':
                this.noteOn(params.note, params.velocity);
                break;
            case 'noteOff':
                this.noteOff();
                break;
            case 'setVolume':
                this.setVolume(params.volume);
                break;
        }
    }

    /**
     * Play audio data received from backend
     */
    async playAudioData(base64Data, sampleRate) {
        const bytes = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
        const floatArray = new Float32Array(bytes.buffer);

        const buffer = this.audioContext.createBuffer(1, floatArray.length, sampleRate);
        buffer.getChannelData(0).set(floatArray);

        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.gainNode);
        source.start();
    }

    /**
     * Load a voice model
     */
    loadVoiceModel(modelData) {
        this.voice = { ...DEFAULT_VOICE, ...modelData };
        this.updateVoiceParams();
    }

    /**
     * Update voice parameters in worklet
     */
    updateVoiceParams() {
        if (!this.workletNode) return;

        this.workletNode.port.postMessage({
            type: 'setVoiceParams',
            data: {
                vibratoRate: this.voice.vibratoRate,
                vibratoDepth: this.voice.vibratoDepth,
                jitter: this.voice.jitter,
                shimmer: this.voice.shimmer,
                breathiness: this.voice.breathiness,
                attackRate: 1 / (this.voice.attackTime * this.audioContext.sampleRate),
                releaseRate: 1 / (this.voice.releaseTime * this.audioContext.sampleRate)
            }
        });
    }

    /**
     * Set the current vowel
     */
    setVowel(vowel) {
        const formants = VOWEL_FORMANTS[vowel] || VOWEL_FORMANTS['a'];
        this.currentVowel = vowel;

        if (this.workletNode) {
            this.workletNode.port.postMessage({
                type: 'setFormants',
                data: {
                    f1: formants.f1,
                    f2: formants.f2,
                    f3: formants.f3
                }
            });
        }
    }

    /**
     * Set the pitch in Hz
     */
    setPitch(pitchHz) {
        this.currentPitch = pitchHz;

        if (this.workletNode) {
            this.workletNode.port.postMessage({
                type: 'setFrequency',
                data: { frequency: pitchHz }
            });
        }
    }

    /**
     * Set pitch from MIDI note number
     */
    setMidiNote(note) {
        const pitch = 440 * Math.pow(2, (note - 69) / 12);
        this.setPitch(pitch);
    }

    /**
     * Trigger note on
     */
    noteOn(midiNote = 60, velocity = 0.8) {
        // Resume audio context if suspended
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }

        this.setMidiNote(midiNote);

        if (this.workletNode) {
            this.workletNode.port.postMessage({
                type: 'noteOn',
                data: { velocity }
            });
        }
    }

    /**
     * Trigger note off
     */
    noteOff() {
        if (this.workletNode) {
            this.workletNode.port.postMessage({ type: 'noteOff' });
        }
    }

    /**
     * Set master volume
     */
    setVolume(volume) {
        if (this.gainNode) {
            this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
        }
    }

    /**
     * Set formant shift (for voice character modification)
     */
    setFormantShift(shift) {
        const formants = VOWEL_FORMANTS[this.currentVowel];
        if (formants && this.workletNode) {
            this.workletNode.port.postMessage({
                type: 'setFormants',
                data: {
                    f1: formants.f1 * shift,
                    f2: formants.f2 * shift,
                    f3: formants.f3 * shift
                }
            });
        }
    }

    /**
     * Speak text (simple implementation - sends to backend for processing)
     */
    async speak(text) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                command: 'speak',
                text: text
            }));
        } else {
            // Fallback: use browser TTS
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                speechSynthesis.speak(utterance);
            }
        }
    }

    /**
     * Play a sequence of vowels
     */
    async playVowelSequence(vowels, noteDuration = 0.5, tempo = 120) {
        const beatDuration = 60 / tempo;
        const duration = beatDuration * noteDuration * 1000;

        this.noteOn(60, 0.8);

        for (const vowel of vowels) {
            this.setVowel(vowel);
            await new Promise(resolve => setTimeout(resolve, duration));
        }

        this.noteOff();
    }

    /**
     * Create XY pad control for pitch and formant
     */
    createXYControl(element) {
        const handleMove = (x, y) => {
            // X controls formant shift (0.7 to 1.4)
            const formantShift = 0.7 + x * 0.7;
            this.setFormantShift(formantShift);

            // Y controls pitch (100Hz to 400Hz)
            const pitch = 100 + y * 300;
            this.setPitch(pitch);
        };

        element.addEventListener('mousemove', (e) => {
            if (e.buttons === 1) {
                const rect = element.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width;
                const y = 1 - (e.clientY - rect.top) / rect.height;
                handleMove(x, y);
            }
        });

        element.addEventListener('mousedown', () => this.noteOn(60, 0.8));
        element.addEventListener('mouseup', () => this.noteOff());
        element.addEventListener('mouseleave', () => this.noteOff());

        // Touch support
        element.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = element.getBoundingClientRect();
            const x = (touch.clientX - rect.left) / rect.width;
            const y = 1 - (touch.clientY - rect.top) / rect.height;
            handleMove(x, y);
        });

        element.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.noteOn(60, 0.8);
        });

        element.addEventListener('touchend', () => this.noteOff());
    }

    /**
     * Connect MIDI input
     */
    async connectMIDI() {
        if (!navigator.requestMIDIAccess) {
            console.warn('MIDI not supported in this browser');
            return;
        }

        try {
            const midiAccess = await navigator.requestMIDIAccess();

            midiAccess.inputs.forEach(input => {
                input.onmidimessage = (event) => {
                    const [status, note, velocity] = event.data;
                    const command = status >> 4;

                    if (command === 9 && velocity > 0) {
                        // Note on
                        this.noteOn(note, velocity / 127);
                    } else if (command === 8 || (command === 9 && velocity === 0)) {
                        // Note off
                        this.noteOff();
                    }
                };
            });

            console.log('MIDI connected');
        } catch (error) {
            console.error('MIDI connection failed:', error);
        }
    }

    /**
     * Clean up resources
     */
    dispose() {
        if (this.ws) {
            this.ws.close();
        }

        if (this.workletNode) {
            this.workletNode.disconnect();
        }

        if (this.gainNode) {
            this.gainNode.disconnect();
        }

        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}


// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DAiWVoiceSynth, VOWEL_FORMANTS, DEFAULT_VOICE };
}

// Global export for browser
if (typeof window !== 'undefined') {
    window.DAiWVoiceSynth = DAiWVoiceSynth;
}
