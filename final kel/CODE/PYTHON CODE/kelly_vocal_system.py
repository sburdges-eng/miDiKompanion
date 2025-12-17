"""
Kelly Vocal System - Unified Integration
Combines all vocal/lyric modules with VAD and quantum emotional fields
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json

# Import all Kelly vocal modules
from kelly_lyric_structures import (
    LyricLine, LyricSection, SongLyrics, VocalTrack, VocalPhrase, VocalNote,
    VocalExpression, Phoneme, Syllable, SectionType, VocalStyleType
)
from kelly_phoneme_processor import (
    G2PConverter, SyllableSegmenter, text_to_phonemes, text_to_syllables,
    text_to_words, apply_coarticulation
)
from kelly_lyric_generator import (
    LyricGenerator, SectionGenerator, ProsodyAnalyzer,
    generate_lyrics, analyze_lyrics, EMOTION_VOCABULARY
)
from kelly_vocal_expression import (
    ExpressionEngine, LyriSync, PitchContourGenerator,
    create_vocal_track, get_style_preset, VOCAL_STYLE_PRESETS
)
from kelly_quantum_voice_field import (
    VADVector, BiometricData, QuantumEmotionState, EmotionBasis,
    QuantumEmotionalVoiceField, VoiceParameters,
    biometric_to_vad, vad_to_voice, circadian_adjustment,
    calculate_resonance, analyze_trends,
    ResonanceMetrics, TrendMetrics
)


# =============================================================================
# UNIFIED VOCAL SYSTEM
# =============================================================================

@dataclass
class VocalSystemConfig:
    """Configuration for the vocal system."""
    base_frequency: float = 200.0
    tempo: int = 120
    key: str = "C"
    enable_quantum: bool = True
    enable_biometrics: bool = False
    enable_circadian: bool = False


@dataclass
class VocalGenerationResult:
    """Complete result from vocal generation."""
    lyrics: SongLyrics
    vocal_track: VocalTrack
    vad: VADVector
    voice_params: VoiceParameters
    resonance: ResonanceMetrics
    trends: TrendMetrics
    quantum_state: Optional[QuantumEmotionState] = None


class KellyVocalSystem:
    """
    Unified Kelly Vocal System.
    
    Integrates:
    - Lyric generation (emotion → words)
    - Phoneme processing (words → phonemes)
    - Vocal synthesis (phonemes → voice params)
    - VAD emotion mapping
    - Quantum emotional fields
    - Biometric integration
    """
    
    def __init__(self, config: VocalSystemConfig = None):
        self.config = config or VocalSystemConfig()
        
        # Core components
        self.lyric_generator = LyricGenerator()
        self.g2p = G2PConverter()
        self.segmenter = SyllableSegmenter(self.g2p)
        self.prosody_analyzer = ProsodyAnalyzer()
        self.expression_engine = ExpressionEngine()
        self.lyrisync = LyriSync(tempo=self.config.tempo)
        
        # Quantum emotional field
        self.qevf = QuantumEmotionalVoiceField(f_base=self.config.base_frequency)
        
        # State
        self.current_vad: Optional[VADVector] = None
        self.current_biometrics: Optional[BiometricData] = None
        self.vad_history: List[VADVector] = []
    
    # -------------------------------------------------------------------------
    # EMOTION INPUT
    # -------------------------------------------------------------------------
    
    def set_emotion_from_wound(
        self,
        wound_description: str,
        intensity: float = 1.0,
        vulnerability: float = 0.7
    ) -> VADVector:
        """Set emotional state from wound description (Kelly Phase 0)."""
        # Analyze wound for emotion keywords
        wound_lower = wound_description.lower()
        
        # Map wound keywords to VAD
        if any(w in wound_lower for w in ["death", "loss", "gone", "died", "grief"]):
            vad = VADVector(valence=-0.7 * intensity, arousal=0.3, dominance=-0.4 * vulnerability)
            emotion = EmotionBasis.SADNESS
        elif any(w in wound_lower for w in ["betrayal", "anger", "rage", "fury"]):
            vad = VADVector(valence=-0.5 * intensity, arousal=0.8, dominance=0.3)
            emotion = EmotionBasis.ANGER
        elif any(w in wound_lower for w in ["fear", "scared", "terror", "panic"]):
            vad = VADVector(valence=-0.6 * intensity, arousal=0.8, dominance=-0.5)
            emotion = EmotionBasis.FEAR
        elif any(w in wound_lower for w in ["longing", "miss", "yearn", "want"]):
            vad = VADVector(valence=-0.3 * intensity, arousal=0.4, dominance=-0.2)
            emotion = EmotionBasis.ANTICIPATION
        elif any(w in wound_lower for w in ["joy", "happy", "love", "hope"]):
            vad = VADVector(valence=0.7 * intensity, arousal=0.6, dominance=0.3)
            emotion = EmotionBasis.JOY
        else:
            # Default: melancholy
            vad = VADVector(valence=-0.4 * intensity, arousal=0.4, dominance=-0.2)
            emotion = EmotionBasis.SADNESS
        
        self.qevf.set_emotion(emotion, intensity)
        self.current_vad = vad
        self._record_vad(vad)
        
        return vad
    
    def set_vad_direct(self, valence: float, arousal: float, dominance: float) -> VADVector:
        """Set VAD state directly."""
        vad = VADVector(valence, arousal, dominance)
        self.qevf.set_vad(vad)
        self.current_vad = vad
        self._record_vad(vad)
        return vad
    
    def set_from_biometrics(self, bio: BiometricData) -> VADVector:
        """Set emotional state from biometric data."""
        self.current_biometrics = bio
        vad = biometric_to_vad(bio)
        
        if self.current_vad:
            # Blend with existing emotional state (70% emotion, 30% bio)
            vad = self.current_vad.blend(vad, 0.3)
        
        self.qevf.set_vad(vad)
        self.current_vad = vad
        self._record_vad(vad)
        
        return vad
    
    def apply_circadian_adjustment(self, hour: int, day_of_week: int = 0):
        """Apply time-of-day adjustments."""
        if self.config.enable_circadian:
            self.qevf.apply_circadian(hour, day_of_week)
            self.current_vad = self.qevf.quantum_state.expected_vad()
    
    # -------------------------------------------------------------------------
    # LYRIC GENERATION
    # -------------------------------------------------------------------------
    
    def generate_lyrics(
        self,
        title: str = "Untitled",
        structure: str = "VCVC",
        emotion_override: str = None
    ) -> SongLyrics:
        """Generate lyrics based on current emotional state."""
        if emotion_override:
            emotion = emotion_override
        elif self.current_vad:
            emotion = self._vad_to_emotion_name(self.current_vad)
        else:
            emotion = "longing"
        
        lyrics = self.lyric_generator.generate(
            title=title,
            emotion=emotion,
            structure=structure,
            tempo=self.config.tempo,
            key=self.config.key
        )
        
        # Apply emotional VAD to each line
        if self.current_vad:
            for section in lyrics.sections:
                for line in section.lines:
                    line.emotion_valence = self.current_vad.valence
                    line.emotion_arousal = self.current_vad.arousal
                    line.emotion_dominance = self.current_vad.dominance
        
        return lyrics
    
    # -------------------------------------------------------------------------
    # VOCAL SYNTHESIS
    # -------------------------------------------------------------------------
    
    def create_vocal_track(
        self,
        lyrics: SongLyrics,
        melody_pitches: List[List[int]]
    ) -> VocalTrack:
        """Create vocal track from lyrics and melody."""
        vad = self.current_vad or VADVector()
        
        return create_vocal_track(
            lines=lyrics.all_lines,
            melody_pitches=melody_pitches,
            emotion_valence=vad.valence,
            emotion_arousal=vad.arousal,
            emotion_dominance=vad.dominance,
            tempo=self.config.tempo
        )
    
    def get_voice_parameters(self) -> VoiceParameters:
        """Get current voice synthesis parameters."""
        return self.qevf.get_voice_params()
    
    def get_vocal_style(self) -> VocalStyleType:
        """Determine vocal style from VAD."""
        if not self.current_vad:
            return VocalStyleType.NATURAL
        
        V, A, D = self.current_vad.valence, self.current_vad.arousal, self.current_vad.dominance
        
        if V < -0.5 and A < 0.4:
            return VocalStyleType.CRY
        elif V < -0.3 and D < -0.3:
            return VocalStyleType.BREATHY
        elif A > 0.7 and D > 0.3:
            return VocalStyleType.BELT
        elif A < 0.3 and V < 0:
            return VocalStyleType.WHISPER
        elif V > 0.5 and A > 0.5:
            return VocalStyleType.NATURAL
        else:
            return VocalStyleType.NATURAL
    
    # -------------------------------------------------------------------------
    # FULL PIPELINE
    # -------------------------------------------------------------------------
    
    def generate_full(
        self,
        wound_description: str,
        title: str = None,
        melody_pitches: List[List[int]] = None,
        structure: str = "VCVC"
    ) -> VocalGenerationResult:
        """
        Full generation pipeline:
        Wound → VAD → Lyrics → Phonemes → Vocal Track
        """
        # 1. Set emotion from wound
        vad = self.set_emotion_from_wound(wound_description)
        
        # 2. Generate title if not provided
        if not title:
            words = wound_description.split()[:3]
            title = " ".join(words).title() if words else "Untitled"
        
        # 3. Generate lyrics
        lyrics = self.generate_lyrics(title=title, structure=structure)
        
        # 4. Default melody if not provided
        if not melody_pitches:
            melody_pitches = self._generate_default_melody(len(lyrics.all_lines))
        
        # 5. Create vocal track
        vocal_track = self.create_vocal_track(lyrics, melody_pitches)
        
        # 6. Get metrics
        voice_params = self.get_voice_parameters()
        resonance = self.qevf.get_resonance()
        trends = self.qevf.get_trends()
        
        return VocalGenerationResult(
            lyrics=lyrics,
            vocal_track=vocal_track,
            vad=vad,
            voice_params=voice_params,
            resonance=resonance,
            trends=trends,
            quantum_state=self.qevf.quantum_state if self.config.enable_quantum else None
        )
    
    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------
    
    def get_resonance(self) -> ResonanceMetrics:
        """Get current resonance metrics."""
        biometric_vad = None
        if self.current_biometrics:
            biometric_vad = biometric_to_vad(self.current_biometrics)
        return self.qevf.get_resonance(biometric_vad)
    
    def get_trends(self) -> TrendMetrics:
        """Get VAD trend analysis."""
        return analyze_trends(self.vad_history)
    
    def get_quantum_state(self) -> Dict:
        """Get quantum state info."""
        if not self.config.enable_quantum:
            return {}
        
        state = self.qevf.quantum_state
        return {
            "probabilities": {e.value: p for e, p in state.probabilities().items()},
            "coherence": state.coherence(),
            "entropy": state.entropy(),
            "expected_vad": {
                "valence": state.expected_vad().valence,
                "arousal": state.expected_vad().arousal,
                "dominance": state.expected_vad().dominance
            }
        }
    
    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    
    def _vad_to_emotion_name(self, vad: VADVector) -> str:
        """Convert VAD to emotion name for lyric generator."""
        V, A, D = vad.valence, vad.arousal, vad.dominance
        
        if V < -0.5:
            if A > 0.6:
                return "anger"
            elif D < -0.3:
                return "grief"
            else:
                return "fear"
        elif V > 0.3:
            if A > 0.5:
                return "joy"
            else:
                return "hope"
        else:
            if A > 0.6:
                return "defiance"
            else:
                return "longing"
    
    def _generate_default_melody(self, num_lines: int) -> List[List[int]]:
        """Generate default melody pitches for given number of lines."""
        # Simple pentatonic patterns
        patterns = [
            [60, 62, 64, 67, 69, 67, 64, 62],  # C D E G A G E D
            [67, 64, 62, 60, 62, 64, 67, 69],  # G E D C D E G A
            [62, 64, 67, 69, 67, 64, 62, 60],  # D E G A G E D C
            [69, 67, 64, 62, 64, 67, 69, 72],  # A G E D E G A C
        ]
        return [patterns[i % len(patterns)] for i in range(num_lines)]
    
    def _record_vad(self, vad: VADVector):
        """Record VAD to history."""
        self.vad_history.append(vad)
        if len(self.vad_history) > 50:
            self.vad_history = self.vad_history[-50:]


# =============================================================================
# CONVENIENCE API
# =============================================================================

_system: Optional[KellyVocalSystem] = None

def get_vocal_system(config: VocalSystemConfig = None) -> KellyVocalSystem:
    """Get or create the global vocal system."""
    global _system
    if _system is None:
        _system = KellyVocalSystem(config)
    return _system


def generate_from_wound(
    wound: str,
    title: str = None,
    tempo: int = 120
) -> VocalGenerationResult:
    """Quick API: Generate everything from wound description."""
    system = get_vocal_system(VocalSystemConfig(tempo=tempo))
    return system.generate_full(wound, title=title)


def process_text_to_phonemes(text: str) -> List[Dict]:
    """Quick API: Convert text to phoneme data."""
    phonemes = text_to_phonemes(text)
    return [
        {
            "ipa": p.ipa,
            "arpabet": p.arpabet,
            "duration_ms": p.duration_ms,
            "is_vowel": p.is_vowel,
            "formants": p.formants
        }
        for p in phonemes
    ]


def vad_to_voice_params(v: float, a: float, d: float) -> Dict:
    """Quick API: Convert VAD to voice parameters."""
    vad = VADVector(v, a, d)
    params = vad_to_voice(vad)
    return {
        "f0": params.f0_modulated,
        "vibrato_rate": params.vibrato_rate,
        "vibrato_depth": params.vibrato_depth,
        "formants": params.formants,
        "brightness": params.brightness,
        "amplitude": params.amplitude,
        "jitter": params.jitter,
        "shimmer": params.shimmer,
        "speech_rate": params.speech_rate
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=== Kelly Vocal System Integration Test ===\n")
    
    # Test full pipeline with "When I Found You Sleeping"
    result = generate_from_wound(
        wound="Finding Kelly's body, the moment everything changed",
        title="When I Found You Sleeping",
        tempo=82
    )
    
    print(f"Title: {result.lyrics.title}")
    print(f"VAD: V={result.vad.valence:.2f}, A={result.vad.arousal:.2f}, D={result.vad.dominance:.2f}")
    print(f"Vocal style: {result.voice_params.f0_modulated:.1f} Hz")
    print(f"Resonance: {result.resonance.coherence:.3f}")
    
    print("\n--- Generated Lyrics ---")
    print(result.lyrics.full_text[:500] + "...")
    
    print("\n--- Voice Parameters ---")
    vp = result.voice_params
    print(f"  F0: {vp.f0_modulated:.1f} Hz")
    print(f"  Vibrato: {vp.vibrato_rate:.1f} Hz @ {vp.vibrato_depth:.1f} st")
    print(f"  Brightness: {vp.brightness:.2f}")
    print(f"  Jitter: {vp.jitter:.3f}")
    
    print("\n--- Vocal Track ---")
    print(f"  Phrases: {len(result.vocal_track.phrases)}")
    print(f"  Total notes: {len(result.vocal_track.all_notes)}")
    print(f"  Duration: {result.vocal_track.duration:.2f}s")
    
    if result.quantum_state:
        print("\n--- Quantum State ---")
        probs = result.quantum_state.probabilities()
        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, prob in top3:
            print(f"  {emotion.value}: {prob:.2%}")
