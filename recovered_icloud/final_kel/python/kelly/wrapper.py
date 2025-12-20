"""
High-level Python wrapper for Kelly MIDI Companion

Provides convenient Pythonic interfaces and utilities.
"""

from typing import List, Optional, Dict, Tuple, Union
from . import (
    KellyBrain,
    Wound,
    EmotionNode,
    IntentResult,
    MidiNote,
    Chord,
    GeneratedMidi,
    EmotionCategory,
)


class Kelly:
    """
    High-level Python interface for Kelly MIDI Companion.
    
    Example:
        >>> kelly = Kelly(tempo=120)
        >>> result, midi = kelly.generate("feeling of loss", intensity=0.8)
        >>> print(f"Generated {len(midi)} MIDI notes")
        >>> kelly.export_midi(midi, "output.mid")
    """
    
    def __init__(self, tempo: int = 120, seed: Optional[int] = None):
        """
        Initialize Kelly MIDI generator.
        
        Args:
            tempo: Default tempo in BPM
            seed: Random seed (None for random)
        """
        self.brain = KellyBrain(tempo, seed if seed is not None else 0)
        self._tempo = tempo
    
    @property
    def tempo(self) -> int:
        """Get current tempo."""
        return self._tempo
    
    @tempo.setter
    def tempo(self, value: int):
        """Set tempo."""
        self._tempo = value
        self.brain.tempo = value
    
    def generate(
        self,
        description: str,
        intensity: float = 0.7,
        source: str = "internal",
        bars: int = 4
    ) -> Tuple[IntentResult, List[MidiNote]]:
        """
        Generate MIDI from emotional description.
        
        Args:
            description: Text description of emotional state
            intensity: Emotional intensity (0.0-1.0)
            source: "internal" or "external"
            bars: Number of bars to generate
            
        Returns:
            Tuple of (IntentResult, list of MidiNote)
        """
        return self.brain.quick_generate(description, intensity, bars)
    
    def generate_from_emotion(
        self,
        emotion_name: str,
        bars: int = 4
    ) -> Optional[Tuple[IntentResult, List[MidiNote]]]:
        """
        Generate MIDI from emotion name.
        
        Args:
            emotion_name: Name of emotion (e.g., "grief", "joy")
            bars: Number of bars to generate
            
        Returns:
            Tuple of (IntentResult, list of MidiNote) or None if emotion not found
        """
        emotion = self.brain.find_emotion_by_name(emotion_name)
        if emotion is None:
            return None
        
        # Create wound from emotion
        wound = Wound(emotion_name, emotion.intensity)
        result = self.brain.process_wound(wound.description, wound.intensity)
        midi = self.brain.generate_midi(result, bars)
        return result, midi
    
    def generate_from_vai(
        self,
        valence: float,
        arousal: float,
        intensity: float,
        key: str = "C",
        mode: str = "Aeolian",
        bars: int = 4
    ) -> List[MidiNote]:
        """
        Generate MIDI from valence/arousal/intensity values.
        
        Args:
            valence: -1.0 (negative) to 1.0 (positive)
            arousal: 0.0 (calm) to 1.0 (excited)
            intensity: 0.0 (subtle) to 1.0 (extreme)
            key: Musical key (e.g., "C", "D", "F#")
            mode: Mode (e.g., "Aeolian", "Ionian", "Dorian")
            bars: Number of bars to generate
            
        Returns:
            List of MidiNote
        """
        return self.brain.generate_from_vai(valence, arousal, intensity, key, mode, bars)
    
    def find_emotion(
        self,
        valence: float,
        arousal: float,
        intensity: float
    ) -> Optional[EmotionNode]:
        """Find nearest emotion for given VAI values."""
        return self.brain.find_emotion(valence, arousal, intensity)
    
    def find_emotion_by_name(self, name: str) -> Optional[EmotionNode]:
        """Find emotion by name."""
        return self.brain.find_emotion_by_name(name)
    
    def get_emotions_by_category(self, category: EmotionCategory) -> List[EmotionNode]:
        """Get all emotions in a category."""
        return self.brain.get_emotions_by_category(category)
    
    def export_midi(
        self,
        notes: List[MidiNote],
        filename: str,
        tempo: Optional[int] = None
    ) -> bool:
        """
        Export MIDI notes to a MIDI file.
        
        Args:
            notes: List of MidiNote objects
            filename: Output filename
            tempo: Tempo in BPM (uses instance tempo if None)
            
        Returns:
            True if successful
        """
        try:
            from mido import MidiFile, MidiTrack, Message
            
            mid = MidiFile()
            track = MidiTrack()
            mid.tracks.append(track)
            
            # Set tempo
            tempo_bpm = tempo or self.tempo
            tempo_microseconds = int(60000000 / tempo_bpm)
            track.append(Message('set_tempo', tempo=tempo_microseconds))
            
            # Convert beats to ticks (assuming 480 ticks per quarter note)
            ticks_per_beat = 480
            ticks_per_second = ticks_per_beat * (tempo_bpm / 60.0)
            
            # Sort notes by start time
            sorted_notes = sorted(notes, key=lambda n: n.startBeat)
            
            current_tick = 0
            for note in sorted_notes:
                # Calculate tick position
                note_tick = int(note.startBeat * ticks_per_beat)
                delta = note_tick - current_tick
                
                # Note on
                track.append(Message('note_on', note=note.pitch, velocity=note.velocity, time=delta))
                
                # Note off
                duration_ticks = int(note.duration * ticks_per_beat)
                track.append(Message('note_off', note=note.pitch, velocity=0, time=duration_ticks))
                
                current_tick = note_tick + duration_ticks
            
            mid.save(filename)
            return True
        except ImportError:
            print("Warning: mido library not installed. Install with: pip install mido")
            return False
        except Exception as e:
            print(f"Error exporting MIDI: {e}")
            return False
    
    def to_dict(self, result: IntentResult) -> Dict:
        """Convert IntentResult to dictionary."""
        return {
            "wound": {
                "description": result.wound.description,
                "intensity": result.wound.intensity,
                "source": result.wound.source,
            },
            "emotion": {
                "id": result.emotion.id,
                "name": result.emotion.name,
                "category": str(result.emotion.category),
                "valence": result.emotion.valence,
                "arousal": result.emotion.arousal,
                "intensity": result.emotion.intensity,
            },
            "rule_breaks": [
                {
                    "type": str(rb.type),
                    "severity": rb.severity,
                    "description": rb.description,
                    "emotional_justification": rb.emotionalJustification,
                }
                for rb in result.ruleBreaks
            ],
            "musical_params": {
                "tempo": result.musicalParams.tempoSuggested,
                "key": result.musicalParams.keySuggested,
                "mode": result.musicalParams.modeSuggested,
                "dissonance": result.musicalParams.dissonance,
                "density": result.musicalParams.density,
            },
        }
    
    def clear_history(self):
        """Clear wound and rule-break history."""
        self.brain.clear_history()


# Convenience function
def create_kelly(tempo: int = 120, seed: Optional[int] = None) -> Kelly:
    """Create a Kelly instance."""
    return Kelly(tempo, seed)
