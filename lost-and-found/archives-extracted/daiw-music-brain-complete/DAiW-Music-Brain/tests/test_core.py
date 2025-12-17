"""
DAiW Music Brain - Test Suite
"""

import pytest
from pathlib import Path


class TestImports:
    """Test that all modules import correctly."""
    
    def test_import_main_package(self):
        import music_brain
        assert hasattr(music_brain, '__version__')
    
    def test_import_models(self):
        from music_brain.structure.models import (
            CoreWoundModel,
            IntentModel,
            ConstraintModel,
            RuleBreakModel,
            NarrativeArc,
            VulnerabilityScale,
        )
        assert CoreWoundModel is not None
    
    def test_import_comprehensive_engine(self):
        from music_brain.structure.comprehensive_engine import (
            TherapySession,
            AffectAnalyzer,
            HarmonyPlan,
        )
        assert TherapySession is not None
    
    def test_import_chord_module(self):
        from music_brain.modules.chord import (
            Chord,
            generate_progression,
        )
        assert Chord is not None
    
    def test_import_groove_engine(self):
        from music_brain.groove.engine import (
            GrooveTemplate,
            GrooveApplicator,
            drunken_drummer,
        )
        assert GrooveTemplate is not None
    
    def test_import_vernacular(self):
        from music_brain.session.vernacular import (
            VernacularTranslator,
            translate_vernacular,
        )
        assert VernacularTranslator is not None


class TestAffectAnalyzer:
    """Test the affect analyzer."""
    
    def test_grief_detection(self):
        from music_brain.structure.comprehensive_engine import AffectAnalyzer
        
        analyzer = AffectAnalyzer()
        result = analyzer.analyze("I found you sleeping and you were cold")
        
        assert result.primary == "grief"
        assert result.intensity > 0
    
    def test_rage_detection(self):
        from music_brain.structure.comprehensive_engine import AffectAnalyzer
        
        analyzer = AffectAnalyzer()
        result = analyzer.analyze("I hate everything and want to destroy it all")
        
        assert result.primary == "rage"
    
    def test_neutral_detection(self):
        from music_brain.structure.comprehensive_engine import AffectAnalyzer
        
        analyzer = AffectAnalyzer()
        result = analyzer.analyze("")
        
        assert result.primary == "neutral"


class TestTherapySession:
    """Test the therapy session."""
    
    def test_session_creation(self):
        from music_brain.structure.comprehensive_engine import TherapySession
        
        session = TherapySession()
        assert session.state.primary_affect == "neutral"
    
    def test_process_input(self):
        from music_brain.structure.comprehensive_engine import TherapySession
        
        session = TherapySession()
        mood = session.process_core_input("I feel broken and lost")
        
        assert mood in ["grief", "dissociation", "confusion", "neutral"]
        assert session.state.suggested_mode in ["aeolian", "locrian", "dorian", "ionian"]
    
    def test_generate_plan(self):
        from music_brain.structure.comprehensive_engine import TherapySession
        
        session = TherapySession()
        session.process_core_input("I miss you so much")
        session.set_scales(motivation=7, chaos=0.3)
        
        plan = session.generate_plan()
        
        assert plan.length_bars > 0
        assert plan.tempo_bpm > 0
        assert plan.mode in ["ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]


class TestChordGeneration:
    """Test chord generation."""
    
    def test_generate_progression(self):
        from music_brain.modules.chord import generate_progression
        
        payload = {
            "mood_primary": "grief",
            "narrative_arc": "Climb-to-Climax",
            "technical_key": "F",
            "technical_mode": "major",
            "song_length_bars": 16,
        }
        
        chords = generate_progression(payload)
        
        assert len(chords) > 0
        assert all(hasattr(c, 'root') for c in chords)
        assert all(hasattr(c, 'midi_notes') for c in chords)


class TestProgressionAnalysis:
    """Test chord progression analysis."""
    
    def test_analyze_progression(self):
        from music_brain.structure.progression import analyze_progression
        
        result = analyze_progression("F - C - Am - Dm")
        
        assert "estimated_key" in result
        assert "roman_numerals" in result
        assert result["num_chords"] == 4
    
    def test_parse_chord(self):
        from music_brain.structure.progression import parse_chord
        
        chord = parse_chord("Am7")
        
        assert chord is not None
        assert chord.root == "A"
        assert "m" in chord.quality or "min" in chord.quality


class TestVernacularTranslation:
    """Test vernacular translation."""
    
    def test_translate_fat(self):
        from music_brain.session.vernacular import translate_vernacular
        
        params = translate_vernacular("fat")
        
        assert "eq.low_mid" in params or "saturation" in params
    
    def test_translate_boom_bap(self):
        from music_brain.session.vernacular import translate_vernacular
        
        params = translate_vernacular("boom bap")
        
        assert "groove.pattern" in params
    
    def test_emotion_to_rule_break(self):
        from music_brain.session.vernacular import VernacularTranslator
        
        translator = VernacularTranslator()
        rules = translator.get_rule_breaks_for_emotion("grief")
        
        assert len(rules) > 0
        assert any("STRUCTURE" in r or "PRODUCTION" in r for r in rules)


class TestGrooveEngine:
    """Test groove engine."""
    
    def test_groove_presets(self):
        from music_brain.groove.engine import GROOVE_PRESETS, EMOTIONAL_PRESETS
        
        assert "funk" in GROOVE_PRESETS
        assert "boom_bap" in GROOVE_PRESETS
        assert "grief" in EMOTIONAL_PRESETS
    
    def test_drunken_drummer(self):
        from music_brain.groove.engine import drunken_drummer, MidiNoteEvent
        
        events = [
            MidiNoteEvent(start_tick=0, duration_tick=100, pitch=60, velocity=100),
            MidiNoteEvent(start_tick=480, duration_tick=100, pitch=62, velocity=100),
        ]
        
        humanized = drunken_drummer(events, vulnerability_scale=0.5, seed=42)
        
        assert len(humanized) == len(events)


class TestTensionCurve:
    """Test tension curve generation."""
    
    def test_climb_curve(self):
        from music_brain.structure.tension import generate_tension_curve
        
        curve = generate_tension_curve(32, "climb")
        
        assert len(curve) == 32
        assert curve[0] < curve[-1]  # Should increase
    
    def test_constant_curve(self):
        from music_brain.structure.tension import generate_tension_curve
        
        curve = generate_tension_curve(16, "constant")
        
        assert len(curve) == 16
        assert all(v == 1.0 for v in curve)


class TestModels:
    """Test data models."""
    
    def test_create_example_payload(self):
        from music_brain.structure.models import create_example_payload
        
        payload = create_example_payload()
        
        assert payload.session_id is not None
        assert payload.wound.core_event != ""
        assert payload.intent.mood_primary != ""
    
    def test_payload_to_dict(self):
        from music_brain.structure.models import create_example_payload
        
        payload = create_example_payload()
        d = payload.to_dict()
        
        assert "core_event" in d
        assert "mood_primary" in d
        assert "technical_key" in d
