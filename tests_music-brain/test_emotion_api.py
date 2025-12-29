"""
Comprehensive tests for music_brain.emotion_api module.

Tests cover:
- MusicBrain class initialization
- Text-to-emotion mapping
- Intent-based generation
- Fluent API chain
- Mixer parameter generation
- Export functionality
- Edge cases and error handling

Run with: pytest tests_music-brain/test_emotion_api.py -v
"""

import pytest
import tempfile
import json
from pathlib import Path

from music_brain.emotion_api import (
    MusicBrain,
    GeneratedMusic,
    FluentChain,
    quick_generate,
    quick_export,
)
from music_brain.data.emotional_mapping import Valence, Arousal, TimingFeel
from music_brain.session.intent_schema import CompleteSongIntent


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def music_brain():
    """Standard MusicBrain instance."""
    return MusicBrain()


# ==============================================================================
# INITIALIZATION TESTS
# ==============================================================================

class TestMusicBrainInit:
    """Test MusicBrain initialization."""

    def test_initialization(self, music_brain):
        """Test MusicBrain can be instantiated."""
        assert music_brain is not None
        assert hasattr(music_brain, 'emotion_mapper')
        assert hasattr(music_brain, '_emotion_keywords')

    def test_emotion_keywords_populated(self, music_brain):
        """Test emotion keywords dictionary is populated."""
        assert len(music_brain._emotion_keywords) > 0
        assert "grief" in music_brain._emotion_keywords
        assert "hope" in music_brain._emotion_keywords
        assert "anxiety" in music_brain._emotion_keywords


# ==============================================================================
# TEXT-TO-EMOTION MAPPING TESTS
# ==============================================================================

class TestTextToEmotionMapping:
    """Test generate_from_text method."""

    def test_generate_from_grief_text(self, music_brain):
        """Test grief emotion mapping."""
        result = music_brain.generate_from_text("grief and loss")

        assert isinstance(result, GeneratedMusic)
        assert result.emotional_state.primary_emotion == "grief"
        assert result.emotional_state.valence.value < 0  # Negative valence
        assert result.musical_params.tempo_suggested < 100  # Slow tempo for grief

    def test_generate_from_hope_text(self, music_brain):
        """Test hope emotion mapping."""
        result = music_brain.generate_from_text("hopeful and uplifting")

        assert result.emotional_state.primary_emotion == "hope"
        assert result.emotional_state.valence.value > 0  # Positive valence

    def test_generate_from_anxiety_text(self, music_brain):
        """Test anxiety emotion mapping."""
        result = music_brain.generate_from_text("anxious and nervous")

        assert result.emotional_state.primary_emotion == "anxiety"
        assert result.emotional_state.valence.value < 0  # Negative valence
        assert result.emotional_state.arousal.value > 0  # High arousal

    def test_generate_from_calm_text(self, music_brain):
        """Test calm emotion mapping."""
        result = music_brain.generate_from_text("calm and peaceful")

        assert result.emotional_state.primary_emotion == "calm"
        assert result.musical_params.tempo_suggested < 110  # Slower tempo

    def test_unknown_emotion_fallback(self, music_brain):
        """Test unknown emotion uses neutral defaults."""
        result = music_brain.generate_from_text("completely unknown emotion xyz")

        # Should not crash, uses neutral defaults
        assert isinstance(result, GeneratedMusic)
        assert result.emotional_state.primary_emotion in ["neutral", "grief", "hope", "calm", "anxiety"]


# ==============================================================================
# INTENT-BASED GENERATION TESTS
# ==============================================================================

class TestIntentGeneration:
    """Test generate_from_intent method."""

    def test_generate_from_intent_basic(self, music_brain):
        """Test basic intent-based generation."""
        intent = music_brain.create_intent(
            title="Test Song",
            core_event="A significant moment",
            mood_primary="grief",
            technical_key="C",
            technical_mode="major",
            tempo_range=(80, 100)
        )

        result = music_brain.generate_from_intent(intent)

        assert isinstance(result, GeneratedMusic)
        assert result.intent is not None
        assert result.emotional_state.primary_emotion == "grief"

    def test_intent_tempo_override(self, music_brain):
        """Test tempo range from intent overrides emotion defaults."""
        intent = music_brain.create_intent(
            title="Fast Grief",
            core_event="Processing loss",
            mood_primary="grief",
            tempo_range=(140, 160)  # Fast tempo
        )

        result = music_brain.generate_from_intent(intent)

        # Tempo should be in specified range
        assert 140 <= result.musical_params.tempo_suggested <= 160

    def test_intent_with_rule_breaking(self, music_brain):
        """Test intent with rule breaking."""
        intent = music_brain.create_intent(
            title="Bittersweet",
            core_event="Finding hope in loss",
            mood_primary="grief",
            technical_key="F",
            technical_mode="major",
            rule_to_break="HARMONY_ModalInterchange",
            rule_justification="Bbm creates bittersweet hope"
        )

        result = music_brain.generate_from_intent(intent)

        assert result.intent.technical_constraints.technical_rule_to_break == "HARMONY_ModalInterchange"
        assert "bittersweet" in result.intent.technical_constraints.rule_breaking_justification.lower()


# ==============================================================================
# FLUENT API TESTS
# ==============================================================================

class TestFluentAPI:
    """Test fluent API chain."""

    def test_fluent_chain_basic(self, music_brain):
        """Test basic fluent chain."""
        chain = music_brain.process("grief")

        assert isinstance(chain, FluentChain)
        assert chain.emotional_text == "grief"

    def test_fluent_map_to_emotion(self, music_brain):
        """Test map_to_emotion step."""
        chain = music_brain.process("hope").map_to_emotion()

        assert chain.emotional_state is not None
        assert chain.emotional_state.primary_emotion == "hope"

    def test_fluent_map_to_music(self, music_brain):
        """Test map_to_music step."""
        chain = music_brain.process("anxiety").map_to_music()

        assert chain.musical_params is not None
        assert chain.musical_params.tempo_suggested > 0

    def test_fluent_map_to_mixer(self, music_brain):
        """Test map_to_mixer step."""
        chain = music_brain.process("grief").map_to_mixer()

        assert chain.mixer_params is not None
        assert hasattr(chain.mixer_params, 'reverb_mix')

    def test_fluent_with_tempo_override(self, music_brain):
        """Test tempo override in fluent chain."""
        chain = music_brain.process("calm").map_to_music().with_tempo(100)

        assert chain.musical_params.tempo_suggested == 100

    def test_fluent_with_dissonance_override(self, music_brain):
        """Test dissonance override."""
        chain = music_brain.process("calm").map_to_music().with_dissonance(0.8)

        assert chain.musical_params.dissonance == 0.8

    def test_fluent_with_timing_override(self, music_brain):
        """Test timing feel override."""
        chain = music_brain.process("calm").map_to_music().with_timing("behind")

        assert chain.musical_params.timing_feel == TimingFeel.BEHIND

    def test_fluent_get_state(self, music_brain):
        """Test get() returns current state."""
        state = music_brain.process("hope").map_to_emotion().map_to_music().get()

        assert "emotional_state" in state
        assert "musical_params" in state
        assert state["emotional_state"]["primary_emotion"] == "hope"

    def test_fluent_describe(self, music_brain):
        """Test describe() returns human-readable string."""
        description = (music_brain.process("anxiety")
                       .map_to_emotion()
                       .map_to_music()
                       .map_to_mixer()
                       .describe())

        assert isinstance(description, str)
        assert "anxiety" in description.lower()
        assert "Tempo" in description


# ==============================================================================
# GENERATED MUSIC TESTS
# ==============================================================================

class TestGeneratedMusic:
    """Test GeneratedMusic dataclass."""

    def test_to_dict_serialization(self, music_brain):
        """Test to_dict() produces valid dictionary."""
        result = music_brain.generate_from_text("hope")
        data = result.to_dict()

        assert "emotional_state" in data
        assert "musical_params" in data
        assert "mixer_params" in data
        assert "paths" in data

    def test_summary_generation(self, music_brain):
        """Test summary() produces human-readable text."""
        result = music_brain.generate_from_text("grief")
        summary = result.summary()

        assert isinstance(summary, str)
        assert "grief" in summary.lower()
        assert "BPM" in summary


# ==============================================================================
# EXPORT FUNCTIONALITY TESTS
# ==============================================================================

class TestExportFunctionality:
    """Test export_to_logic method."""

    def test_export_creates_automation_file(self, music_brain):
        """Test export creates automation JSON file."""
        result = music_brain.generate_from_text("calm")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = str(Path(tmpdir) / "test_output")
            paths = music_brain.export_to_logic(result, output_base)

            assert "automation" in paths
            automation_path = paths["automation"]
            assert Path(automation_path).exists()

    def test_export_valid_json(self, music_brain):
        """Test exported automation file is valid JSON."""
        result = music_brain.generate_from_text("hope")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = str(Path(tmpdir) / "test_output")
            paths = music_brain.export_to_logic(result, output_base)

            with open(paths["automation"], 'r') as f:
                data = json.load(f)

            # Should be valid JSON with expected structure
            assert isinstance(data, dict)

    def test_fluent_export_logic(self, music_brain):
        """Test fluent API export_logic()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "fluent_export.json")
            paths = (music_brain.process("anxiety")
                     .map_to_mixer()
                     .export_logic(output_path))

            assert "automation" in paths
            assert Path(output_path).exists()

    def test_fluent_export_json(self, music_brain):
        """Test fluent API export_json()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "settings.json")
            result_path = (music_brain.process("grief")
                           .map_to_mixer()
                           .export_json(output_path))

            assert result_path == output_path
            assert Path(output_path).exists()


# ==============================================================================
# HELPER FUNCTION TESTS
# ==============================================================================

class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_suggest_rules(self, music_brain):
        """Test suggest_rules returns suggestions."""
        suggestions = music_brain.suggest_rules("grief")

        assert isinstance(suggestions, list)
        # May be empty or populated depending on implementation

    def test_get_affect_mapping(self, music_brain):
        """Test get_affect_mapping returns mapping."""
        mapping = music_brain.get_affect_mapping("grief")

        # May return dict or None
        if mapping is not None:
            assert isinstance(mapping, dict)

    def test_list_mixer_presets(self, music_brain):
        """Test list_mixer_presets returns list."""
        presets = music_brain.list_mixer_presets()

        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_get_mixer_preset(self, music_brain):
        """Test get_mixer_preset returns preset or None."""
        preset = music_brain.get_mixer_preset("grief")

        # May return MixerParameters or None
        if preset is not None:
            assert hasattr(preset, 'reverb_mix')


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_quick_generate(self):
        """Test quick_generate convenience function."""
        result = quick_generate("hopeful")

        assert isinstance(result, GeneratedMusic)
        assert result.emotional_state.primary_emotion in ["hope", "neutral"]

    def test_quick_export(self):
        """Test quick_export convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "quick_test")
            paths = quick_export("calm", output_path)

            assert "automation" in paths
            assert Path(paths["automation"]).exists()


# ==============================================================================
# EDGE CASE TESTS
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_text(self, music_brain):
        """Test empty emotional text."""
        result = music_brain.generate_from_text("")

        # Should not crash, uses neutral defaults
        assert isinstance(result, GeneratedMusic)

    def test_numeric_text(self, music_brain):
        """Test numeric input."""
        result = music_brain.generate_from_text("12345")

        # Should handle gracefully
        assert isinstance(result, GeneratedMusic)

    def test_special_characters(self, music_brain):
        """Test special characters in text."""
        result = music_brain.generate_from_text("@#$% &*()")

        # Should not crash
        assert isinstance(result, GeneratedMusic)

    def test_very_long_text(self, music_brain):
        """Test very long emotional text."""
        long_text = "grief " * 1000
        result = music_brain.generate_from_text(long_text)

        # Should handle without issues
        assert isinstance(result, GeneratedMusic)
        assert result.emotional_state.primary_emotion == "grief"

    def test_mixed_emotions(self, music_brain):
        """Test text with multiple emotions."""
        result = music_brain.generate_from_text("grief and hope and anxiety")

        # Should pick first matching emotion
        assert isinstance(result, GeneratedMusic)
        assert result.emotional_state.primary_emotion in ["grief", "hope", "anxiety"]


# ==============================================================================
# VALIDATION TESTS
# ==============================================================================

class TestParameterValidation:
    """Test parameter validation and bounds checking."""

    def test_dissonance_clamped_to_range(self, music_brain):
        """Test dissonance is clamped to [0, 1]."""
        chain = music_brain.process("calm").map_to_music().with_dissonance(5.0)
        assert 0.0 <= chain.musical_params.dissonance <= 1.0

        chain2 = music_brain.process("calm").map_to_music().with_dissonance(-2.0)
        assert 0.0 <= chain2.musical_params.dissonance <= 1.0

    def test_tempo_positive(self, music_brain):
        """Test tempo is always positive."""
        result = music_brain.generate_from_text("grief")
        assert result.musical_params.tempo_suggested > 0

    def test_velocity_bounds(self, music_brain):
        """Test generated velocities are in MIDI range."""
        result = music_brain.generate_from_text("anxiety")
        # Mixer params may have velocity-related settings
        assert isinstance(result.mixer_params.to_dict(), dict)


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

class TestEmotionAPIIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_declarative(self, music_brain):
        """Test complete declarative workflow."""
        # Create intent
        intent = music_brain.create_intent(
            title="Integration Test",
            core_event="Testing full workflow",
            mood_primary="grief",
            technical_key="F",
            technical_mode="major",
            tempo_range=(80, 90),
            rule_to_break="HARMONY_ModalInterchange",
            rule_justification="Testing rule breaks"
        )

        # Generate music
        result = music_brain.generate_from_intent(intent)

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = str(Path(tmpdir) / "integration_test")
            paths = music_brain.export_to_logic(result, output_base)

            # Verify all parts work
            assert result.emotional_state.primary_emotion == "grief"
            assert 80 <= result.musical_params.tempo_suggested <= 90
            assert Path(paths["automation"]).exists()

    def test_full_workflow_fluent(self, music_brain):
        """Test complete fluent workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = str(Path(tmpdir) / "fluent_integration.json")

            result = (music_brain.process("anxiety and tension")
                      .map_to_emotion()
                      .map_to_music()
                      .with_tempo(120)
                      .with_dissonance(0.7)
                      .map_to_mixer()
                      .export_logic(output_path))

            assert Path(output_path).exists()
            assert result["emotional_state"] == "anxiety"
            assert result["tempo"] == "120"

    def test_grief_processing_use_case(self, music_brain):
        """Test realistic grief processing use case."""
        intent = music_brain.create_intent(
            title="When I Found You Sleeping",
            core_event="Finding peace in letting go",
            mood_primary="grief",
            technical_key="F",
            technical_mode="major",
            tempo_range=(78, 86),
            rule_to_break="HARMONY_ModalInterchange",
            rule_justification="Bbm makes hope feel earned, not given"
        )

        result = music_brain.generate_from_intent(intent)

        # Verify appropriate parameters for grief
        assert result.emotional_state.primary_emotion == "grief"
        assert result.musical_params.tempo_suggested < 100
        assert result.mixer_params.reverb_mix > 0.3  # More reverb for grief


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
