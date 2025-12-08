#!/usr/bin/env python3
"""
Tests for Emotional → Musical Mapping

Tests:
- Emotional presets exist and have valid parameters
- Grief tempo is 60-82 BPM
- Interval emotion mappings are correct
- get_parameters_for_state() returns valid MusicalParameters
- get_interrogation_prompts() returns meaningful questions
- Humanization parameters match reference song ("When I Found You Sleeping")
"""

import pytest
from music_brain.models import (
    EmotionalState,
    MusicalParameters,
    TimingFeel,
    Register,
    HarmonicRhythm,
    Density,
    EMOTIONAL_PRESETS,
    INTERVAL_EMOTIONS,
    EMOTION_MODIFIERS,
    EMOTIONAL_STATE_PRESETS,
    get_parameters_for_state,
    get_interrogation_prompts,
    get_misdirection_technique,
)


class TestEmotionalPresets:
    """Test that all emotional presets exist and have valid parameters"""

    def test_all_presets_exist(self):
        """Test that all expected emotional presets exist"""
        expected_presets = ["grief", "anxiety", "nostalgia", "anger", "calm"]
        for preset in expected_presets:
            assert preset in EMOTIONAL_PRESETS, f"Missing preset: {preset}"

    def test_grief_preset(self):
        """Test grief preset parameters"""
        grief = EMOTIONAL_PRESETS["grief"]
        assert grief.tempo_min == 60
        assert grief.tempo_max == 82
        assert grief.tempo_suggested == 72
        assert "minor" in grief.mode_weights
        assert grief.timing_feel == TimingFeel.BEHIND
        assert grief.dissonance == 0.3
        assert grief.density == Density.SPARSE

    def test_anxiety_preset(self):
        """Test anxiety preset parameters"""
        anxiety = EMOTIONAL_PRESETS["anxiety"]
        assert anxiety.tempo_min == 100
        assert anxiety.tempo_max == 140
        assert "phrygian" in anxiety.mode_weights
        assert anxiety.timing_feel == TimingFeel.AHEAD
        assert anxiety.dissonance == 0.6
        assert anxiety.density == Density.DENSE

    def test_nostalgia_preset(self):
        """Test nostalgia preset parameters (Kelly song reference)"""
        nostalgia = EMOTIONAL_PRESETS["nostalgia"]
        assert nostalgia.tempo_min == 70
        assert nostalgia.tempo_max == 90
        assert nostalgia.tempo_suggested == 82  # Kelly song tempo
        assert "mixolydian" in nostalgia.mode_weights
        assert nostalgia.timing_feel == TimingFeel.BEHIND
        assert nostalgia.dissonance == 0.25

    def test_anger_preset(self):
        """Test anger preset parameters"""
        anger = EMOTIONAL_PRESETS["anger"]
        assert anger.tempo_min == 120
        assert anger.tempo_max == 160
        assert "phrygian" in anger.mode_weights
        assert anger.timing_feel == TimingFeel.AHEAD
        assert anger.dissonance == 0.5

    def test_calm_preset(self):
        """Test calm preset parameters"""
        calm = EMOTIONAL_PRESETS["calm"]
        assert calm.tempo_min == 60
        assert calm.tempo_max == 80
        assert "major" in calm.mode_weights or "lydian" in calm.mode_weights
        assert calm.timing_feel == TimingFeel.BEHIND
        assert calm.dissonance == 0.1

    def test_mode_weights_sum_to_one(self):
        """Test that all mode weights are normalized"""
        for name, preset in EMOTIONAL_PRESETS.items():
            total = sum(preset.mode_weights.values())
            assert 0.99 <= total <= 1.01, f"{name} mode weights sum to {total}"

    def test_dissonance_in_range(self):
        """Test that dissonance is in [0, 1] for all presets"""
        for name, preset in EMOTIONAL_PRESETS.items():
            assert 0 <= preset.dissonance <= 1, \
                f"{name} dissonance {preset.dissonance} not in [0, 1]"

    def test_space_probability_in_range(self):
        """Test that space_probability is in [0, 1] for all presets"""
        for name, preset in EMOTIONAL_PRESETS.items():
            assert 0 <= preset.space_probability <= 1, \
                f"{name} space_probability {preset.space_probability} not in [0, 1]"


class TestIntervalEmotions:
    """Test interval → emotion mappings"""

    def test_interval_emotions_exist(self):
        """Test that key intervals are mapped"""
        expected_intervals = [
            "P1", "m2", "M2", "m3", "M3", "P4", "tritone",
            "P5", "m6", "M6", "m7", "M7", "P8"
        ]
        for interval in expected_intervals:
            assert interval in INTERVAL_EMOTIONS, f"Missing interval: {interval}"

    def test_interval_values_in_range(self):
        """Test that all interval tensions are in [0, 1]"""
        for interval, tension in INTERVAL_EMOTIONS.items():
            assert 0 <= tension <= 1, \
                f"{interval} tension {tension} not in [0, 1]"

    def test_consonance_hierarchy(self):
        """Test that consonant intervals have lower tension than dissonant"""
        # Perfect consonances should be very stable
        assert INTERVAL_EMOTIONS["P1"] == 0.0
        assert INTERVAL_EMOTIONS["P8"] == 0.0
        assert INTERVAL_EMOTIONS["P5"] < 0.2

        # Imperfect consonances should be moderately stable
        assert INTERVAL_EMOTIONS["M3"] < 0.3
        assert INTERVAL_EMOTIONS["m3"] < 0.5

        # Dissonances should be tense
        assert INTERVAL_EMOTIONS["tritone"] >= 0.9
        assert INTERVAL_EMOTIONS["M7"] >= 0.8
        assert INTERVAL_EMOTIONS["m2"] >= 0.8


class TestEmotionModifiers:
    """Test emotion modifiers (PTSD, misdirection, etc.)"""

    def test_ptsd_intrusion_modifier(self):
        """Test PTSD intrusion modifier exists and has correct properties"""
        ptsd = EMOTION_MODIFIERS["ptsd_intrusion"]
        assert "intrusion_probability" in ptsd
        assert "types" in ptsd
        assert "register_spike" in ptsd["types"]
        assert "unresolved_dissonance" in ptsd["types"]

    def test_misdirection_modifier(self):
        """Test misdirection modifier (Kelly song technique)"""
        misdirection = EMOTION_MODIFIERS["misdirection"]
        assert "progression_pattern" in misdirection
        assert "emotional_impact" in misdirection
        assert "example" in misdirection
        # Kelly song reference
        assert "F-C-Am-Dm" in misdirection["example"]


class TestGetParametersForState:
    """Test get_parameters_for_state() function"""

    def test_profound_grief_mapping(self):
        """Test mapping profound grief to musical parameters"""
        state = EMOTIONAL_STATE_PRESETS["profound_grief"]
        params = get_parameters_for_state(state)

        assert isinstance(params, MusicalParameters)
        assert 60 <= params.tempo_suggested <= 82
        assert params.timing_feel == TimingFeel.BEHIND
        assert params.dissonance >= 0.3  # Adjusted for valence

    def test_ptsd_anxiety_mapping(self):
        """Test PTSD anxiety with intrusions"""
        state = EMOTIONAL_STATE_PRESETS["ptsd_anxiety"]
        params = get_parameters_for_state(state)

        assert isinstance(params, MusicalParameters)
        # PTSD intrusions should increase dissonance and space
        assert params.dissonance > EMOTIONAL_PRESETS["anxiety"].dissonance
        assert params.space_probability > EMOTIONAL_PRESETS["anxiety"].space_probability

    def test_valence_affects_dissonance(self):
        """Test that negative valence increases dissonance"""
        # Positive valence state
        positive_state = EmotionalState(
            valence=0.8,
            arousal=0.5,
            primary_emotion="calm"
        )
        positive_params = get_parameters_for_state(positive_state)

        # Negative valence state
        negative_state = EmotionalState(
            valence=-0.8,
            arousal=0.5,
            primary_emotion="grief"
        )
        negative_params = get_parameters_for_state(negative_state)

        # Negative valence should have more dissonance
        assert negative_params.dissonance > positive_params.dissonance

    def test_arousal_affects_tempo(self):
        """Test that higher arousal increases tempo"""
        # Low arousal
        low_arousal = EmotionalState(
            valence=-0.5,
            arousal=0.2,
            primary_emotion="grief"
        )
        low_params = get_parameters_for_state(low_arousal)

        # High arousal
        high_arousal = EmotionalState(
            valence=-0.5,
            arousal=0.9,
            primary_emotion="grief"
        )
        high_params = get_parameters_for_state(high_arousal)

        # Higher arousal should suggest faster tempo
        assert high_params.tempo_suggested > low_params.tempo_suggested


class TestGetInterrogationPrompts:
    """Test get_interrogation_prompts() function"""

    def test_returns_list_of_strings(self):
        """Test that function returns list of strings"""
        params = EMOTIONAL_PRESETS["grief"]
        prompts = get_interrogation_prompts(params)

        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)

    def test_slow_tempo_prompts(self):
        """Test prompts for slow tempo"""
        params = EMOTIONAL_PRESETS["grief"]  # Slow tempo
        prompts = get_interrogation_prompts(params)

        # Should ask about slow, reflective feeling
        assert any("slow" in p.lower() or "reflective" in p.lower() for p in prompts)

    def test_fast_tempo_prompts(self):
        """Test prompts for fast tempo"""
        params = EMOTIONAL_PRESETS["anger"]  # Fast tempo (140 BPM > 120)
        prompts = get_interrogation_prompts(params)

        # Should ask about urgency or energy
        assert any("urgent" in p.lower() or "energy" in p.lower() for p in prompts)

    def test_high_dissonance_prompts(self):
        """Test prompts for high dissonance"""
        params = EMOTIONAL_PRESETS["anxiety"]  # High dissonance
        prompts = get_interrogation_prompts(params)

        # Should ask about tension
        assert any("tension" in p.lower() for p in prompts)

    def test_behind_beat_prompts(self):
        """Test prompts for behind-the-beat timing"""
        params = EMOTIONAL_PRESETS["grief"]  # Behind beat
        prompts = get_interrogation_prompts(params)

        # Should ask about timing feel
        assert any("timing" in p.lower() or "drag" in p.lower() for p in prompts)


class TestMisdirectionTechnique:
    """Test misdirection technique (Kelly song)"""

    def test_nostalgia_to_grief_misdirection(self):
        """Test major → minor tonic misdirection (Kelly technique)"""
        misdirection = get_misdirection_technique("nostalgia", "grief")

        assert misdirection is not None
        assert misdirection["name"] == "Major → Minor Tonic Gut Punch"
        assert "F-C-Am-Dm" in misdirection["example"]
        assert misdirection["reveal_chord"] == "i"  # Minor tonic
        assert "Kelly technique" in misdirection["emotional_impact"]

    def test_invalid_combination_returns_none(self):
        """Test that invalid combinations return None"""
        misdirection = get_misdirection_technique("grief", "grief")
        # Same emotion shouldn't have misdirection
        assert misdirection is None or misdirection == {}


class TestEmotionalStateValidation:
    """Test EmotionalState validation"""

    def test_valid_emotional_state(self):
        """Test creating valid EmotionalState"""
        state = EmotionalState(
            valence=-0.5,
            arousal=0.7,
            primary_emotion="grief"
        )
        assert state.valence == -0.5
        assert state.arousal == 0.7

    def test_valence_out_of_range_raises(self):
        """Test that valence outside [-1, 1] raises AssertionError"""
        with pytest.raises(AssertionError):
            EmotionalState(
                valence=2.0,  # Invalid
                arousal=0.5,
                primary_emotion="grief"
            )

    def test_arousal_out_of_range_raises(self):
        """Test that arousal outside [0, 1] raises AssertionError"""
        with pytest.raises(AssertionError):
            EmotionalState(
                valence=0.5,
                arousal=1.5,  # Invalid
                primary_emotion="anger"
            )


class TestKellySongReference:
    """Test parameters match Kelly's 'When I Found You Sleeping'"""

    def test_kelly_song_tempo(self):
        """Test that nostalgia preset matches Kelly song tempo (82 BPM)"""
        nostalgia = EMOTIONAL_PRESETS["nostalgia"]
        assert nostalgia.tempo_suggested == 82

    def test_kelly_song_timing_feel(self):
        """Test behind-the-beat timing (lo-fi bedroom emo)"""
        nostalgia = EMOTIONAL_PRESETS["nostalgia"]
        assert nostalgia.timing_feel == TimingFeel.BEHIND

    def test_kelly_song_mode(self):
        """Test mixolydian mode for nostalgia (misdirection to minor)"""
        nostalgia = EMOTIONAL_PRESETS["nostalgia"]
        assert "mixolydian" in nostalgia.mode_weights
        # Mixolydian should be dominant mode
        assert nostalgia.mode_weights["mixolydian"] >= 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
