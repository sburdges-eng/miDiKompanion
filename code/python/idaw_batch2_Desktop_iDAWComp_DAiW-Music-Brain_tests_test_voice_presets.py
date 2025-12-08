"""
Unit tests for voice preset helpers.
"""

from music_brain.voice.auto_tune import get_auto_tune_preset, AutoTuneSettings
from music_brain.voice.modulator import get_modulation_preset, ModulationSettings
from music_brain.voice.synthesizer import get_voice_profile, SynthConfig


def test_auto_tune_preset_lookup():
    preset = get_auto_tune_preset("transparent")
    assert isinstance(preset, AutoTuneSettings)
    assert 0 <= preset.strength <= 1


def test_modulation_preset_lookup():
    preset = get_modulation_preset("intimate_whisper")
    assert isinstance(preset, ModulationSettings)
    assert preset.low_pass_hz is not None


def test_voice_profile_lookup():
    profile = get_voice_profile("guide_vulnerable")
    assert isinstance(profile, SynthConfig)
    assert profile.timbre == "breathy"

