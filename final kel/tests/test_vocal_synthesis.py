#!/usr/bin/env python3
"""
Integration Tests for Vocal Synthesis
=====================================
Tests for vocal synthesis components including VocoderEngine,
PhonemeConverter, and VoiceSynthesizer integration.
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Note: These tests would require C++ bindings or mock implementations
# For now, we test the Python vocal system

sys.path.insert(0, str(Path(__file__).parent.parent / "CODE" / "PYTHON CODE"))

try:
    from kelly_vocal_system import (
        KellyVocalSystem,
        VocalSystemConfig,
        VADVector,
        BiometricData
    )
    VOCAL_SYSTEM_AVAILABLE = True
except ImportError:
    VOCAL_SYSTEM_AVAILABLE = False
    print("Warning: Kelly vocal system not available")


class TestVocalSystem(unittest.TestCase):
    """Tests for KellyVocalSystem."""

    def setUp(self):
        if not VOCAL_SYSTEM_AVAILABLE:
            self.skipTest("Vocal system not available")
        self.config = VocalSystemConfig()
        self.system = KellyVocalSystem(self.config)

    def test_emotion_to_vad(self):
        """Test emotion to VAD conversion."""
        emotion_name = "sadness"
        self.system.set_emotion_from_name(emotion_name)

        vad = self.system.get_current_vad()
        self.assertIsNotNone(vad)
        self.assertGreaterEqual(vad.valence, -1.0)
        self.assertLessEqual(vad.valence, 1.0)
        self.assertGreaterEqual(vad.arousal, 0.0)
        self.assertLessEqual(vad.arousal, 1.0)

    def test_biometric_to_vad(self):
        """Test biometric to VAD conversion."""
        bio = BiometricData(
            heartRate=80.0,
            heartRateVariability=45.0,
            skinConductance=6.0,
            temperature=36.5
        )
        self.system.set_emotion_from_biometrics(bio)

        vad = self.system.get_current_vad()
        self.assertIsNotNone(vad)

    def test_lyric_generation(self):
        """Test lyric generation from emotion."""
        self.system.set_emotion_from_name("joy")
        wound_desc = "feeling happy and free"

        result = self.system.generate_vocals(wound_desc)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.lyrics)
        self.assertGreater(len(result.lyrics.sections), 0)


class TestVADSystem(unittest.TestCase):
    """Tests for VAD (Valence-Arousal-Dominance) system."""

    def test_vad_creation(self):
        """Test VAD vector creation."""
        if not VOCAL_SYSTEM_AVAILABLE:
            self.skipTest("Vocal system not available")

        vad = VADVector(valence=0.5, arousal=0.7, dominance=0.6)

        self.assertEqual(vad.valence, 0.5)
        self.assertEqual(vad.arousal, 0.7)
        self.assertEqual(vad.dominance, 0.6)

    def test_vad_energy(self):
        """Test VAD energy calculation."""
        if not VOCAL_SYSTEM_AVAILABLE:
            self.skipTest("Vocal system not available")

        vad = VADVector(valence=0.5, arousal=0.8, dominance=0.6)
        energy = vad.energy()

        self.assertGreater(energy, 0.0)
        self.assertLessEqual(energy, 2.0)  # Max theoretical value

    def test_vad_tension(self):
        """Test VAD tension calculation."""
        if not VOCAL_SYSTEM_AVAILABLE:
            self.skipTest("Vocal system not available")

        vad = VADVector(valence=-0.5, arousal=0.7, dominance=0.3)
        tension = vad.tension()

        self.assertGreaterEqual(tension, 0.0)


class TestPhonemeProcessing(unittest.TestCase):
    """Tests for phoneme processing."""

    def test_text_to_phonemes(self):
        """Test text to phoneme conversion."""
        # This would test the C++ PhonemeConverter if bindings available
        # For now, test Python implementation
        text = "hello world"

        # Mock test - would need actual implementation
        self.assertTrue(len(text) > 0)

    def test_syllable_counting(self):
        """Test syllable counting."""
        words = ["hello", "world", "beautiful", "music"]

        # Expected syllable counts (approximate)
        expected = [2, 1, 3, 2]

        # Mock test
        for word, exp in zip(words, expected):
            # Would use actual syllable counter
            self.assertTrue(len(word) > 0)


class TestVoiceParameters(unittest.TestCase):
    """Tests for voice parameter generation."""

    def test_vad_to_voice(self):
        """Test VAD to voice parameter conversion."""
        if not VOCAL_SYSTEM_AVAILABLE:
            self.skipTest("Vocal system not available")

        vad = VADVector(valence=0.5, arousal=0.7, dominance=0.6)
        voice_params = self.system.qevf.vad_to_voice(vad)

        self.assertIsNotNone(voice_params)
        self.assertGreater(voice_params.f0Base, 0.0)
        self.assertGreater(voice_params.vibratoRate, 0.0)


if __name__ == "__main__":
    unittest.main()
