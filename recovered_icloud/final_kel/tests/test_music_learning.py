#!/usr/bin/env python3
"""
Tests for music learning system (melody, harmony, groove, bass, arrangement,
expression, rule-breaking) and unified manager.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from music_brain.learning.melody_learning import MelodyExample, MelodyLearningManager
from music_brain.learning.harmony_learning import ChordExample, HarmonyLearningManager
from music_brain.learning.groove_learning import GrooveExample, GrooveLearningManager
from music_brain.learning.bass_learning import BassExample, BassLearningManager
from music_brain.learning.arrangement_learning import ArrangementExample, ArrangementLearningManager
from music_brain.learning.expression_learning import ExpressionExample, ExpressionLearningManager
from music_brain.learning.rulebreak_learning import RuleBreakExample, RuleBreakLearningManager
from music_brain.learning.music_learning_manager import MusicLearningManager


def test_melody_learning():
    mgr = MelodyLearningManager(storage_dir=Path("/tmp/parrot_tests/melodies"))
    mgr.add_example(MelodyExample(melody=[60, 62, 64, 65], emotion="joy", tempo=120))
    profile = mgr.learn_profile("test")
    assert profile.example_count == 1
    melody = mgr.generate("joy", profile_name="test", length=4)
    assert len(melody) == 4


def test_harmony_learning():
    mgr = HarmonyLearningManager(storage_dir=Path("/tmp/parrot_tests/harmonies"))
    mgr.add_example(ChordExample(progression=["C", "G", "Am", "F"], roman_numerals=["I", "V", "vi", "IV"], emotion="hope"))
    profile = mgr.learn_profile("test")
    assert profile.example_count == 1
    chords = mgr.generate("hope", profile_name="test", length=4)
    assert len(chords) == 4


def test_groove_learning():
    mgr = GrooveLearningManager(storage_dir=Path("/tmp/parrot_tests/grooves"))
    mgr.add_example(GrooveExample(timing_offsets_16th=[0.0]*16, velocity_curve=[80]*16, emotion="calm"))
    profile = mgr.learn_profile("test")
    assert profile.example_count == 1
    groove = mgr.generate("calm", profile_name="test")
    assert "timing_offsets_16th" in groove


def test_bass_learning():
    mgr = BassLearningManager(storage_dir=Path("/tmp/parrot_tests/bass"))
    mgr.add_example(BassExample(notes=[36, 38, 40], emotion="neutral"))
    profile = mgr.learn_profile("test")
    assert profile.example_count == 1
    bass = mgr.generate("neutral", profile_name="test", length=3)
    assert len(bass) == 3


def test_arrangement_learning():
    mgr = ArrangementLearningManager(storage_dir=Path("/tmp/parrot_tests/arrangements"))
    mgr.add_example(ArrangementExample(sections=["verse", "chorus"], instruments=["drums", "bass"], emotion="joy"))
    profile = mgr.learn_profile("test")
    assert profile.example_count == 1
    arrangement = mgr.generate("joy", profile_name="test", length=2)
    assert len(arrangement.get("sections", [])) == 2


def test_expression_learning():
    mgr = ExpressionLearningManager(storage_dir=Path("/tmp/parrot_tests/expression"))
    mgr.add_example(ExpressionExample(velocity_curve=[80]*16, emotion="joy"))
    profile = mgr.learn_profile("test")
    assert profile.example_count == 1
    expr = mgr.generate("joy", profile_name="test", length=16)
    assert len(expr.get("velocity_curve", [])) == 16


def test_rulebreak_learning():
    mgr = RuleBreakLearningManager(storage_dir=Path("/tmp/parrot_tests/rulebreaks"))
    mgr.add_example(RuleBreakExample(rule_break="HARMONY_ModalInterchange", emotion="grief", accepted=True))
    profile = mgr.learn_profile("test")
    assert profile.example_count == 1
    rb = mgr.choose("grief", profile="test")
    assert rb == "HARMONY_ModalInterchange"


def test_unified_manager():
    mgr = MusicLearningManager(storage_root=Path("/tmp/parrot_tests"))
    mgr.add_melody_example(MelodyExample(melody=[60, 62, 64], emotion="joy"), name="m1")
    mgr.learn_melody_profile("default")
    melody = mgr.generate_melody("joy", profile="default", length=3)
    assert len(melody) == 3


if __name__ == "__main__":
    test_melody_learning()
    test_harmony_learning()
    test_groove_learning()
    test_bass_learning()
    test_arrangement_learning()
    test_expression_learning()
    test_rulebreak_learning()
    test_unified_manager()
    print("All learning tests passed")
