import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from music_brain.audio.analyzer import (
    AudioAnalyzer,
    BPMDetectionResult,
    KeyDetectionResult,
    detect_bpm,
    detect_key,
    LIBROSA_AVAILABLE,
)

try:
    import soundfile as sf
    SOUND_FILE_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SOUND_FILE_AVAILABLE = False


pytestmark = pytest.mark.skipif(not LIBROSA_AVAILABLE, reason="librosa required for audio tests")


def _sine_wave(freq: float, sr: int, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * math.pi * freq * t)


def test_detect_key_identifies_c_major():
    sr = 22050
    duration = 2.0
    signal = (_sine_wave(261.63, sr, duration) + _sine_wave(329.63, sr, duration) + _sine_wave(392.0, sr, duration)) / 3
    result: KeyDetectionResult = detect_key(signal, sr)
    assert result.key == "C"
    assert result.mode in {result.mode.MAJOR, result.mode.MINOR}
    assert 0.0 <= result.confidence <= 1.0


def test_detect_bpm_click_track():
    sr = 22050
    tempo = 120
    duration = 5.0
    samples = int(sr * duration)
    click_track = np.zeros(samples, dtype=float)
    step = int(sr * 60 / tempo)
    click_track[::step] = 1.0
    result: BPMDetectionResult = detect_bpm(click_track, sr)
    assert abs(result.bpm - tempo) <= 3
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.skipif(not SOUND_FILE_AVAILABLE, reason="soundfile required for end-to-end audio analyzer test")
def test_audio_analyzer_full_pipeline(tmp_path: Path):
    sr = 22050
    duration = 4.0
    signal = (_sine_wave(440.0, sr, duration) + _sine_wave(880.0, sr, duration)) / 2
    audio_path = tmp_path / "test.wav"
    sf.write(audio_path, signal, sr)

    analyzer = AudioAnalyzer()
    analysis = analyzer.analyze_file(
        str(audio_path),
        detect_key=True,
        detect_bpm=True,
        extract_features_flag=True,
        analyze_segments=True,
        num_segments=3,
    )

    assert analysis.key_result is not None
    assert analysis.bpm_result is not None
    assert analysis.feature_summary
    assert len(analysis.segments) >= 1

