import numpy as np

from music_brain.voice.singing_voice import SingingVoice


def test_singing_voice_preview_formant():
    voice = SingingVoice(backend="formant", sample_rate=16000)
    audio = voice.preview("la la", [60, 62], tempo_bpm=120)
    assert isinstance(audio, np.ndarray)
    assert audio.size > 0
    assert np.isfinite(audio).all()


def test_singing_voice_sing_formant():
    voice = SingingVoice(backend="formant", sample_rate=16000)
    audio = voice.sing("hello world", [60, 62, 64], tempo_bpm=100)
    assert audio.size > 0
    assert np.isfinite(audio).all()
