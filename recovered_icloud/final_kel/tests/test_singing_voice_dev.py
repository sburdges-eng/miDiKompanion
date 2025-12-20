from music_brain.voice import SingingVoiceDev, SINGING_VOICE_DEV_PROMPT


def test_dev_prompt_present():
    assert isinstance(SINGING_VOICE_DEV_PROMPT, str)
    assert "NeuralBackend" in SINGING_VOICE_DEV_PROMPT
    assert len(SINGING_VOICE_DEV_PROMPT) > 50


def test_singing_voice_dev_instantiates():
    voice = SingingVoiceDev(backend="formant", sample_rate=8000)
    audio = voice.preview("la", [60], tempo_bpm=120)
    assert audio.size > 0
