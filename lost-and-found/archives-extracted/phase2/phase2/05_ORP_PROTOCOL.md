# Kelly Phase 2 - Omega Resonance Protocol (ORP)

## Message Schema

```json
{
  "header": {
    "version": "1.0",
    "timestamp": "ISO8601",
    "agent_id": "string",
    "session_id": "string"
  },
  "emotion_state": {
    "valence": -1.0 to 1.0,
    "arousal": 0.0 to 1.0,
    "dominance": 0.0 to 1.0
  },
  "context_state": {
    "circadian_phase": 0.0 to 1.0,
    "location": "string",
    "sleep_quality": 0.0 to 1.0
  },
  "action_state": {
    "tempo": 60-180,
    "mode": "major|minor|dorian|...",
    "key": "C|C#|D|..."
  },
  "reward_feedback": {
    "bio_reward": 0.0 to 1.0,
    "coherence": 0.0 to 1.0,
    "resonance_score": 0.0 to 1.0
  }
}
```

## OSC Routes

| Path | Type | Description |
|------|------|-------------|
| `/kelly/emotion/valence` | float | -1 to 1 |
| `/kelly/emotion/arousal` | float | 0 to 1 |
| `/kelly/emotion/dominance` | float | 0 to 1 |
| `/kelly/music/tempo` | int | BPM |
| `/kelly/music/mode` | string | Scale mode |
| `/kelly/music/key` | string | Musical key |
| `/kelly/reward/coherence` | float | 0 to 1 |

## Python Implementation

```python
from pythonosc import udp_client

class ORPClient:
    def __init__(self, host="127.0.0.1", port=8000):
        self.osc = udp_client.SimpleUDPClient(host, port)
    
    def send_emotion(self, e: EmotionVector):
        self.osc.send_message("/kelly/emotion/valence", e.valence)
        self.osc.send_message("/kelly/emotion/arousal", e.arousal)
        self.osc.send_message("/kelly/emotion/dominance", e.dominance)
    
    def send_music(self, params: dict):
        self.osc.send_message("/kelly/music/tempo", params["tempo"])
        self.osc.send_message("/kelly/music/mode", params["mode"])
    
    def send_reward(self, coherence: float, resonance: float):
        self.osc.send_message("/kelly/reward/coherence", coherence)
        self.osc.send_message("/kelly/reward/resonance", resonance)
```

## MIDI CC Mapping

| CC | Parameter | Range |
|----|-----------|-------|
| 1 | Mod (arousal) | 0-127 |
| 7 | Volume (dominance) | 0-127 |
| 10 | Pan (stereo width) | 0-127 |
| 11 | Expression | 0-127 |
| 74 | Brightness (valence) | 0-127 |
| 91 | Reverb (1-arousal) | 0-127 |

```python
def emotion_to_cc(e: EmotionVector) -> dict:
    return {
        1: int((e.arousal) * 127),
        7: int((e.dominance) * 127),
        74: int((e.valence + 1) * 63.5),
        91: int((1 - e.arousal) * 127)
    }
```
