# Kelly Phase 2 - Multi-Agent System

## Base Agent

```python
@dataclass
class AgentMessage:
    timestamp: str
    agent_id: str
    emotion_state: dict
    action_state: dict
    coherence: float = 1.0

class MusicAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = {}
        self.coherence = 1.0
    
    def update(self, emotion: EmotionVector, context: dict): pass
    
    def message(self) -> AgentMessage:
        return AgentMessage(datetime.utcnow().isoformat(), self.agent_id,
                           self.state.get("emotion",{}), self.state.get("action",{}), self.coherence)
```

## Harmony Agent

```python
class HarmonyAgent(MusicAgent):
    def update(self, e: EmotionVector, ctx: dict):
        mode = "major" if e.valence > 0 else "minor"
        ext = "9th" if e.arousal > 0.7 else "7th" if abs(e.valence) > 0.6 else "triad"
        prog = "I-V-vi-IV" if e.valence > 0.3 else "i-bVI-bVII-i" if e.valence < -0.3 else "i-iv-V-i"
        self.state = {"emotion": {"v": e.valence}, "action": {"mode": mode, "ext": ext, "prog": prog}}
```

## Rhythm Agent

```python
class RhythmAgent(MusicAgent):
    def update(self, e: EmotionVector, ctx: dict):
        genre = ctx.get("genre", "pop")
        base_swing = {"jazz": 0.67, "hiphop": 0.55, "rock": 0.5}.get(genre, 0.5)
        self.state = {"action": {
            "density": 0.3 + 0.7*e.arousal,
            "swing": base_swing + 0.1*(1-e.arousal),
            "syncopation": 0.2 + 0.4*e.arousal + 0.2*(1-e.dominance)
        }}
```

## Melody Agent

```python
class MelodyAgent(MusicAgent):
    def update(self, e: EmotionVector, ctx: dict):
        self.state = {"action": {
            "range": 12 + int(e.arousal * 12),  # Octave range
            "leap_prob": 0.2 + 0.3*e.arousal,
            "rest_prob": 0.1 + 0.2*(1-e.arousal),
            "ornament_prob": 0.1 + 0.2*e.dominance
        }}
```

## Dynamics Agent

```python
class DynamicsAgent(MusicAgent):
    def update(self, e: EmotionVector, ctx: dict):
        self.state = {"action": {
            "base_velocity": int(60 + 67*e.dominance),
            "velocity_range": int(20 + 40*e.arousal),
            "crescendo_prob": 0.3 if e.arousal > 0.5 else 0.1,
            "accent_strength": 1.0 + 0.3*e.dominance
        }}
```

## Coordinator

```python
class AgentCoordinator:
    def __init__(self):
        self.agents = {
            "harmony": HarmonyAgent("harmony"),
            "rhythm": RhythmAgent("rhythm"),
            "melody": MelodyAgent("melody"),
            "dynamics": DynamicsAgent("dynamics")
        }
    
    def update_all(self, emotion: EmotionVector, context: dict) -> dict:
        for agent in self.agents.values():
            agent.update(emotion, context)
        return {k: v.state["action"] for k, v in self.agents.items()}
    
    def coherence(self) -> float:
        return sum(a.coherence for a in self.agents.values()) / len(self.agents)
```
