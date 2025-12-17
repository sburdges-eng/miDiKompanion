"""
Kelly Phase 2 - Multi-Agent System
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from kelly_phase2_core import EmotionVector

@dataclass
class AgentMessage:
    timestamp: str
    agent_id: str
    emotion_state: dict
    action_state: dict
    coherence: float = 1.0

class MusicAgent:
    """Base agent class."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = {}
        self.coherence = 1.0
    
    def update(self, emotion: EmotionVector, context: dict) -> None:
        raise NotImplementedError
    
    def message(self) -> AgentMessage:
        return AgentMessage(
            datetime.utcnow().isoformat(),
            self.agent_id,
            self.state.get("emotion", {}),
            self.state.get("action", {}),
            self.coherence
        )

class HarmonyAgent(MusicAgent):
    """Chord progression generation."""
    def update(self, e: EmotionVector, ctx: dict):
        mode = "major" if e.valence > 0 else "minor"
        ext = "9th" if e.arousal > 0.7 else "7th" if abs(e.valence) > 0.6 else "triad"
        
        if e.valence > 0.3:
            prog = "I-V-vi-IV"
        elif e.valence < -0.3:
            prog = "i-bVI-bVII-i"
        else:
            prog = "i-iv-V-i"
        
        self.state = {
            "emotion": {"v": e.valence, "a": e.arousal},
            "action": {"mode": mode, "extensions": ext, "progression": prog}
        }

class RhythmAgent(MusicAgent):
    """Rhythmic pattern generation."""
    SWING = {"jazz": 0.67, "hiphop": 0.55, "rock": 0.5, "funk": 0.52}
    
    def update(self, e: EmotionVector, ctx: dict):
        genre = ctx.get("genre", "pop")
        base_swing = self.SWING.get(genre, 0.5)
        
        self.state = {"action": {
            "density": 0.3 + 0.7 * e.arousal,
            "swing": base_swing + 0.1 * (1 - e.arousal),
            "syncopation": 0.2 + 0.4 * e.arousal + 0.2 * (1 - e.dominance)
        }}

class MelodyAgent(MusicAgent):
    """Melodic line generation."""
    def update(self, e: EmotionVector, ctx: dict):
        self.state = {"action": {
            "range": 12 + int(e.arousal * 12),
            "leap_prob": 0.2 + 0.3 * e.arousal,
            "rest_prob": 0.1 + 0.2 * (1 - e.arousal),
            "ornament_prob": 0.1 + 0.2 * e.dominance
        }}

class DynamicsAgent(MusicAgent):
    """Dynamics control."""
    def update(self, e: EmotionVector, ctx: dict):
        self.state = {"action": {
            "base_velocity": int(60 + 67 * e.dominance),
            "velocity_range": int(20 + 40 * e.arousal),
            "crescendo_prob": 0.3 if e.arousal > 0.5 else 0.1,
            "accent_strength": 1.0 + 0.3 * e.dominance
        }}

class AgentCoordinator:
    """Coordinates all agents."""
    def __init__(self):
        self.agents = {
            "harmony": HarmonyAgent("harmony"),
            "rhythm": RhythmAgent("rhythm"),
            "melody": MelodyAgent("melody"),
            "dynamics": DynamicsAgent("dynamics")
        }
    
    def update_all(self, emotion: EmotionVector, context: dict) -> Dict[str, dict]:
        for agent in self.agents.values():
            agent.update(emotion, context)
        return {k: v.state.get("action", {}) for k, v in self.agents.items()}
    
    def coherence(self) -> float:
        return sum(a.coherence for a in self.agents.values()) / len(self.agents)
    
    def messages(self) -> List[AgentMessage]:
        return [a.message() for a in self.agents.values()]

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    coord = AgentCoordinator()
    emotion = EmotionVector(valence=-0.5, arousal=0.3, dominance=0.4)
    context = {"genre": "lofi"}
    
    params = coord.update_all(emotion, context)
    
    print("Agent Outputs:")
    for name, actions in params.items():
        print(f"  {name}: {actions}")
    print(f"Coherence: {coord.coherence():.2f}")
