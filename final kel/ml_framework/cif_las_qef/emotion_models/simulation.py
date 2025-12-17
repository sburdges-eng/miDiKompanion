"""
Emotional Field Simulator

Simulation and visualization capabilities for emotional fields.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from .classical import VADModel, VADState, PlutchikWheel, EmotionBasis
from .quantum import QuantumEmotionalField, EmotionSuperposition
from .hybrid import HybridEmotionalField


class EmotionalFieldSimulator:
    """
    Emotional Field Simulator
    
    Simulates and visualizes emotional field evolution.
    """
    
    def __init__(self):
        """Initialize simulator."""
        self.hybrid_field = HybridEmotionalField()
        self.time_history: List[float] = []
        self.field_history: List[Dict] = []
        
    def simulate(
        self,
        duration: float = 10.0,
        dt: float = 0.01,
        initial_vad: Optional[VADState] = None,
        initial_superposition: Optional[EmotionSuperposition] = None
    ) -> Dict:
        """
        Simulate emotional field evolution.
        
        Args:
            duration: Simulation duration
            dt: Time step
            initial_vad: Initial VAD state
            initial_superposition: Initial superposition
        
        Returns:
            Simulation results
        """
        # Initialize field
        self.hybrid_field.initialize(initial_vad, initial_superposition)
        
        # Clear history
        self.time_history = []
        self.field_history = []
        
        # Time array
        time_steps = np.arange(0, duration, dt)
        
        for t in time_steps:
            # Compute field
            field_state = self.hybrid_field.compute_field(t)
            
            # Store
            self.time_history.append(t)
            self.field_history.append(field_state)
            
            # Evolve
            self.hybrid_field.evolve(dt)
        
        return {
            "duration": duration,
            "time_steps": len(time_steps),
            "final_state": self.hybrid_field.get_status()
        }
    
    def plot_quantum_oscillations(
        self,
        emotions_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot quantum emotional field oscillations.
        
        Args:
            emotions_to_plot: List of emotion names to plot (default: all)
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not self.field_history:
            raise ValueError("No simulation data. Run simulate() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get emotion names
        if emotions_to_plot is None:
            # Get from first field state
            first_state = self.field_history[0]
            n_emotions = len(first_state["quantum_amplitudes"])
            emotions_to_plot = [f"Emotion_{i}" for i in range(n_emotions)]
        
        # Extract time series for each emotion
        time_array = np.array(self.time_history)
        
        for i, emotion_name in enumerate(emotions_to_plot):
            if i < len(self.field_history[0]["quantum_amplitudes"]):
                amplitudes = [
                    state["quantum_amplitudes"][i]
                    for state in self.field_history
                ]
                ax.plot(time_array, amplitudes, label=emotion_name, alpha=0.7)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title("Quantum Emotional Field Oscillations")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_vad_evolution(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot VAD evolution over time.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not self.field_history:
            raise ValueError("No simulation data. Run simulate() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        time_array = np.array(self.time_history)
        
        valence = [state["classical_vad"]["valence"] for state in self.field_history]
        arousal = [state["classical_vad"]["arousal"] for state in self.field_history]
        dominance = [state["classical_vad"]["dominance"] for state in self.field_history]
        
        ax.plot(time_array, valence, label="Valence", linewidth=2)
        ax.plot(time_array, arousal, label="Arousal", linewidth=2)
        ax.plot(time_array, dominance, label="Dominance", linewidth=2)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("VAD Evolution Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        return fig
    
    def plot_probability_evolution(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot probability evolution for each emotion.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not self.field_history:
            raise ValueError("No simulation data. Run simulate() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        time_array = np.array(self.time_history)
        
        # Get number of emotions
        n_emotions = len(self.field_history[0]["quantum_probabilities"])
        
        # Plot each emotion's probability
        for i in range(n_emotions):
            probabilities = [
                state["quantum_probabilities"][i]
                for state in self.field_history
            ]
            ax.plot(time_array, probabilities, label=f"Emotion {i}", alpha=0.7)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        ax.set_title("Emotion Probability Evolution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_coherence_entropy(self, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot coherence and entropy over time.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not self.field_history:
            raise ValueError("No simulation data. Run simulate() first.")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        time_array = np.array(self.time_history)
        
        coherence = [state["coherence"] for state in self.field_history]
        entropy = [state["entropy"] for state in self.field_history]
        
        ax1.plot(time_array, coherence, 'b-', linewidth=2)
        ax1.set_ylabel("Coherence")
        ax1.set_title("Emotional Coherence Over Time")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(time_array, entropy, 'r-', linewidth=2)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Entropy")
        ax2.set_title("Emotional Entropy Over Time")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def demonstrate_interference(
        self,
        emotion1: str = "joy",
        emotion2: str = "fear",
        duration: float = 5.0
    ) -> Dict:
        """
        Demonstrate emotional interference between two fields.
        
        Args:
            emotion1: First emotion
            emotion2: Second emotion
            duration: Simulation duration
        
        Returns:
            Interference results
        """
        from .quantum import EmotionalInterference, EmotionBasis
        
        # Create two superpositions
        qf = QuantumEmotionalField()
        
        # Create superposition favoring emotion1
        n = len(qf.basis_emotions)
        amp1 = np.zeros(n, dtype=complex)
        try:
            idx1 = qf.basis_emotions.index(EmotionBasis(emotion1.upper()))
            amp1[idx1] = 0.8 + 0.2j
        except (ValueError, AttributeError):
            amp1[0] = 0.8 + 0.2j
        
        psi1 = EmotionSuperposition(
            basis_emotions=qf.basis_emotions,
            amplitudes=amp1,
            phases=np.zeros(n)
        )
        
        # Create superposition favoring emotion2
        amp2 = np.zeros(n, dtype=complex)
        try:
            idx2 = qf.basis_emotions.index(EmotionBasis(emotion2.upper()))
            amp2[idx2] = 0.8 + 0.2j
        except (ValueError, AttributeError):
            amp2[1] = 0.8 + 0.2j
        
        psi2 = EmotionSuperposition(
            basis_emotions=qf.basis_emotions,
            amplitudes=amp2,
            phases=np.zeros(n)
        )
        
        # Compute interference
        interference = EmotionalInterference()
        result = interference.compute_interference(psi1, psi2)
        
        return {
            "emotion1": emotion1,
            "emotion2": emotion2,
            "interference_type": result["interference_type"],
            "total_intensity": result["total_intensity"],
            "interference_term": result["interference_term"]
        }
    
    def demonstrate_entanglement(
        self,
        agent_a_emotions: List[str],
        agent_b_emotions: List[str],
        n_observations: int = 10
    ) -> Dict:
        """
        Demonstrate emotional entanglement between two agents.
        
        Args:
            agent_a_emotions: Emotions for agent A
            agent_b_emotions: Emotions for agent B
            n_observations: Number of observations
        
        Returns:
            Entanglement results
        """
        from .quantum import EmotionalEntanglement, EmotionBasis
        
        # Convert to enums
        a_emotions = [EmotionBasis(e.upper()) for e in agent_a_emotions]
        b_emotions = [EmotionBasis(e.upper()) for e in agent_b_emotions]
        
        # Create entangled state
        entanglement = EmotionalEntanglement(
            agent_a_emotions=a_emotions,
            agent_b_emotions=b_emotions,
            entanglement_strength=0.7
        )
        
        # Make observations
        observations = []
        for _ in range(n_observations):
            # Randomly observe A
            import random
            a_emotion = random.choice(a_emotions)
            result = entanglement.observe_agent_a(a_emotion)
            observations.append(result)
        
        # Count correlations
        correlations = sum(
            1 for obs in observations
            if obs.get("agent_b_probability", 0) > 0.5
        )
        
        return {
            "n_observations": n_observations,
            "correlations": correlations,
            "correlation_rate": correlations / n_observations,
            "observations": observations[:5]  # First 5
        }
