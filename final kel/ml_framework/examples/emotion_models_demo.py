"""
Emotion Models Demo

Demonstrates classical and quantum emotional models.
"""

import numpy as np
import matplotlib.pyplot as plt
from cif_las_qef.emotion_models import (
    VADModel, VADState, PlutchikWheel, EmotionBasis,
    QuantumEmotionalField, EmotionSuperposition,
    HybridEmotionalField, EmotionalFieldSimulator
)


def demo_vad_model():
    """Demonstrate VAD Model."""
    print("=== VAD Model Demo ===\n")
    
    vad_model = VADModel()
    
    # Create VAD state
    vad = VADState(valence=0.5, arousal=0.7, dominance=0.3)
    
    # Compute metrics
    metrics = vad_model.compute_all_metrics(vad)
    
    print(f"VAD State: V={vad.valence:.2f}, A={vad.arousal:.2f}, D={vad.dominance:.2f}")
    print(f"Energy Level: {metrics['energy_level']:.2f}")
    print(f"Emotional Tension: {metrics['emotional_tension']:.2f}")
    print(f"Stability Index: {metrics['stability_index']:.2f}\n")


def demo_plutchik_wheel():
    """Demonstrate Plutchik's Wheel."""
    print("=== Plutchik's Wheel Demo ===\n")
    
    plutchik = PlutchikWheel()
    
    # Get emotion VAD
    joy_vad = plutchik.emotion_to_vad(EmotionBasis.JOY, intensity=1.0)
    print(f"Joy VAD: V={joy_vad.valence:.2f}, A={joy_vad.arousal:.2f}, D={joy_vad.dominance:.2f}")
    
    # Combine emotions (Joy + Trust → Love)
    love_vad = plutchik.combine_emotions(EmotionBasis.JOY, EmotionBasis.TRUST)
    print(f"Love (Joy+Trust) VAD: V={love_vad.valence:.2f}, A={love_vad.arousal:.2f}, D={love_vad.dominance:.2f}")
    
    # Map VAD to emotions
    test_vad = VADState(valence=0.6, arousal=0.7, dominance=0.4)
    emotions = plutchik.vad_to_emotion(test_vad, threshold=0.3)
    print(f"\nVAD {test_vad.valence:.2f}, {test_vad.arousal:.2f}, {test_vad.dominance:.2f} maps to:")
    for emotion, similarity in emotions[:3]:
        print(f"  {emotion.value}: {similarity:.2f}")
    print()


def demo_quantum_superposition():
    """Demonstrate Quantum Superposition."""
    print("=== Quantum Superposition Demo ===\n")
    
    qf = QuantumEmotionalField()
    
    # Create superposition
    superposition = qf.create_superposition()
    
    # Get probabilities
    probabilities = superposition.get_probabilities()
    print("Emotion Probabilities:")
    for i, emotion in enumerate(superposition.basis_emotions):
        print(f"  {emotion.value}: {probabilities[i]:.3f}")
    
    # Compute coherence and entropy
    coherence = superposition.compute_coherence()
    entropy = superposition.compute_entropy()
    print(f"\nCoherence: {coherence:.3f}")
    print(f"Entropy: {entropy:.3f}")
    
    # Collapse
    collapsed_emotion, probability = superposition.collapse()
    print(f"\nCollapsed to: {collapsed_emotion.value} (probability: {probability:.3f})\n")


def demo_hybrid_field():
    """Demonstrate Hybrid Emotional Field."""
    print("=== Hybrid Emotional Field Demo ===\n")
    
    hybrid = HybridEmotionalField()
    
    # Initialize
    initial_vad = VADState(valence=0.3, arousal=0.6, dominance=0.2)
    hybrid.initialize(initial_vad)
    
    # Compute field
    field_state = hybrid.compute_field(t=0.0)
    print(f"Classical VAD: V={field_state['classical_vad']['valence']:.2f}, "
          f"A={field_state['classical_vad']['arousal']:.2f}, "
          f"D={field_state['classical_vad']['dominance']:.2f}")
    print(f"Quantum Coherence: {field_state['coherence']:.3f}")
    print(f"Quantum Entropy: {field_state['entropy']:.3f}")
    
    # Evolve
    evolution = hybrid.evolve(dt=0.1)
    print(f"\nAfter evolution:")
    print(f"New Coherence: {evolution['new_coherence']:.3f}")
    print(f"New Entropy: {evolution['new_entropy']:.3f}")
    
    # Observe (collapse)
    observation = hybrid.observe()
    print(f"\nAfter observation:")
    print(f"Collapsed Emotion: {observation['collapsed_emotion']}")
    print(f"New VAD: V={observation['new_vad']['valence']:.2f}, "
          f"A={observation['new_vad']['arousal']:.2f}, "
          f"D={observation['new_vad']['dominance']:.2f}\n")


def demo_simulation():
    """Demonstrate Emotional Field Simulation."""
    print("=== Emotional Field Simulation Demo ===\n")
    
    simulator = EmotionalFieldSimulator()
    
    # Initialize with specific VAD
    initial_vad = VADState(valence=0.5, arousal=0.7, dominance=0.3)
    
    # Simulate
    print("Running simulation...")
    result = simulator.simulate(
        duration=5.0,
        dt=0.05,
        initial_vad=initial_vad
    )
    
    print(f"Simulation complete: {result['time_steps']} time steps")
    print(f"Final Coherence: {result['final_state']['quantum_coherence']:.3f}")
    print(f"Final Entropy: {result['final_state']['quantum_entropy']:.3f}\n")
    
    # Plot (if matplotlib available)
    try:
        print("Generating plots...")
        
        # Quantum oscillations
        fig1 = simulator.plot_quantum_oscillations()
        fig1.suptitle("Quantum Emotional Field Oscillations")
        plt.tight_layout()
        plt.savefig("quantum_oscillations.png", dpi=150)
        print("Saved: quantum_oscillations.png")
        
        # VAD evolution
        fig2 = simulator.plot_vad_evolution()
        plt.tight_layout()
        plt.savefig("vad_evolution.png", dpi=150)
        print("Saved: vad_evolution.png")
        
        # Probability evolution
        fig3 = simulator.plot_probability_evolution()
        plt.tight_layout()
        plt.savefig("probability_evolution.png", dpi=150)
        print("Saved: probability_evolution.png")
        
        # Coherence and entropy
        fig4 = simulator.plot_coherence_entropy()
        plt.tight_layout()
        plt.savefig("coherence_entropy.png", dpi=150)
        print("Saved: coherence_entropy.png")
        
        print("\nAll plots saved successfully!\n")
        
    except Exception as e:
        print(f"Plotting failed: {e}\n")


def demo_interference():
    """Demonstrate Emotional Interference."""
    print("=== Emotional Interference Demo ===\n")
    
    simulator = EmotionalFieldSimulator()
    
    result = simulator.demonstrate_interference(
        emotion1="joy",
        emotion2="fear",
        duration=5.0
    )
    
    print(f"Interference between {result['emotion1']} and {result['emotion2']}:")
    print(f"Type: {result['interference_type']}")
    print(f"Total Intensity: {result['total_intensity']:.3f}")
    print(f"Interference Term: {result['interference_term']:.3f}\n")


def demo_entanglement():
    """Demonstrate Emotional Entanglement."""
    print("=== Emotional Entanglement Demo ===\n")
    
    simulator = EmotionalFieldSimulator()
    
    result = simulator.demonstrate_entanglement(
        agent_a_emotions=["joy", "fear", "anger"],
        agent_b_emotions=["joy", "fear", "anger"],
        n_observations=20
    )
    
    print(f"Entanglement Test: {result['n_observations']} observations")
    print(f"Correlations: {result['correlations']}")
    print(f"Correlation Rate: {result['correlation_rate']:.2%}")
    print(f"\nSample Observations:")
    for obs in result['observations']:
        print(f"  A: {obs['agent_a_emotion'].value} → B: {obs['agent_b_emotion'].value} "
              f"(prob: {obs['agent_b_probability']:.2f})")
    print()


def main():
    """Run all demos."""
    print("=" * 60)
    print("Emotion Models Demonstration")
    print("=" * 60)
    print()
    
    demo_vad_model()
    demo_plutchik_wheel()
    demo_quantum_superposition()
    demo_hybrid_field()
    demo_interference()
    demo_entanglement()
    
    # Simulation (may generate plots)
    try:
        demo_simulation()
    except Exception as e:
        print(f"Simulation demo failed: {e}\n")
    
    print("=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
