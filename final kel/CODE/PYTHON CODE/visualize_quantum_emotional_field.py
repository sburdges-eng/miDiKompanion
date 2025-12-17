#!/usr/bin/env python3
"""
Quantum Emotional Field Visualization

Creates visualizations of the quantum emotional field dynamics including:
- Wavefunction evolution
- Probability distributions
- Energy landscapes
- Resonance patterns
- Network coherence
"""

import sys
from pathlib import Path
import numpy as np

# Add path for quantum emotional field
base_path = Path(__file__).parent
sys.path.insert(0, str(base_path / "reference" / "python_kelly" / "core"))

try:
    from quantum_emotional_field import (
        EmotionalStateVector, QuantumEmotionalWavefunction, EmotionalHamiltonian,
        MusicGenerator, VoiceModulator, EmotionalNetwork, QuantumEmotionalField,
        vad_to_wavefunction, simulate_emotional_field, EmotionBasis,
        emotional_interference, EmotionalResonance
    )
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError as e:
    print(f"Error importing quantum emotional field: {e}")
    print("Make sure all dependencies are installed")
    sys.exit(1)


def create_wavefunction_visualization(output_file: str = "quantum_emotion_wavefunction.html"):
    """Visualize quantum wavefunction evolution"""
    
    # Create initial state
    initial_state = EmotionalStateVector(valence=0.5, arousal=0.7, dominance=0.3)
    qef = QuantumEmotionalField()
    
    # Simulate
    results = simulate_emotional_field(initial_state, duration=5.0, dt=0.05, qef=qef)
    
    # Create figure
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Emotion Probabilities Over Time",
            "VAD State Evolution",
            "Emotional Entropy",
            "Total Energy",
            "Music Frequency",
            "Voice Pitch"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Emotion probabilities
    colors = {
        EmotionBasis.JOY: "#FFD700",
        EmotionBasis.SADNESS: "#4169E1",
        EmotionBasis.ANGER: "#DC143C",
        EmotionBasis.FEAR: "#8B008B",
        EmotionBasis.SURPRISE: "#FF8C00",
        EmotionBasis.DISGUST: "#228B22",
        EmotionBasis.TRUST: "#00CED1",
        EmotionBasis.ANTICIPATION: "#FF69B4"
    }
    
    for emotion in EmotionBasis:
        fig.add_trace(
            go.Scatter(
                x=results['time'],
                y=results['probabilities'][emotion],
                name=emotion.value.capitalize(),
                line=dict(color=colors[emotion], width=2),
                mode='lines'
            ),
            row=1, col=1
        )
    
    # 2. VAD state
    fig.add_trace(
        go.Scatter(x=results['time'], y=results['valence'], name="Valence",
                  line=dict(color='green', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=results['time'], y=results['arousal'], name="Arousal",
                  line=dict(color='red', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=results['time'], y=results['dominance'], name="Dominance",
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )
    
    # 3. Entropy
    fig.add_trace(
        go.Scatter(x=results['time'], y=results['entropy'], name="Entropy",
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    # 4. Energy
    fig.add_trace(
        go.Scatter(x=results['time'], y=results['energy'], name="Energy",
                  line=dict(color='orange', width=2)),
        row=2, col=2
    )
    
    # 5. Music frequency
    fig.add_trace(
        go.Scatter(x=results['time'], y=results['music_freq'], name="Music Freq",
                  line=dict(color='cyan', width=2)),
        row=3, col=1
    )
    
    # 6. Voice pitch
    fig.add_trace(
        go.Scatter(x=results['time'], y=results['voice_pitch'], name="Voice Pitch",
                  line=dict(color='magenta', width=2)),
        row=3, col=2
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=2)
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="Entropy", row=2, col=1)
    fig.update_yaxes(title_text="Energy", row=2, col=2)
    fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)
    fig.update_yaxes(title_text="Pitch (Hz)", row=3, col=2)
    
    # Update layout
    fig.update_layout(
        title={'text': 'Quantum Emotional Field Evolution', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 24}},
        height=1200,
        width=1600,
        margin=dict(l=20, r=20, t=100, b=20),
        legend=dict(x=1.02, y=0.98)
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Wavefunction visualization saved to {output_file}")
    return fig


def create_energy_landscape_visualization(output_file: str = "quantum_emotion_energy_landscape.html"):
    """Visualize emotional potential energy landscape"""
    
    # Create grid
    v_range = np.linspace(-1, 1, 50)
    a_range = np.linspace(0, 1, 50)
    V, A = np.meshgrid(v_range, a_range)
    
    # Calculate energy for different dominance values
    D_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    fig = go.Figure()
    
    for d in D_values:
        energy = np.zeros_like(V)
        for i in range(len(a_range)):
            for j in range(len(v_range)):
                state = EmotionalStateVector(V[i, j], A[i, j], d)
                energy[i, j] = state.emotional_potential_energy()
        
        fig.add_trace(
            go.Surface(
                x=V, y=A, z=energy,
                name=f"Dominance = {d:.1f}",
                colorscale='Viridis',
                showscale=True
            )
        )
    
    fig.update_layout(
        title={'text': 'Emotional Potential Energy Landscape', 'x': 0.5, 'xanchor': 'center'},
        scene=dict(
            xaxis_title='Valence',
            yaxis_title='Arousal',
            zaxis_title='Potential Energy',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        width=1200,
        height=800
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Energy landscape saved to {output_file}")
    return fig


def create_network_coherence_visualization(output_file: str = "quantum_emotion_network.html"):
    """Visualize emotional network dynamics"""
    
    # Create network
    network = EmotionalNetwork(n_agents=5, coupling_strength=0.2)
    
    # Initialize with different states
    network.states[0] = EmotionalStateVector(0.8, 0.7, 0.5)  # Joyful
    network.states[1] = EmotionalStateVector(-0.6, 0.3, -0.2)  # Sad
    network.states[2] = EmotionalStateVector(-0.4, 0.9, 0.8)   # Angry
    network.states[3] = EmotionalStateVector(0.3, 0.5, 0.0)    # Neutral
    network.states[4] = EmotionalStateVector(0.5, 0.6, 0.3)     # Positive
    
    # Simulate
    duration = 10.0
    dt = 0.1
    n_steps = int(duration / dt)
    
    time = np.arange(0, duration, dt)
    coherence = np.zeros(n_steps)
    states_history = {i: {'v': [], 'a': [], 'd': []} for i in range(5)}
    
    for step in range(n_steps):
        coherence[step] = network.coherence()
        for i, state in enumerate(network.states):
            states_history[i]['v'].append(state.valence)
            states_history[i]['a'].append(state.arousal)
            states_history[i]['d'].append(state.dominance)
        network.emotional_coupling(dt)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Network Coherence Over Time",
            "Valence Evolution",
            "Arousal Evolution",
            "Dominance Evolution"
        )
    )
    
    # Coherence
    fig.add_trace(
        go.Scatter(x=time, y=coherence, name="Coherence",
                  line=dict(color='purple', width=3)),
        row=1, col=1
    )
    
    # States
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
    for i in range(5):
        fig.add_trace(
            go.Scatter(x=time, y=states_history[i]['v'], 
                      name=f"Agent {i+1} Valence",
                      line=dict(color=colors[i], width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=time, y=states_history[i]['a'],
                      name=f"Agent {i+1} Arousal",
                      line=dict(color=colors[i], width=2), showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time, y=states_history[i]['d'],
                      name=f"Agent {i+1} Dominance",
                      line=dict(color=colors[i], width=2), showlegend=False),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Coherence", row=1, col=1)
    fig.update_yaxes(title_text="Valence", row=1, col=2)
    fig.update_yaxes(title_text="Arousal", row=2, col=1)
    fig.update_yaxes(title_text="Dominance", row=2, col=2)
    
    fig.update_layout(
        title={'text': 'Emotional Network Dynamics', 'x': 0.5, 'xanchor': 'center'},
        height=800,
        width=1400
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Network visualization saved to {output_file}")
    return fig


def create_resonance_visualization(output_file: str = "quantum_emotion_resonance.html"):
    """Visualize resonance patterns and interference"""
    
    # Create two emotional states
    state1 = EmotionalStateVector(0.7, 0.6, 0.4)
    state2 = EmotionalStateVector(0.5, 0.8, 0.2)
    
    wf1 = vad_to_wavefunction(state1)
    wf2 = vad_to_wavefunction(state2)
    
    # Calculate interference
    interference = emotional_interference(wf1, wf2)
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Wavefunction 1 Probabilities",
            "Wavefunction 2 Probabilities",
            "Interference Pattern",
            "Resonance Energy"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    colors = {
        EmotionBasis.JOY: "#FFD700",
        EmotionBasis.SADNESS: "#4169E1",
        EmotionBasis.ANGER: "#DC143C",
        EmotionBasis.FEAR: "#8B008B",
        EmotionBasis.SURPRISE: "#FF8C00",
        EmotionBasis.DISGUST: "#228B22",
        EmotionBasis.TRUST: "#00CED1",
        EmotionBasis.ANTICIPATION: "#FF69B4"
    }
    
    # Probabilities
    probs1 = wf1.all_probabilities()
    probs2 = wf2.all_probabilities()
    
    emotions = [e.value for e in EmotionBasis]
    fig.add_trace(
        go.Bar(x=emotions, y=[probs1[e] for e in EmotionBasis],
              marker_color=[colors[e] for e in EmotionBasis]),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=emotions, y=[probs2[e] for e in EmotionBasis],
              marker_color=[colors[e] for e in EmotionBasis]),
        row=1, col=2
    )
    
    # Interference over time
    time = np.linspace(0, 5, 100)
    hamiltonian = EmotionalHamiltonian()
    interference_over_time = []
    
    wf1_evolved = wf1
    wf2_evolved = wf2
    
    for t in time:
        interference_over_time.append(emotional_interference(wf1_evolved, wf2_evolved))
        wf1_evolved = hamiltonian.evolve(wf1_evolved, 0.05)
        wf2_evolved = hamiltonian.evolve(wf2_evolved, 0.05)
    
    fig.add_trace(
        go.Scatter(x=time, y=interference_over_time, name="Interference",
                  line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    # Resonance energy
    resonance = EmotionalResonance()
    k_e = 1.0
    m_e = 1.0
    f_res = resonance.resonance_frequency(k_e, m_e)
    
    resonance_energy = []
    for t in time:
        # Simplified resonance energy calculation
        energy = np.cos(2 * np.pi * f_res * t) * np.exp(-0.1 * t)
        resonance_energy.append(energy)
    
    fig.add_trace(
        go.Scatter(x=time, y=resonance_energy, name="Resonance Energy",
                  line=dict(color='orange', width=2)),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Emotion", row=1, col=1)
    fig.update_xaxes(title_text="Emotion", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=2)
    fig.update_yaxes(title_text="Interference", row=2, col=1)
    fig.update_yaxes(title_text="Energy", row=2, col=2)
    
    fig.update_layout(
        title={'text': 'Quantum Emotional Resonance & Interference', 'x': 0.5, 'xanchor': 'center'},
        height=800,
        width=1400
    )
    
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"Resonance visualization saved to {output_file}")
    return fig


def main():
    """Generate all quantum emotional field visualizations"""
    print("Generating Quantum Emotional Field visualizations...")
    
    print("\n1. Wavefunction Evolution...")
    create_wavefunction_visualization()
    
    print("\n2. Energy Landscape...")
    create_energy_landscape_visualization()
    
    print("\n3. Network Dynamics...")
    create_network_coherence_visualization()
    
    print("\n4. Resonance Patterns...")
    create_resonance_visualization()
    
    print("\n✅ All quantum emotional field visualizations created!")


if __name__ == "__main__":
    main()
