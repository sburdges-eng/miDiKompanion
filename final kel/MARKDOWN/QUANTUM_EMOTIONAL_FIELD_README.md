# Quantum Emotional Field (QEF) Implementation

Complete implementation of the Quantum Emotional Field mathematical framework, bridging emotion, music, voice, and consciousness through quantum field theory.

## 📐 Implemented Formulas

### I. Emotional State Vector
- **VAD State**: `E(t) = [V(t), A(t), D(t)]`
  - Valence ∈ [-1, +1] (pleasantness)
  - Arousal ∈ [0, 1] (energy level)
  - Dominance ∈ [-1, +1] (control/submission)

- **Potential Energy**: `U_E = (1/2) * k_V * V² + (1/2) * k_A * A² + (1/2) * k_D * D²`
- **Emotional Force**: `F_E = -∇U_E = [-k_V*V, -k_A*A, -k_D*D]`
- **Stability**: `S_E = 1 - sqrt((V² + A² + D²) / 3)`

### II. Quantum Emotional Field
- **Wavefunction**: `|Ψ_E(t)⟩ = Σ α_i(t) |e_i⟩`
- **Probability Density**: `P_i = |α_i|²`
- **Hamiltonian**: `Ĥ_E = Σ ℏω_i |e_i⟩⟨e_i|`
- **Time Evolution**: `iℏ d|Ψ_E⟩/dt = Ĥ_E |Ψ_E⟩`
- **Interference**: `I(t) = |Ψ₁ + Ψ₂|²`
- **Entanglement**: Correlated emotional states between agents

### III. Emotion → Music
- **Base Frequency**: `f_E = f₀ * (1 + 0.4*A + 0.2*V)`
- **Harmonic Structure**: `H(t) = Σ a_n * sin(2π * n * f_E * t + φ_n)`
- **Chordal Shift**: `Δf = (V + D) * 30 Hz`
- **Resonance Energy**: `E_res = Σ a_i² * f_i`

### IV. Voice Modulation
- **Pitch**: `f₀ = f_base * (1 + 0.5*A + 0.3*V)`
- **Amplitude**: `A_voice = A_base * (1 + 0.4*D + 0.2*A)`
- **Formant Shift**: `F_i' = F_i * (1 + 0.2*V - 0.1*D)`
- **Vibrato**: `f₀(t) = f₀ * (1 + v_d * sin(2π * v_r * t))`
- **Speech Rate**: `R = R₀ * (1 + 0.7*A - 0.3*V)`
- **Voice Entropy**: `S_V = -Σ P_i * log(P_i)`

### V. Sound Synthesis
- **Timbre Spectrum**: `S(f,t) = A(t) * exp(-β(t) * f)`
- **Resonance Filter**: `H(f) = Π 1 / (1 + j*Q_i*(f/F_i' - F_i'/f))`
- **Intermodulation**: `s(t) = Σ a_i * a_j * cos(2π * (f_i ± f_j) * t)`

### VI. Network Dynamics
- **Emotional Coupling**: `dE_i/dt = Σ k_ij * (E_j - E_i)`
- **Coherence**: `C = (1/N²) * Σ cos(θ_i - θ_j)`
- **Phase Locking Value**: `PLV = |(1/N) * Σ exp(i*(φ₁ - φ₂))|`

### VII. Temporal & Memory
- **Temporal Decay**: `E(t+Δt) = E(t) * exp(-Δt / τ_E)`
- **Hysteresis**: `E(t) = E₀ + ∫ K(τ) * S(t-τ) dτ`
- **Emotional Momentum**: `p_E = m_E * dE/dt`

### VIII. Resonance & Coherence
- **Resonance Frequency**: `f_res = (1/(2π)) * sqrt(k_E / m_E)`
- **Beat Frequency**: `f_beat = |f₁ - f₂|`
- **Quality Factor**: `Q_E = f_res / Δf`
- **Coherence Energy**: `E_coh = ∫ |Ψ₁* * Ψ₂|² dx`

### IX. Unified Field Equation
```
iℏ dΨ_E/dt = Ĥ_E Ψ_E + g_M S(f,t) + g_V H(f) + g_N Σ k_ij (Ψ_j - Ψ_E)
```

## 📁 Files Created

1. **`quantum_emotional_field.py`** - Core implementation
   - `EmotionalStateVector` - VAD state representation
   - `QuantumEmotionalWavefunction` - Quantum state
   - `EmotionalHamiltonian` - Evolution operator
   - `MusicGenerator` - Emotion → music conversion
   - `VoiceModulator` - Voice parameter modulation
   - `EmotionalNetwork` - Multi-agent dynamics
   - `EmotionalMemory` - Temporal decay & hysteresis
   - `EmotionalResonance` - Resonance calculations
   - `QuantumEmotionalField` - Unified field system

2. **`visualize_quantum_emotional_field.py`** - Visualization generator
   - Wavefunction evolution
   - Energy landscapes
   - Network coherence
   - Resonance patterns

## 🎯 Usage Examples

### Basic Emotional State
```python
from quantum_emotional_field import EmotionalStateVector

state = EmotionalStateVector(valence=0.5, arousal=0.7, dominance=0.3)
energy = state.emotional_potential_energy()
stability = state.emotional_stability()
```

### Quantum Wavefunction
```python
from quantum_emotional_field import vad_to_wavefunction, QuantumEmotionalWavefunction

state = EmotionalStateVector(0.5, 0.7, 0.3)
wavefunction = vad_to_wavefunction(state)
probabilities = wavefunction.all_probabilities()
entropy = wavefunction.emotional_entropy()
```

### Music Generation
```python
from quantum_emotional_field import MusicGenerator
import numpy as np

state = EmotionalStateVector(0.5, 0.7, 0.3)
music_gen = MusicGenerator()
freq = music_gen.base_frequency_mapping(state)

t = np.linspace(0, 1, 44100)
harmonics = music_gen.harmonic_structure(state, t)
```

### Network Simulation
```python
from quantum_emotional_field import EmotionalNetwork

network = EmotionalNetwork(n_agents=5, coupling_strength=0.2)
network.emotional_coupling(dt=0.01)
coherence = network.coherence()
```

### Full Simulation
```python
from quantum_emotional_field import simulate_emotional_field, EmotionalStateVector

initial_state = EmotionalStateVector(0.5, 0.7, 0.3)
results = simulate_emotional_field(initial_state, duration=10.0, dt=0.01)

# Access results
valence_over_time = results['valence']
probabilities = results['probabilities']
entropy = results['entropy']
```

## 📊 Visualizations Generated

1. **`quantum_emotion_wavefunction.html`**
   - Emotion probabilities over time
   - VAD state evolution
   - Entropy and energy
   - Music frequency and voice pitch

2. **`quantum_emotion_energy_landscape.html`**
   - 3D energy landscape for different dominance values
   - Shows potential energy surface

3. **`quantum_emotion_network.html`**
   - Network coherence over time
   - Individual agent state evolution
   - Synchronization patterns

4. **`quantum_emotion_resonance.html`**
   - Wavefunction probabilities
   - Interference patterns
   - Resonance energy

## 🚀 Running

```bash
# Generate all visualizations
python3 visualize_quantum_emotional_field.py

# Or use in your own code
python3 -c "from quantum_emotional_field import *; ..."
```

## 🔬 Key Features

- **Quantum Mechanics**: Wavefunctions, Hamiltonians, interference, entanglement
- **Classical Mechanics**: Potential energy, forces, momentum
- **Network Dynamics**: Multi-agent coupling, coherence, synchronization
- **Music Synthesis**: Frequency mapping, harmonics, resonance
- **Voice Modulation**: Pitch, amplitude, formants, vibrato
- **Temporal Evolution**: Memory, decay, hysteresis
- **Unified Field**: All systems coupled through single equation

## 📈 Mathematical Completeness

The implementation covers:
- ✅ All 23 formula sections from the specification
- ✅ VAD state vectors
- ✅ Quantum wavefunctions
- ✅ Music generation
- ✅ Voice modulation
- ✅ Network dynamics
- ✅ Resonance & coherence
- ✅ Temporal memory
- ✅ Unified field equation

## 🔗 Integration

The QEF system integrates with:
- Existing emotion thesaurus (96 emotions, 8 categories)
- Musical parameter mappings
- Voice synthesis parameters
- Network/group dynamics
- Visualization systems

## 📝 Notes

- All formulas are implemented as specified
- Numerical stability handled with normalization
- Extensible architecture for adding new features
- Compatible with existing emotion visualization tools
- Ready for real-time applications

This is a complete, working implementation of the Quantum Emotional Field theory, ready for use in emotion-driven music generation, voice synthesis, and multi-agent emotional systems.
