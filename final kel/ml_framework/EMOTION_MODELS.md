# Emotional Models - Base Layer

## Overview

This module implements classical and quantum emotional models as the foundational layer for the CIF/LAS/QEF framework. It provides mathematical models for emotional states, superposition, interference, and entanglement.

---

## 1. Classical Emotional Models

### A. VAD (Valence-Arousal-Dominance) Model

**Class**: `VADModel`

A 3D coordinate system for emotional states:

**E = (V, A, D)**

| Symbol | Meaning | Range |
|--------|---------|-------|
| V | Valence (pleasant–unpleasant) | [-1, 1] |
| A | Arousal (calm–excited) | [0, 1] |
| D | Dominance (control–submission) | [-1, 1] |

**Formulas Implemented**:

1. **Energy Level**: `E_n = A × (1 + |V|)`
2. **Emotional Tension**: `T = |V| × (1 - D)`
3. **Stability Index**: `S = 1 - sqrt((V² + A² + D²) / 3)`

**Usage**:
```python
from cif_las_qef.emotion_models import VADModel, VADState

vad_model = VADModel()
vad = VADState(valence=0.5, arousal=0.7, dominance=0.3)
metrics = vad_model.compute_all_metrics(vad)
```

### B. Plutchik's Wheel

**Class**: `PlutchikWheel`

Eight basic emotions with intensity and relationships:

| Emotion | V | A | D | Relation |
|---------|---|---|---|----------|
| Joy | +1.0 | +0.6 | +0.3 | Opposite of Sadness |
| Trust | +0.5 | +0.4 | +0.2 | Combines with Joy → Love |
| Fear | -0.6 | +0.8 | -0.4 | Opposite of Anger |
| Surprise | 0.0 | +0.9 | -0.1 | Neutral valence, high arousal |
| Sadness | -0.8 | +0.3 | -0.2 | Opposite of Joy |
| Disgust | -0.7 | +0.5 | -0.3 | Opposite of Trust |
| Anger | -0.5 | +0.9 | +0.4 | Opposite of Fear |
| Anticipation | +0.3 | +0.7 | +0.1 | Opposite of Surprise |

**Emotion Intensity Formula**: `k × A × (1 + V)`

**Usage**:
```python
from cif_las_qef.emotion_models import PlutchikWheel, EmotionBasis

plutchik = PlutchikWheel()
joy_vad = plutchik.emotion_to_vad(EmotionBasis.JOY, intensity=1.0)
love_vad = plutchik.combine_emotions(EmotionBasis.JOY, EmotionBasis.TRUST)
```

---

## 2. Quantum Emotional Field

### A. Emotion Superposition

**Class**: `EmotionSuperposition`

Each emotional state is a superposition of basis emotions:

**|Ψ_E⟩ = Σ α_i |e_i⟩**

where:
- `|e_i⟩` = basic emotion basis vector
- `α_i ∈ C` = complex amplitude (intensity + phase)
- Normalization: `Σ |α_i|² = 1`
- `|α_i|²` = probability of experiencing emotion `e_i`

**Features**:
- Collapse function: `|Ψ_E⟩ → |e_j⟩` with probability `|α_j|²`
- Coherence: `C = |Σ α_i|`
- Entropy: `S = -Σ |α_i|² log(|α_i|²)`

**Usage**:
```python
from cif_las_qef.emotion_models import QuantumEmotionalField

qf = QuantumEmotionalField()
superposition = qf.create_superposition()
probabilities = superposition.get_probabilities()
collapsed_emotion, prob = superposition.collapse()
```

### B. Emotional Interference

**Class**: `EmotionalInterference`

Models interference between overlapping emotional fields:

**I = |Ψ₁ + Ψ₂|² = |Ψ₁|² + |Ψ₂|² + 2 Re(Ψ₁* Ψ₂)**

- **Constructive interference** = empathy / resonance
- **Destructive interference** = cognitive dissonance

**Usage**:
```python
from cif_las_qef.emotion_models import EmotionalInterference

interference = EmotionalInterference()
result = interference.compute_interference(psi1, psi2)
```

### C. Emotional Entanglement

**Class**: `EmotionalEntanglement`

Two agents (A and B) share a coupled emotional state:

**|Ψ_AB⟩ = (1/√2)(|Joy_A, Joy_B⟩ + |Fear_A, Fear_B⟩)**

Observation on A instantaneously influences B's emotional probability field.

**Usage**:
```python
from cif_las_qef.emotion_models import EmotionalEntanglement

entanglement = EmotionalEntanglement(
    agent_a_emotions=[EmotionBasis.JOY, EmotionBasis.FEAR],
    agent_b_emotions=[EmotionBasis.JOY, EmotionBasis.FEAR]
)
result = entanglement.observe_agent_a(EmotionBasis.JOY)
```

### D. Quantum Emotional Energy

**Formula**: `E_emotion = ℏω(n + 1/2)`

where:
- `n`: emotional excitation level (0 = calm, 1 = agitated, ...)
- `ω`: frequency of emotional fluctuation
- `ℏ`: emotional sensitivity constant

**Emotional Temperature**: `T_E = k_B^(-1) E_emotion`

**Usage**:
```python
energy = qf.compute_quantum_energy(superposition, omega=1.0, n=1)
temperature = qf.compute_emotional_temperature(energy)
```

---

## 3. Hybrid Emotional Field

**Class**: `HybridEmotionalField`

Combines classical and quantum parts:

**F_E(t) = VAD(t) + Re[Σ α_i(t) e^(iφ_i(t)) |e_i⟩]**

**Time Evolution**: `dΨ/dt = -i Ĥ Ψ`

where `Ĥ` = Emotional Hamiltonian (memory, stimuli, empathy links)

**Usage**:
```python
from cif_las_qef.emotion_models import HybridEmotionalField, VADState

hybrid = HybridEmotionalField()
hybrid.initialize(initial_vad=VADState(valence=0.3, arousal=0.6))
field_state = hybrid.compute_field(t=0.0)
hybrid.evolve(dt=0.01)
observation = hybrid.observe()  # Collapse
```

---

## 4. Simulation and Visualization

**Class**: `EmotionalFieldSimulator`

Provides simulation and plotting capabilities:

**Methods**:
- `simulate()`: Run time evolution simulation
- `plot_quantum_oscillations()`: Plot emotion amplitudes over time
- `plot_vad_evolution()`: Plot VAD coordinates over time
- `plot_probability_evolution()`: Plot emotion probabilities
- `plot_coherence_entropy()`: Plot coherence and entropy
- `demonstrate_interference()`: Show interference effects
- `demonstrate_entanglement()`: Show entanglement correlations

**Usage**:
```python
from cif_las_qef.emotion_models import EmotionalFieldSimulator

simulator = EmotionalFieldSimulator()
simulator.simulate(duration=10.0, dt=0.01)
simulator.plot_quantum_oscillations()
simulator.plot_vad_evolution()
```

---

## 5. Integration with Framework

The emotional models integrate with existing components:

### CIF Integration
- SFL can use VAD model for bio-to-ESV mapping
- ASL can use quantum superposition for hybrid output

### LAS Integration
- EI can use VAD and Plutchik for emotion processing
- ABC can use quantum models for intent formation

### QEF Integration
- LEN can use VAD for local emotion capture
- QSL can use quantum superposition for synchronization
- PRL can use hybrid field for global resonance

---

## Example: Complete Workflow

```python
from cif_las_qef.emotion_models import (
    VADModel, VADState,
    QuantumEmotionalField,
    HybridEmotionalField,
    EmotionalFieldSimulator
)

# 1. Classical VAD
vad_model = VADModel()
vad = VADState(valence=0.5, arousal=0.7, dominance=0.3)
metrics = vad_model.compute_all_metrics(vad)

# 2. Quantum Superposition
qf = QuantumEmotionalField()
superposition = qf.create_superposition()

# 3. Hybrid Field
hybrid = HybridEmotionalField()
hybrid.initialize(vad, superposition)
field_state = hybrid.compute_field(t=0.0)

# 4. Simulation
simulator = EmotionalFieldSimulator()
simulator.simulate(duration=10.0, initial_vad=vad)
simulator.plot_vad_evolution()
```

---

## Mathematical Reference

### Classical Formulas

- **Energy**: `E_n = A × (1 + |V|)`
- **Tension**: `T = |V| × (1 - D)`
- **Stability**: `S = 1 - sqrt((V² + A² + D²) / 3)`
- **Intensity**: `I = k × A × (1 + V)`

### Quantum Formulas

- **Superposition**: `|Ψ_E⟩ = Σ α_i |e_i⟩`
- **Normalization**: `Σ |α_i|² = 1`
- **Interference**: `I = |Ψ₁ + Ψ₂|²`
- **Energy**: `E = ℏω(n + 1/2)`
- **Evolution**: `dΨ/dt = -i Ĥ Ψ`

### Hybrid Formula

- **Field**: `F_E(t) = VAD(t) + Re[Σ α_i(t) e^(iφ_i(t)) |e_i⟩]`

---

## Status

✅ **All models implemented**
✅ **Formulas verified**
✅ **Simulation capabilities added**
✅ **Visualization tools included**
✅ **Integration with framework complete**

---

## Files

- `classical.py`: VAD Model, Plutchik's Wheel
- `quantum.py`: Superposition, Interference, Entanglement
- `hybrid.py`: Hybrid Field Equation
- `simulation.py`: Simulation and Visualization

---

**Version**: 0.1.0-alpha  
**Status**: Research Phase - Complete
