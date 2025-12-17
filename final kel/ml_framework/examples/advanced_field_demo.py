"""
Advanced Quantum Emotional Field Demo

Demonstrates advanced field dynamics, network effects, and unified field equations.
"""

import numpy as np
import matplotlib.pyplot as plt
from cif_las_qef.emotion_models import (
    VADState, EmotionBasis,
    EmotionalPotential, QuantumEmotionalHamiltonian,
    EmotionalNetworkDynamics, PhysiologicalResonance,
    TemporalEmotionalDynamics, GeometricEmotionalProperties,
    QuantumEmotionalEntropy, ResonanceFormulas,
    UnifiedEmotionalField, EmotionalFieldController,
    EmotionColorMapper, QuantumEmotionalField
)


def demo_potential_energy():
    """Demonstrate emotional potential energy and force."""
    print("=== Emotional Potential Energy Demo ===\n")
    
    potential = EmotionalPotential(k_v=1.0, k_a=1.0, k_d=1.0)
    
    vad = VADState(valence=0.6, arousal=0.7, dominance=0.4)
    
    U = potential.compute_potential_energy(vad)
    force = potential.compute_force(vad)
    
    print(f"VAD: V={vad.valence:.2f}, A={vad.arousal:.2f}, D={vad.dominance:.2f}")
    print(f"Potential Energy: U_E = {U:.3f}")
    print(f"Force: F_E = [{force[0]:.3f}, {force[1]:.3f}, {force[2]:.3f}]")
    print(f"Equilibrium: {potential.find_equilibrium()}\n")


def demo_quantum_hamiltonian():
    """Demonstrate quantum emotional Hamiltonian."""
    print("=== Quantum Emotional Hamiltonian Demo ===\n")
    
    qf = QuantumEmotionalField()
    superposition = qf.create_superposition()
    
    hamiltonian = QuantumEmotionalHamiltonian(hbar=1.0)
    H = hamiltonian.build_hamiltonian(superposition)
    
    print(f"Hamiltonian matrix shape: {H.shape}")
    print(f"Diagonal elements (energies): {np.diag(H).real}")
    
    # Evolve
    evolved = hamiltonian.evolve_superposition(superposition, dt=0.1)
    print(f"Evolved probabilities: {evolved.get_probabilities()[:4]}\n")


def demo_network_dynamics():
    """Demonstrate network dynamics."""
    print("=== Network Dynamics Demo ===\n")
    
    network = EmotionalNetworkDynamics()
    
    E_i = VADState(valence=0.3, arousal=0.5, dominance=0.2)
    E_j = VADState(valence=0.6, arousal=0.7, dominance=0.4)
    
    coupling = network.compute_emotional_coupling(E_i, E_j, k_ij=0.5)
    print(f"Agent i: V={E_i.valence:.2f}, A={E_i.arousal:.2f}")
    print(f"Agent j: V={E_j.valence:.2f}, A={E_j.arousal:.2f}")
    print(f"Coupling contribution: [{coupling.valence:.3f}, {coupling.arousal:.3f}, {coupling.dominance:.3f}]")
    
    # Coherence
    phases = [0.0, 0.5, 1.0, 1.5]
    coherence = network.compute_coherence(phases)
    print(f"Coherence (4 agents): {coherence:.3f}")
    
    # Phase locking
    phase1 = np.linspace(0, 2*np.pi, 100)
    phase2 = phase1 + 0.1 * np.sin(phase1)
    plv = network.compute_phase_locking_value(phase1, phase2)
    print(f"Phase Locking Value: {plv:.3f}\n")


def demo_physiological_resonance():
    """Demonstrate physiological resonance."""
    print("=== Physiological Resonance Demo ===\n")
    
    bio = PhysiologicalResonance()
    
    E_bio = bio.compute_bio_energy(
        heart_rate=75.0,
        respiration_rate=15.0,
        gsr=0.6
    )
    print(f"Biological Energy: E_bio = {E_bio:.2f}")
    
    # Coupling constant
    k_bio = bio.compute_emotion_coupling_constant(
        delta_E=10.0,
        delta_H=5.0,
        delta_R=2.0,
        delta_G=0.1
    )
    print(f"Emotion-Bio Coupling: k_bio = {k_bio:.3f}")
    
    # Neural synchrony
    phase1 = np.linspace(0, 4*np.pi, 100)
    phase2 = phase1 + 0.2 * np.sin(phase1)
    synchrony, coherence = bio.compute_neural_phase_synchrony(phase1, phase2)
    print(f"Neural Coherence: {coherence:.3f}")
    
    # Total energy
    E_total = bio.compute_total_energy(
        E_emotion=50.0,
        E_bio=E_bio,
        E_env=10.0
    )
    print(f"Total Energy: E_total = {E_total:.2f}\n")


def demo_temporal_dynamics():
    """Demonstrate temporal dynamics."""
    print("=== Temporal Dynamics Demo ===\n")
    
    temporal = TemporalEmotionalDynamics()
    
    vad = VADState(valence=0.5, arousal=0.7, dominance=0.3)
    stability = temporal.compute_emotional_stability(vad)
    print(f"Emotional Stability: S_E = {stability:.3f}")
    
    # Drift
    E_eq = VADState(valence=0.0, arousal=0.5, dominance=0.0)
    drift = temporal.compute_emotional_drift(vad, E_eq, lambda_rate=0.1)
    print(f"Emotional Drift: [{drift.valence:.3f}, {drift.arousal:.3f}, {drift.dominance:.3f}]")
    
    # Decay
    decayed = temporal.compute_temporal_decay(vad, dt=0.5, tau=1.0)
    print(f"After decay (τ=1.0, dt=0.5): V={decayed.valence:.3f}, A={decayed.arousal:.3f}")
    
    # Momentum
    E_prev = VADState(valence=0.4, arousal=0.6, dominance=0.2)
    momentum, force = temporal.compute_emotional_momentum(vad, E_prev, dt=0.1)
    print(f"Emotional Momentum: {momentum}")
    print(f"Emotional Force: {force}\n")


def demo_geometric_properties():
    """Demonstrate geometric properties."""
    print("=== Geometric Properties Demo ===\n")
    
    geometric = GeometricEmotionalProperties()
    
    E1 = VADState(valence=0.5, arousal=0.7, dominance=0.3)
    E2 = VADState(valence=0.3, arousal=0.5, dominance=0.2)
    
    distance = geometric.compute_emotional_distance(E1, E2)
    print(f"Emotional Distance: d(E1, E2) = {distance:.3f}")
    
    # Curvature (simplified)
    E_dot = np.array([0.1, 0.2, 0.05])
    E_ddot = np.array([0.01, 0.02, 0.005])
    curvature = geometric.compute_curvature(E_dot, E_ddot)
    print(f"Emotional Curvature: κ = {curvature:.3f}\n")


def demo_entropy():
    """Demonstrate quantum entropy."""
    print("=== Quantum Entropy Demo ===\n")
    
    entropy_calc = QuantumEmotionalEntropy()
    
    probabilities = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    S = entropy_calc.compute_emotional_entropy(probabilities)
    print(f"Emotional Entropy: S_E = {S:.3f}")
    
    # Decoherence
    rho_t = entropy_calc.compute_decoherence(rho_0=1.0, t=2.0, Gamma=0.1)
    print(f"Decoherence (t=2.0, Γ=0.1): ρ(t) = {rho_t:.3f}\n")


def demo_resonance():
    """Demonstrate resonance formulas."""
    print("=== Resonance Formulas Demo ===\n")
    
    resonance = ResonanceFormulas()
    
    f_res = resonance.compute_resonance_frequency(k_e=10.0, m_e=1.0)
    print(f"Resonance Frequency: f_res = {f_res:.3f} Hz")
    
    f_beat = resonance.compute_beat_frequency(440.0, 445.0)
    print(f"Beat Frequency: f_beat = {f_beat:.1f} Hz")
    
    Q = resonance.compute_quality_factor(f_res=10.0, delta_f=1.0)
    print(f"Quality Factor: Q_E = {Q:.1f}")
    
    # Coherence energy
    psi1 = np.array([0.7, 0.3, 0.2], dtype=complex)
    psi2 = np.array([0.6, 0.4, 0.3], dtype=complex)
    E_coh = resonance.compute_resonant_coherence_energy(psi1, psi2)
    print(f"Resonant Coherence Energy: E_coh = {E_coh:.3f}\n")


def demo_unified_field():
    """Demonstrate unified field."""
    print("=== Unified Field Demo ===\n")
    
    unified = UnifiedEmotionalField(g_bio=0.3, g_net=0.2, g_res=0.1)
    
    # Lagrangian
    grad_psi = np.array([0.1+0.1j, 0.2+0.1j, 0.15+0.05j])
    U_e = 5.0
    lagrangian = unified.compute_lagrangian(
        grad_psi, U_e, E_bio=10.0, network_term=2.0, psi_magnitude=1.0
    )
    print(f"Lagrangian: L_QEF = {lagrangian:.3f}")
    
    # Total energy
    E_total = unified.compute_total_energy(
        E_emotion=50.0,
        E_music=20.0,
        E_voice=15.0,
        E_bio=10.0,
        E_network=5.0,
        E_resonance=3.0
    )
    print(f"Total QEF Energy: E_total = {E_total:.1f}\n")


def demo_controller():
    """Demonstrate field controller."""
    print("=== Field Controller Demo ===\n")
    
    controller = EmotionalFieldController(eta=0.1)
    potential = EmotionalPotential()
    
    E_current = VADState(valence=0.6, arousal=0.7, dominance=0.4)
    E_updated = controller.gradient_descent_step(E_current, potential)
    
    print(f"Before: V={E_current.valence:.2f}, A={E_current.arousal:.2f}")
    print(f"After gradient descent: V={E_updated.valence:.2f}, A={E_updated.arousal:.2f}")
    
    # Field center
    E_list = [
        VADState(0.5, 0.6, 0.3),
        VADState(0.4, 0.7, 0.2),
        VADState(0.6, 0.5, 0.4)
    ]
    center = controller.compute_field_center(E_list)
    print(f"Field Center: V={center.valence:.2f}, A={center.arousal:.2f}, D={center.dominance:.2f}\n")


def demo_color_mapping():
    """Demonstrate color mapping."""
    print("=== Color Mapping Demo ===\n")
    
    mapper = EmotionColorMapper()
    
    # Emotion to color
    joy_color = mapper.emotion_to_color(EmotionBasis.JOY)
    print(f"Joy: λ={joy_color.wavelength_nm:.0f} nm, f={joy_color.frequency_thz:.0f} THz")
    print(f"  RGB: {joy_color.rgb}")
    
    # VAD to color
    vad = VADState(valence=0.6, arousal=0.7, dominance=0.4)
    f_color = mapper.vad_to_color_frequency(vad)
    wavelength = mapper.frequency_to_wavelength(f_color)
    rgb = mapper.wavelength_to_rgb(wavelength)
    print(f"\nVAD {vad.valence:.2f}, {vad.arousal:.2f} → f={f_color:.0f} THz, λ={wavelength:.0f} nm")
    print(f"  RGB: {rgb}")
    
    # Direct VAD to RGB
    rgb_direct = mapper.vad_to_rgb(vad)
    print(f"  Direct RGB: {rgb_direct}\n")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Advanced Quantum Emotional Field Dynamics")
    print("=" * 60)
    print()
    
    demo_potential_energy()
    demo_quantum_hamiltonian()
    demo_network_dynamics()
    demo_physiological_resonance()
    demo_temporal_dynamics()
    demo_geometric_properties()
    demo_entropy()
    demo_resonance()
    demo_unified_field()
    demo_controller()
    demo_color_mapping()
    
    print("=" * 60)
    print("All advanced demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
