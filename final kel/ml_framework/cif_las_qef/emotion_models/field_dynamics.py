"""
Advanced Quantum Emotional Field Dynamics

Implements:
- Emotional potential energy and force
- Quantum emotional Hamiltonian
- Network dynamics (diffusion, coupling, coherence)
- Physiological resonance
- Temporal and memory effects
- Geometric and topological properties
- Unified field equation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .classical import VADState
from .quantum import EmotionSuperposition


@dataclass
class EmotionalPotential:
    """
    Emotional Potential Energy
    
    U_E = (1/2) k_V V² + (1/2) k_A A² + (1/2) k_D D²
    """
    k_v: float = 1.0  # Valence stiffness constant
    k_a: float = 1.0  # Arousal stiffness constant
    k_d: float = 1.0  # Dominance stiffness constant
    
    def compute_potential_energy(self, vad: VADState) -> float:
        """
        Compute emotional potential energy.
        
        Args:
            vad: VAD state
        
        Returns:
            Potential energy
        """
        U = (
            0.5 * self.k_v * vad.valence**2 +
            0.5 * self.k_a * vad.arousal**2 +
            0.5 * self.k_d * vad.dominance**2
        )
        return float(U)
    
    def compute_force(self, vad: VADState) -> np.ndarray:
        """
        Compute emotional force: F_E = -∇U_E = [-k_V V, -k_A A, -k_D D]
        
        Args:
            vad: VAD state
        
        Returns:
            Force vector [F_V, F_A, F_D]
        """
        force = np.array([
            -self.k_v * vad.valence,
            -self.k_a * vad.arousal,
            -self.k_d * vad.dominance
        ])
        return force
    
    def find_equilibrium(self) -> VADState:
        """
        Find equilibrium point where F_E = 0.
        
        Returns:
            Equilibrium VAD state (neutral)
        """
        return VADState(valence=0.0, arousal=0.5, dominance=0.0)


class QuantumEmotionalHamiltonian:
    """
    Emotional Hamiltonian Operator
    
    Ĥ_E = Σ_i ℏω_i |e_i⟩⟨e_i|
    """
    
    def __init__(self, hbar: float = 1.0):
        """
        Initialize Hamiltonian.
        
        Args:
            hbar: Reduced Planck constant (emotional sensitivity)
        """
        self.hbar = hbar
        self.omega_frequencies: Optional[np.ndarray] = None
    
    def build_hamiltonian(
        self,
        superposition: EmotionSuperposition,
        omega_frequencies: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Build Hamiltonian matrix.
        
        Args:
            superposition: Emotion superposition
            omega_frequencies: Optional frequencies (default: from superposition)
        
        Returns:
            Hamiltonian matrix
        """
        n = len(superposition.basis_emotions)
        
        if omega_frequencies is None:
            # Default: use probabilities as frequencies
            probs = superposition.get_probabilities()
            omega_frequencies = 2 * np.pi * (0.5 + probs)  # 0.5-1.5 Hz range
        
        self.omega_frequencies = omega_frequencies
        
        # Build diagonal Hamiltonian: H = Σ ℏω_i |i⟩⟨i|
        H = np.zeros((n, n), dtype=complex)
        for i in range(n):
            H[i, i] = self.hbar * omega_frequencies[i]
        
        return H
    
    def evolve_superposition(
        self,
        superposition: EmotionSuperposition,
        dt: float = 0.01
    ) -> EmotionSuperposition:
        """
        Evolve superposition: iℏ d|Ψ_E⟩/dt = Ĥ_E |Ψ_E⟩
        
        Args:
            superposition: Current superposition
            dt: Time step
        
        Returns:
            Evolved superposition
        """
        H = self.build_hamiltonian(superposition)
        
        # Time evolution: |Ψ(t+dt)⟩ = exp(-i H dt / ℏ) |Ψ(t)⟩
        # Simplified: |Ψ(t+dt)⟩ ≈ (1 - i H dt / ℏ) |Ψ(t)⟩
        evolution_op = np.eye(len(superposition.amplitudes), dtype=complex) - \
                       1j * H * dt / self.hbar
        
        new_amplitudes = evolution_op @ superposition.amplitudes
        
        # Normalize
        norm = np.linalg.norm(new_amplitudes)
        if norm > 0:
            new_amplitudes = new_amplitudes / norm
        
        return EmotionSuperposition(
            basis_emotions=superposition.basis_emotions,
            amplitudes=new_amplitudes,
            phases=np.angle(new_amplitudes)
        )


class EmotionalNetworkDynamics:
    """
    Network dynamics for multi-agent emotional systems.
    """
    
    def __init__(self):
        """Initialize network dynamics."""
        pass
    
    def compute_emotional_coupling(
        self,
        E_i: VADState,
        E_j: VADState,
        k_ij: float = 1.0
    ) -> VADState:
        """
        Compute coupling: dE_i/dt = Σ_j k_ij (E_j - E_i)
        
        Args:
            E_i: Current agent's VAD
            E_j: Neighbor agent's VAD
            k_ij: Coupling strength
        
        Returns:
            Coupling contribution to dE_i/dt
        """
        delta = np.array([
            E_j.valence - E_i.valence,
            E_j.arousal - E_i.arousal,
            E_j.dominance - E_i.dominance
        ])
        
        coupling = k_ij * delta
        
        return VADState(
            valence=float(coupling[0]),
            arousal=float(coupling[1]),
            dominance=float(coupling[2])
        )
    
    def compute_coherence(
        self,
        phases: List[float]
    ) -> float:
        """
        Compute coherence: C = (1/N²) Σ_i,j cos(θ_i - θ_j)
        
        Args:
            phases: List of emotional phases
        
        Returns:
            Coherence (0-1)
        """
        if len(phases) < 2:
            return 1.0
        
        N = len(phases)
        coherence_sum = 0.0
        
        for i in range(N):
            for j in range(N):
                coherence_sum += np.cos(phases[i] - phases[j])
        
        coherence = coherence_sum / (N * N)
        
        # Normalize to 0-1
        coherence = (coherence + 1.0) / 2.0
        
        return float(np.clip(coherence, 0.0, 1.0))
    
    def compute_phase_locking_value(
        self,
        phase1: np.ndarray,
        phase2: np.ndarray
    ) -> float:
        """
        Compute Phase Locking Value: PLV = |(1/N) Σ_n e^(i(φ₁(n) - φ₂(n)))|
        
        Args:
            phase1: First phase time series
            phase2: Second phase time series
        
        Returns:
            PLV (0-1)
        """
        if len(phase1) != len(phase2):
            raise ValueError("Phase arrays must have same length")
        
        phase_diff = phase1 - phase2
        complex_sum = np.mean(np.exp(1j * phase_diff))
        plv = np.abs(complex_sum)
        
        return float(plv)
    
    def compute_weighted_connectivity(
        self,
        x_i: np.ndarray,
        x_j: np.ndarray,
        E_i: VADState,
        E_j: VADState,
        L: float = 1.0
    ) -> float:
        """
        Compute weighted connectivity: K_ij = e^(-||x_i - x_j||/L) / (1 + |E_i - E_j|)
        
        Args:
            x_i: Position of agent i
            x_j: Position of agent j
            E_i: Emotional state of agent i
            E_j: Emotional state of agent j
            L: Correlation length
        
        Returns:
            Connectivity strength
        """
        # Spatial distance
        spatial_dist = np.linalg.norm(x_i - x_j)
        spatial_factor = np.exp(-spatial_dist / L)
        
        # Emotional distance
        E_i_array = np.array([E_i.valence, E_i.arousal, E_i.dominance])
        E_j_array = np.array([E_j.valence, E_j.arousal, E_j.dominance])
        emotional_dist = np.linalg.norm(E_i_array - E_j_array)
        
        # Connectivity
        K_ij = spatial_factor / (1.0 + emotional_dist)
        
        return float(K_ij)


class PhysiologicalResonance:
    """
    Physiological resonance: emotional energy ↔ biological signals.
    """
    
    def __init__(self):
        """Initialize physiological resonance."""
        self.alpha_h = 0.4  # Heart rate coupling
        self.alpha_r = 0.3  # Respiration coupling
        self.alpha_g = 0.3  # GSR coupling
    
    def compute_bio_energy(
        self,
        heart_rate: float,
        respiration_rate: float,
        gsr: float
    ) -> float:
        """
        Compute biological energy: E_bio = α_H H + α_R R + α_G G
        
        Args:
            heart_rate: Heart rate (BPM)
            respiration_rate: Respiration rate (breaths/min)
            gsr: Galvanic skin response (0-1)
        
        Returns:
            Biological energy
        """
        E_bio = (
            self.alpha_h * heart_rate +
            self.alpha_r * respiration_rate +
            self.alpha_g * gsr * 100.0  # Scale GSR
        )
        return float(E_bio)
    
    def compute_emotion_coupling_constant(
        self,
        delta_E: float,
        delta_H: float,
        delta_R: float,
        delta_G: float
    ) -> float:
        """
        Compute coupling constant: k_bio = dE / dE_bio
        
        Args:
            delta_E: Change in emotional energy
            delta_H: Change in heart rate
            delta_R: Change in respiration
            delta_G: Change in GSR
        
        Returns:
            Coupling constant
        """
        delta_E_bio = (
            self.alpha_h * delta_H +
            self.alpha_r * delta_R +
            self.alpha_g * delta_G * 100.0
        )
        
        if abs(delta_E_bio) < 1e-8:
            return 0.0
        
        k_bio = delta_E / delta_E_bio
        return float(k_bio)
    
    def compute_neural_phase_synchrony(
        self,
        phase1: np.ndarray,
        phase2: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute neural phase synchrony: Φ(t) = cos(Δφ(t))
        
        Args:
            phase1: First phase time series
            phase2: Second phase time series
        
        Returns:
            (synchrony_time_series, average_coherence)
        """
        phase_diff = phase1 - phase2
        synchrony = np.cos(phase_diff)
        
        # Average coherence
        coherence = np.mean(synchrony)
        
        return synchrony, float(coherence)
    
    def compute_total_energy(
        self,
        E_emotion: float,
        E_bio: float,
        E_env: float = 0.0,
        beta: float = 0.3,
        gamma: float = 0.2
    ) -> float:
        """
        Compute total energy: E_total = E_emotion + β E_bio + γ E_env
        
        Args:
            E_emotion: Emotional energy
            E_bio: Biological energy
            E_env: Environmental energy
            beta: Bio coupling
            gamma: Environment coupling
        
        Returns:
            Total energy
        """
        return float(E_emotion + beta * E_bio + gamma * E_env)


class TemporalEmotionalDynamics:
    """
    Temporal dynamics: memory, decay, momentum.
    """
    
    def __init__(self):
        """Initialize temporal dynamics."""
        pass
    
    def compute_emotional_stability(self, vad: VADState) -> float:
        """
        Compute stability: S_E = 1 - sqrt((V² + A² + D²) / 3)
        
        Args:
            vad: VAD state
        
        Returns:
            Stability index (0-1)
        """
        magnitude = np.sqrt(
            (vad.valence**2 + vad.arousal**2 + vad.dominance**2) / 3.0
        )
        stability = 1.0 - magnitude
        return float(np.clip(stability, 0.0, 1.0))
    
    def compute_emotional_drift(
        self,
        E_current: VADState,
        E_equilibrium: VADState,
        lambda_rate: float = 0.1,
        noise: float = 0.0
    ) -> VADState:
        """
        Compute drift: dE/dt = -λ(E - E_eq) + η(t)
        
        Args:
            E_current: Current VAD state
            E_equilibrium: Equilibrium VAD state
            lambda_rate: Recovery rate
            noise: Stochastic noise
        
        Returns:
            Drift vector
        """
        delta = np.array([
            E_current.valence - E_equilibrium.valence,
            E_current.arousal - E_equilibrium.arousal,
            E_current.dominance - E_equilibrium.dominance
        ])
        
        drift = -lambda_rate * delta
        
        # Add noise
        if noise > 0:
            drift += np.random.normal(0, noise, 3)
        
        return VADState(
            valence=float(drift[0]),
            arousal=float(drift[1]),
            dominance=float(drift[2])
        )
    
    def compute_temporal_decay(
        self,
        E: VADState,
        dt: float,
        tau: float = 1.0
    ) -> VADState:
        """
        Compute temporal decay: E(t+Δt) = E(t) e^(-Δt/τ)
        
        Args:
            E: Current VAD state
            dt: Time step
            tau: Half-life constant
        
        Returns:
            Decayed VAD state
        """
        decay_factor = np.exp(-dt / tau)
        
        return VADState(
            valence=E.valence * decay_factor,
            arousal=0.5 + (E.arousal - 0.5) * decay_factor,  # Decay toward 0.5
            dominance=E.dominance * decay_factor
        )
    
    def compute_emotional_momentum(
        self,
        E_current: VADState,
        E_previous: VADState,
        dt: float,
        m_e: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute momentum: p_E = m_E dE/dt, F_E = dp_E/dt
        
        Args:
            E_current: Current VAD state
            E_previous: Previous VAD state
            dt: Time step
            m_e: Emotional mass
        
        Returns:
            (momentum, force)
        """
        # Velocity
        dE_dt = np.array([
            (E_current.valence - E_previous.valence) / dt,
            (E_current.arousal - E_previous.arousal) / dt,
            (E_current.dominance - E_previous.dominance) / dt
        ])
        
        # Momentum
        momentum = m_e * dE_dt
        
        # Force (simplified: F = m d²E/dt², approximated as F = m dE/dt / dt)
        force = m_e * dE_dt / dt
        
        return momentum, force


class GeometricEmotionalProperties:
    """
    Geometric and topological properties of emotional space.
    """
    
    def __init__(self):
        """Initialize geometric properties."""
        pass
    
    def compute_curvature(
        self,
        E_dot: np.ndarray,
        E_ddot: np.ndarray
    ) -> float:
        """
        Compute curvature: κ = ||Ė × Ë|| / ||Ė||³
        
        Args:
            E_dot: First derivative (velocity)
            E_ddot: Second derivative (acceleration)
        
        Returns:
            Curvature
        """
        if np.linalg.norm(E_dot) < 1e-8:
            return 0.0
        
        cross_product = np.cross(E_dot, E_dot)
        curvature = np.linalg.norm(cross_product) / (np.linalg.norm(E_dot)**3 + 1e-8)
        
        return float(curvature)
    
    def compute_emotional_distance(
        self,
        E1: VADState,
        E2: VADState
    ) -> float:
        """
        Compute distance: d(E1, E2) = sqrt((V1-V2)² + (A1-A2)² + (D1-D2)²)
        
        Args:
            E1: First VAD state
            E2: Second VAD state
        
        Returns:
            Distance
        """
        delta = np.array([
            E1.valence - E2.valence,
            E1.arousal - E2.arousal,
            E1.dominance - E2.dominance
        ])
        
        distance = np.linalg.norm(delta)
        return float(distance)
    
    def find_attractors(
        self,
        potential: 'EmotionalPotential',
        grid_resolution: int = 10
    ) -> List[VADState]:
        """
        Find emotional attractors: ∇U_E = 0, det(∇²U_E) > 0
        
        Args:
            potential: Emotional potential
            grid_resolution: Grid resolution for search
        
        Returns:
            List of attractor states
        """
        # Simplified: equilibrium is at origin
        attractors = [potential.find_equilibrium()]
        
        # Could add more sophisticated search for other attractors
        return attractors


class QuantumEmotionalEntropy:
    """
    Quantum entropy and information measures.
    """
    
    def __init__(self):
        """Initialize entropy calculator."""
        pass
    
    def compute_emotional_entropy(
        self,
        probabilities: np.ndarray
    ) -> float:
        """
        Compute entropy: S_E = -Σ P_i ln(P_i)
        
        Args:
            probabilities: Probability distribution
        
        Returns:
            Entropy
        """
        # Avoid log(0)
        probabilities = probabilities + 1e-10
        probabilities = probabilities / np.sum(probabilities)  # Normalize
        
        entropy = -np.sum(probabilities * np.log(probabilities))
        return float(entropy)
    
    def compute_mutual_information(
        self,
        P_AB: np.ndarray,
        P_A: np.ndarray,
        P_B: np.ndarray
    ) -> float:
        """
        Compute mutual information: I(E_A; E_B) = Σ P(E_A, E_B) ln(P(E_A, E_B) / (P(E_A) P(E_B)))
        
        Args:
            P_AB: Joint probability
            P_A: Marginal probability A
            P_B: Marginal probability B
        
        Returns:
            Mutual information
        """
        # Avoid log(0)
        P_AB = P_AB + 1e-10
        P_A = P_A + 1e-10
        P_B = P_B + 1e-10
        
        # Normalize
        P_AB = P_AB / np.sum(P_AB)
        P_A = P_A / np.sum(P_A)
        P_B = P_B / np.sum(P_B)
        
        # Compute MI
        mi = 0.0
        for i in range(len(P_A)):
            for j in range(len(P_B)):
                if P_AB[i, j] > 0:
                    mi += P_AB[i, j] * np.log(
                        P_AB[i, j] / (P_A[i] * P_B[j] + 1e-10)
                    )
        
        return float(mi)
    
    def compute_decoherence(
        self,
        rho_0: float,
        t: float,
        Gamma: float = 0.1
    ) -> float:
        """
        Compute decoherence: ρ(t) = ρ_0 e^(-Γt)
        
        Args:
            rho_0: Initial coherence
            t: Time
            Gamma: Decoherence rate
        
        Returns:
            Coherence at time t
        """
        return float(rho_0 * np.exp(-Gamma * t))


class ResonanceFormulas:
    """
    Resonance and coherence formulas.
    """
    
    def __init__(self):
        """Initialize resonance calculator."""
        pass
    
    def compute_resonance_frequency(
        self,
        k_e: float,
        m_e: float
    ) -> float:
        """
        Compute resonance frequency: f_res = (1/2π) sqrt(k_E / m_E)
        
        Args:
            k_e: Emotional stiffness
            m_e: Emotional mass
        
        Returns:
            Resonance frequency
        """
        if m_e <= 0:
            return 0.0
        
        f_res = (1.0 / (2 * np.pi)) * np.sqrt(k_e / m_e)
        return float(f_res)
    
    def compute_beat_frequency(
        self,
        f1: float,
        f2: float
    ) -> float:
        """
        Compute beat frequency: f_beat = |f1 - f2|
        
        Args:
            f1: First frequency
            f2: Second frequency
        
        Returns:
            Beat frequency
        """
        return float(abs(f1 - f2))
    
    def compute_quality_factor(
        self,
        f_res: float,
        delta_f: float
    ) -> float:
        """
        Compute quality factor: Q_E = f_res / Δf
        
        Args:
            f_res: Resonance frequency
            delta_f: Bandwidth
        
        Returns:
            Quality factor
        """
        if delta_f <= 0:
            return float('inf')
        
        return float(f_res / delta_f)
    
    def compute_resonant_coherence_energy(
        self,
        psi1: np.ndarray,
        psi2: np.ndarray
    ) -> float:
        """
        Compute resonant coherence energy: E_coh = ∫ |Ψ₁* Ψ₂|² dx
        
        Args:
            psi1: First wavefunction
            psi2: Second wavefunction
        
        Returns:
            Coherence energy
        """
        # Simplified: discrete integration
        overlap = np.conj(psi1) * psi2
        coherence_energy = np.sum(np.abs(overlap)**2)
        
        return float(coherence_energy)


class UnifiedEmotionalField:
    """
    Unified Quantum Emotional Field with Lagrangian.
    """
    
    def __init__(
        self,
        g_bio: float = 0.3,
        g_net: float = 0.2,
        g_res: float = 0.1
    ):
        """
        Initialize unified field.
        
        Args:
            g_bio: Bio coupling constant
            g_net: Network coupling constant
            g_res: Resonance coupling constant
        """
        self.g_bio = g_bio
        self.g_net = g_net
        self.g_res = g_res
    
    def compute_lagrangian(
        self,
        grad_psi: np.ndarray,
        U_e: float,
        E_bio: float = 0.0,
        network_term: float = 0.0,
        psi_magnitude: float = 1.0
    ) -> float:
        """
        Compute Lagrangian: L = (1/2)|∇Ψ_E|² - U_E + g_bio E_bio + g_net Σ K_ij(E_j-E_i)² + g_res |Ψ_E|⁴
        
        Args:
            grad_psi: Gradient of wavefunction
            U_e: Potential energy
            E_bio: Biological energy
            network_term: Network coupling term
            psi_magnitude: |Ψ_E| magnitude
        
        Returns:
            Lagrangian
        """
        kinetic = 0.5 * np.sum(np.abs(grad_psi)**2)
        potential = U_e
        bio_term = self.g_bio * E_bio
        net_term = self.g_net * network_term
        resonance_term = self.g_res * (psi_magnitude**4)
        
        lagrangian = kinetic - potential + bio_term + net_term + resonance_term
        
        return float(lagrangian)
    
    def compute_total_energy(
        self,
        E_emotion: float,
        E_music: float,
        E_voice: float,
        E_bio: float,
        E_network: float,
        E_resonance: float
    ) -> float:
        """
        Compute total QEF energy.
        
        Args:
            E_emotion: Emotional energy
            E_music: Music energy
            E_voice: Voice energy
            E_bio: Biological energy
            E_network: Network energy
            E_resonance: Resonance energy
        
        Returns:
            Total energy
        """
        return float(
            E_emotion + E_music + E_voice +
            E_bio + E_network + E_resonance
        )
    
    def solve_wave_equation(
        self,
        psi_initial: np.ndarray,
        c_e: float = 1.0,
        gamma: float = 0.1,
        mu: float = 0.5,
        source: Optional[np.ndarray] = None,
        dt: float = 0.01,
        dx: float = 0.1,
        n_steps: int = 100
    ) -> np.ndarray:
        """
        Solve wave equation: ∂²Ψ_E/∂t² - c_E²∇²Ψ_E + γ∂Ψ_E/∂t + μ²Ψ_E = S(x,t)
        
        Args:
            psi_initial: Initial wavefunction
            c_e: Emotional propagation velocity
            gamma: Damping constant
            mu: Emotional mass term
            source: Source term S(x,t)
            dt: Time step
            dx: Spatial step
            n_steps: Number of time steps
        
        Returns:
            Wavefunction evolution
        """
        # Simplified 1D wave equation solver
        n_points = len(psi_initial)
        psi = np.zeros((n_steps, n_points), dtype=complex)
        psi[0] = psi_initial
        
        if n_steps > 1:
            # Initialize second time step (simplified)
            psi[1] = psi[0] * (1.0 - gamma * dt)
        
        for n in range(1, n_steps - 1):
            # Laplacian (simplified 1D)
            laplacian = np.zeros_like(psi[n])
            for i in range(1, n_points - 1):
                laplacian[i] = (psi[n, i+1] - 2*psi[n, i] + psi[n, i-1]) / (dx**2)
            
            # Wave equation
            d2psi_dt2 = (
                c_e**2 * laplacian -
                gamma * (psi[n] - psi[n-1]) / dt -
                mu**2 * psi[n]
            )
            
            if source is not None:
                d2psi_dt2 += source[n] if len(source.shape) > 1 else source
            
            # Update
            psi[n+1] = 2*psi[n] - psi[n-1] + d2psi_dt2 * dt**2
        
        return psi


class EmotionalFieldController:
    """
    Navigation and control for emotional fields.
    """
    
    def __init__(self, eta: float = 0.1):
        """
        Initialize controller.
        
        Args:
            eta: Learning rate (emotional adaptation speed)
        """
        self.eta = eta
    
    def gradient_descent_step(
        self,
        E_current: VADState,
        potential: 'EmotionalPotential'
    ) -> VADState:
        """
        Gradient descent: E(t+Δt) = E(t) - η ∇U_E
        
        Args:
            E_current: Current VAD state
            potential: Emotional potential
        
        Returns:
            Updated VAD state
        """
        force = potential.compute_force(E_current)
        
        new_E = VADState(
            valence=E_current.valence - self.eta * force[0],
            arousal=E_current.arousal - self.eta * force[1],
            dominance=E_current.dominance - self.eta * force[2]
        )
        
        new_E.clip()
        return new_E
    
    def compute_field_center(
        self,
        E_list: List[VADState],
        weights: Optional[List[float]] = None
    ) -> VADState:
        """
        Compute field center: E_center = Σ w_i E_i / Σ w_i
        
        Args:
            E_list: List of VAD states
            weights: Optional weights
        
        Returns:
            Center VAD state
        """
        if weights is None:
            weights = [1.0] * len(E_list)
        
        total_weight = sum(weights)
        if total_weight == 0:
            return VADState()
        
        center = VADState()
        for E, w in zip(E_list, weights):
            center.valence += w * E.valence
            center.arousal += w * E.arousal
            center.dominance += w * E.dominance
        
        center.valence /= total_weight
        center.arousal /= total_weight
        center.dominance /= total_weight
        
        return center
    
    def adaptive_regulation(
        self,
        E_error: float,
        alpha: float = 0.1,
        beta: float = 0.01
    ) -> float:
        """
        Adaptive regulation: dη/dt = α E_error - β η
        
        Args:
            E_error: Emotional error
            alpha: Error gain
            beta: Decay rate
        
        Returns:
            Updated learning rate
        """
        d_eta = alpha * E_error - beta * self.eta
        self.eta = max(0.01, min(1.0, self.eta + d_eta))
        return self.eta
