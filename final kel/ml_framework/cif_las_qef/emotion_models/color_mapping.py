"""
Color, Light, and Frequency Mappings

Maps emotions to visual/light properties for AR/visualization systems.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from .classical import EmotionBasis, VADState


@dataclass
class ColorMapping:
    """Color properties for an emotion."""
    wavelength_nm: float  # Wavelength in nanometers
    frequency_thz: float   # Frequency in THz
    energy_ev: float       # Energy in electron volts
    rgb: Tuple[float, float, float]  # RGB values (0-1)


class EmotionColorMapper:
    """
    Maps emotions to colors, wavelengths, and frequencies.
    """
    
    # Emotion to color mappings
    COLOR_MAPPINGS = {
        EmotionBasis.JOY: ColorMapping(
            wavelength_nm=580.0,  # Yellow
            frequency_thz=517.0,
            energy_ev=2.14,
            rgb=(1.0, 0.9, 0.0)
        ),
        EmotionBasis.SADNESS: ColorMapping(
            wavelength_nm=470.0,  # Blue
            frequency_thz=638.0,
            energy_ev=2.64,
            rgb=(0.2, 0.4, 0.8)
        ),
        EmotionBasis.ANGER: ColorMapping(
            wavelength_nm=620.0,  # Red
            frequency_thz=484.0,
            energy_ev=2.00,
            rgb=(0.9, 0.1, 0.1)
        ),
        EmotionBasis.FEAR: ColorMapping(
            wavelength_nm=400.0,  # Violet
            frequency_thz=749.0,
            energy_ev=3.10,
            rgb=(0.6, 0.2, 0.8)
        ),
        EmotionBasis.TRUST: ColorMapping(
            wavelength_nm=540.0,  # Green
            frequency_thz=556.0,
            energy_ev=2.30,
            rgb=(0.2, 0.8, 0.3)
        ),
    }
    
    def __init__(self):
        """Initialize color mapper."""
        # Visible light range
        self.f_min_thz = 400.0  # ~750 nm (red)
        self.f_max_thz = 750.0  # ~400 nm (violet)
    
    def emotion_to_color(self, emotion: EmotionBasis) -> ColorMapping:
        """
        Get color mapping for emotion.
        
        Args:
            emotion: Basic emotion
        
        Returns:
            ColorMapping
        """
        return self.COLOR_MAPPINGS.get(
            emotion,
            ColorMapping(550.0, 545.0, 2.25, (0.5, 0.5, 0.5))  # Default: green
        )
    
    def vad_to_color_frequency(self, vad: VADState) -> float:
        """
        Map VAD to color frequency: f_color = f_min + (V+1)(f_max - f_min)/2
        
        Args:
            vad: VAD state
        
        Returns:
            Frequency in THz
        """
        f_color = self.f_min_thz + (vad.valence + 1.0) * (self.f_max_thz - self.f_min_thz) / 2.0
        return float(np.clip(f_color, self.f_min_thz, self.f_max_thz))
    
    def frequency_to_wavelength(self, frequency_thz: float) -> float:
        """
        Convert frequency to wavelength: λ = c / f
        
        Args:
            frequency_thz: Frequency in THz
        
        Returns:
            Wavelength in nm
        """
        c = 299792458.0  # Speed of light in m/s
        wavelength_m = c / (frequency_thz * 1e12)
        wavelength_nm = wavelength_m * 1e9
        return float(wavelength_nm)
    
    def wavelength_to_rgb(self, wavelength_nm: float) -> Tuple[float, float, float]:
        """
        Convert wavelength to RGB (simplified approximation).
        
        Args:
            wavelength_nm: Wavelength in nm
        
        Returns:
            RGB tuple (0-1)
        """
        wavelength_nm = np.clip(wavelength_nm, 380.0, 780.0)
        
        # Simplified wavelength to RGB conversion
        if wavelength_nm < 440:
            r = -(wavelength_nm - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif wavelength_nm < 490:
            r = 0.0
            g = (wavelength_nm - 440) / (490 - 440)
            b = 1.0
        elif wavelength_nm < 580:
            r = 0.0
            g = 1.0
            b = -(wavelength_nm - 580) / (580 - 490)
        elif wavelength_nm < 645:
            r = (wavelength_nm - 580) / (645 - 580)
            g = 1.0
            b = 0.0
        else:
            r = 1.0
            g = -(wavelength_nm - 645) / (780 - 645)
            b = 0.0
        
        # Intensity adjustment
        if wavelength_nm < 420:
            intensity = 0.3 + 0.7 * (wavelength_nm - 380) / (420 - 380)
        elif wavelength_nm > 700:
            intensity = 0.3 + 0.7 * (780 - wavelength_nm) / (780 - 700)
        else:
            intensity = 1.0
        
        r = np.clip(r * intensity, 0.0, 1.0)
        g = np.clip(g * intensity, 0.0, 1.0)
        b = np.clip(b * intensity, 0.0, 1.0)
        
        return (float(r), float(g), float(b))
    
    def vad_to_rgb(self, vad: VADState) -> Tuple[float, float, float]:
        """
        Convert VAD directly to RGB.
        
        Args:
            vad: VAD state
        
        Returns:
            RGB tuple (0-1)
        """
        # Map VAD dimensions to RGB
        # Valence → Red-Green axis
        # Arousal → Brightness
        # Dominance → Saturation
        
        # Valence: -1 (blue) to +1 (red)
        r = (vad.valence + 1.0) / 2.0
        b = 1.0 - r
        
        # Arousal: brightness
        brightness = vad.arousal
        
        # Dominance: saturation
        saturation = abs(vad.dominance)
        
        # Apply brightness and saturation
        r = r * brightness + (1.0 - brightness) * 0.5
        g = saturation * 0.5 + (1.0 - saturation) * 0.5
        b = b * brightness + (1.0 - brightness) * 0.5
        
        return (
            float(np.clip(r, 0.0, 1.0)),
            float(np.clip(g, 0.0, 1.0)),
            float(np.clip(b, 0.0, 1.0))
        )
