#!/usr/bin/env python3
"""
Biometric Client - Python interface for biometric integration
==============================================================
Provides Python interface to biometric data for use with Kelly vocal system.
"""

import time
from typing import Optional, Dict, Callable
from dataclasses import dataclass


@dataclass
class BiometricReading:
    """Biometric data reading."""
    heart_rate: Optional[float] = None  # BPM
    heart_rate_variability: Optional[float] = None  # ms
    skin_conductance: Optional[float] = None  # microsiemens
    temperature: Optional[float] = None  # Celsius
    timestamp: float = 0.0

    def is_valid(self) -> bool:
        """Check if reading has valid data."""
        return self.heart_rate is not None and self.heart_rate > 0.0


class BiometricClient:
    """
    Client for accessing biometric data.

    Supports:
    - HealthKit (macOS)
    - Fitbit API
    - Simulated data (for testing)
    """

    def __init__(self, source: str = "simulated"):
        """
        Initialize biometric client.

        Args:
            source: "healthkit", "fitbit", or "simulated"
        """
        self.source = source
        self.enabled = False
        self.callback: Optional[Callable[[BiometricReading], None]] = None
        self.baseline: Optional[BiometricReading] = None
        self.history: list = []

    def enable(self):
        """Enable biometric monitoring."""
        self.enabled = True
        print(f"Biometric client enabled (source: {self.source})")

    def disable(self):
        """Disable biometric monitoring."""
        self.enabled = False

    def set_callback(self, callback: Callable[[BiometricReading], None]):
        """Set callback for when new data arrives."""
        self.callback = callback

    def get_latest_reading(self) -> BiometricReading:
        """Get latest biometric reading."""
        if self.source == "simulated":
            # Generate simulated data
            import random
            return BiometricReading(
                heart_rate=70.0 + random.uniform(-10, 20),
                heart_rate_variability=50.0 + random.uniform(-10, 10),
                skin_conductance=5.0 + random.uniform(-1, 2),
                temperature=36.5 + random.uniform(-0.5, 0.5),
                timestamp=time.time()
            )
        elif self.source == "healthkit":
            # Would interface with HealthKit
            print("HealthKit integration not yet implemented")
            return BiometricReading()
        elif self.source == "fitbit":
            # Would interface with Fitbit API
            print("Fitbit integration not yet implemented")
            return BiometricReading()
        else:
            return BiometricReading()

    def establish_baseline(self, days: int = 7) -> BiometricReading:
        """Establish baseline from historical data."""
        if len(self.history) == 0:
            # Default baseline
            self.baseline = BiometricReading(
                heart_rate=70.0,
                heart_rate_variability=50.0,
                skin_conductance=5.0,
                temperature=36.5
            )
            return self.baseline

        # Calculate averages from history
        valid_readings = [r for r in self.history if r.is_valid()]

        if len(valid_readings) == 0:
            return self.baseline or BiometricReading()

        # Use recent readings (last N days)
        cutoff_time = time.time() - (days * 24 * 3600)
        recent = [r for r in valid_readings if r.timestamp >= cutoff_time]

        if len(recent) == 0:
            recent = valid_readings[-100:]  # Use last 100 readings

        avg_hr = sum(r.heart_rate for r in recent if r.heart_rate) / len([r for r in recent if r.heart_rate])
        avg_hrv = sum(r.heart_rate_variability for r in recent if r.heart_rate_variability) / len([r for r in recent if r.heart_rate_variability])
        avg_sc = sum(r.skin_conductance for r in recent if r.skin_conductance) / len([r for r in recent if r.skin_conductance])
        avg_temp = sum(r.temperature for r in recent if r.temperature) / len([r for r in recent if r.temperature])

        self.baseline = BiometricReading(
            heart_rate=avg_hr,
            heart_rate_variability=avg_hrv,
            skin_conductance=avg_sc,
            temperature=avg_temp,
            timestamp=time.time()
        )

        return self.baseline

    def normalize_reading(self, reading: BiometricReading) -> Dict[str, float]:
        """
        Normalize reading relative to baseline.

        Returns normalized values in 0.0-1.0 range.
        """
        if not self.baseline or not self.baseline.is_valid():
            self.establish_baseline()

        normalized = {}

        if reading.heart_rate and self.baseline.heart_rate:
            deviation = reading.heart_rate - self.baseline.heart_rate
            normalized['heart_rate'] = 0.5 + (deviation / 60.0)  # ±30 BPM = ±0.5
            normalized['heart_rate'] = max(0.0, min(1.0, normalized['heart_rate']))

        if reading.heart_rate_variability and self.baseline.heart_rate_variability:
            ratio = reading.heart_rate_variability / self.baseline.heart_rate_variability
            normalized['hrv'] = min(2.0, ratio) / 2.0  # 0-2x baseline -> 0-1.0

        if reading.skin_conductance and self.baseline.skin_conductance:
            ratio = reading.skin_conductance / self.baseline.skin_conductance
            normalized['skin_conductance'] = min(2.0, ratio) / 2.0

        return normalized

    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start monitoring (simulated - would use actual sensor in production)."""
        if not self.enabled:
            return

        print(f"Starting biometric monitoring (interval: {interval_seconds}s)")

        # In production, this would set up actual sensor callbacks
        # For now, simulate periodic readings
        while self.enabled:
            reading = self.get_latest_reading()
            self.history.append(reading)

            # Keep history limited
            if len(self.history) > 10000:
                self.history = self.history[-5000:]

            if self.callback:
                self.callback(reading)

            time.sleep(interval_seconds)


def main():
    """Example usage."""
    client = BiometricClient(source="simulated")
    client.enable()

    def on_reading(reading: BiometricReading):
        print(f"Heart Rate: {reading.heart_rate:.1f} BPM, "
              f"HRV: {reading.heart_rate_variability:.1f} ms")

    client.set_callback(on_reading)

    # Establish baseline
    baseline = client.establish_baseline()
    print(f"Baseline: HR={baseline.heart_rate:.1f}, HRV={baseline.heart_rate_variability:.1f}")

    # Get normalized reading
    reading = client.get_latest_reading()
    normalized = client.normalize_reading(reading)
    print(f"Normalized: {normalized}")


if __name__ == "__main__":
    main()
