#!/usr/bin/env python3
"""
Unit Tests for ML Models
=========================
Tests for the 5 ML models: EmotionRecognizer, MelodyTransformer,
HarmonyPredictor, DynamicsEngine, and GroovePredictor.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add training scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "training_pipe" / "scripts"))

try:
    from train_all_models import (
        EmotionRecognizer,
        MelodyTransformer,
        HarmonyPredictor,
        DynamicsEngine,
        GroovePredictor,
        export_to_rtneural
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    MODELS_AVAILABLE = False
    print(f"Warning: Could not import models: {e}")


class TestEmotionRecognizer(unittest.TestCase):
    """Tests for EmotionRecognizer model."""

    def setUp(self):
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")
        self.model = EmotionRecognizer()
        self.model.eval()

    def test_forward_pass(self):
        """Test forward pass with correct input size."""
        batch_size = 4
        input_size = 128
        x = torch.randn(batch_size, input_size)

        output = self.model(x)

        self.assertEqual(output.shape, (batch_size, 64))
        self.assertTrue(torch.all(output >= -1.0) and torch.all(output <= 1.0))

    def test_output_range(self):
        """Test that output is in expected range [-1, 1]."""
        x = torch.randn(1, 128)
        output = self.model(x)

        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_batch_processing(self):
        """Test processing multiple samples."""
        batch_sizes = [1, 4, 16, 32]
        for bs in batch_sizes:
            x = torch.randn(bs, 128)
            output = self.model(x)
            self.assertEqual(output.shape[0], bs)


class TestMelodyTransformer(unittest.TestCase):
    """Tests for MelodyTransformer model."""

    def setUp(self):
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")
        self.model = MelodyTransformer()
        self.model.eval()

    def test_forward_pass(self):
        """Test forward pass with emotion embedding."""
        batch_size = 4
        input_size = 64
        x = torch.randn(batch_size, input_size)

        output = self.model(x)

        self.assertEqual(output.shape, (batch_size, 128))
        self.assertTrue(torch.all(output >= 0.0) and torch.all(output <= 1.0))

    def test_output_probabilities(self):
        """Test that output can be interpreted as probabilities."""
        x = torch.randn(1, 64)
        output = self.model(x)

        # Should be in [0, 1] range (sigmoid output)
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))


class TestHarmonyPredictor(unittest.TestCase):
    """Tests for HarmonyPredictor model."""

    def setUp(self):
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")
        self.model = HarmonyPredictor()
        self.model.eval()

    def test_forward_pass(self):
        """Test forward pass with context."""
        batch_size = 4
        input_size = 128
        x = torch.randn(batch_size, input_size)

        output = self.model(x)

        self.assertEqual(output.shape, (batch_size, 64))
        # Should sum to ~1.0 (softmax probabilities)
        sums = output.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones(batch_size), atol=0.01))

    def test_probability_distribution(self):
        """Test that output forms valid probability distribution."""
        x = torch.randn(1, 128)
        output = self.model(x)

        self.assertAlmostEqual(output.sum().item(), 1.0, places=2)
        self.assertTrue(torch.all(output >= 0.0))


class TestDynamicsEngine(unittest.TestCase):
    """Tests for DynamicsEngine model."""

    def setUp(self):
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")
        self.model = DynamicsEngine()
        self.model.eval()

    def test_forward_pass(self):
        """Test forward pass with compact context."""
        batch_size = 4
        input_size = 32
        x = torch.randn(batch_size, input_size)

        output = self.model(x)

        self.assertEqual(output.shape, (batch_size, 16))
        self.assertTrue(torch.all(output >= 0.0) and torch.all(output <= 1.0))

    def test_output_range(self):
        """Test output is in [0, 1] range."""
        x = torch.randn(1, 32)
        output = self.model(x)

        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))


class TestGroovePredictor(unittest.TestCase):
    """Tests for GroovePredictor model."""

    def setUp(self):
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")
        self.model = GroovePredictor()
        self.model.eval()

    def test_forward_pass(self):
        """Test forward pass with emotion embedding."""
        batch_size = 4
        input_size = 64
        x = torch.randn(batch_size, input_size)

        output = self.model(x)

        self.assertEqual(output.shape, (batch_size, 32))
        self.assertTrue(torch.all(output >= -1.0) and torch.all(output <= 1.0))

    def test_output_range(self):
        """Test output is in [-1, 1] range."""
        x = torch.randn(1, 64)
        output = self.model(x)

        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))


class TestModelExport(unittest.TestCase):
    """Tests for RTNeural export functionality."""

    def test_export_emotion_recognizer(self):
        """Test exporting EmotionRecognizer to RTNeural format."""
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")

        model = EmotionRecognizer()
        output_dir = Path("/tmp/test_export")
        output_dir.mkdir(exist_ok=True)

        try:
            json_data = export_to_rtneural(model, "EmotionRecognizer", output_dir)

            self.assertIn("layers", json_data)
            self.assertIn("metadata", json_data)
            self.assertGreater(len(json_data["layers"]), 0)

            # Check first layer
            first_layer = json_data["layers"][0]
            self.assertIn("type", first_layer)
            self.assertIn("in_size", first_layer)
            self.assertIn("out_size", first_layer)

        finally:
            # Cleanup
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)

    def test_export_all_models(self):
        """Test exporting all models."""
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")

        models = {
            'EmotionRecognizer': EmotionRecognizer(),
            'MelodyTransformer': MelodyTransformer(),
            'HarmonyPredictor': HarmonyPredictor(),
            'DynamicsEngine': DynamicsEngine(),
            'GroovePredictor': GroovePredictor()
        }

        output_dir = Path("/tmp/test_export_all")
        output_dir.mkdir(exist_ok=True)

        try:
            for name, model in models.items():
                json_data = export_to_rtneural(model, name, output_dir)
                self.assertIn("layers", json_data)
                self.assertIn("metadata", json_data)
        finally:
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)


class TestModelInference(unittest.TestCase):
    """Tests for model inference performance."""

    def test_inference_latency(self):
        """Test that inference is fast enough (<10ms target)."""
        if not MODELS_AVAILABLE:
            self.skipTest("Models not available")

        import time

        model = EmotionRecognizer()
        model.eval()

        x = torch.randn(1, 128)

        # Warmup
        with torch.no_grad():
            _ = model(x)

        # Measure latency
        times = []
        for _ in range(100):
            start = time.time()
            with torch.no_grad():
                _ = model(x)
            times.append((time.time() - start) * 1000)  # Convert to ms

        avg_latency = np.mean(times)
        max_latency = np.max(times)

        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Max latency: {max_latency:.2f}ms")

        # Should be under 10ms on average
        self.assertLess(avg_latency, 10.0, "Inference too slow")
        self.assertLess(max_latency, 50.0, "Max latency too high")


if __name__ == "__main__":
    unittest.main()
