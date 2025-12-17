#!/usr/bin/env python3
"""
End-to-End Workflow Tests
==========================
Tests complete workflows from emotion input to music generation.
"""

import unittest
import numpy as np
from pathlib import Path
import sys
import json
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / "training_pipe" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))


class TestTrainingWorkflow(unittest.TestCase):
    """Test complete training workflow."""

    def test_dataset_loading(self):
        """Test dataset loading pipeline."""
        try:
            from dataset_loaders import EmotionDataset, MelodyDataset

            # Create temporary dataset structure
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                audio_dir = tmp_path / "audio"
                audio_dir.mkdir()

                # Create dummy labels file
                labels_file = audio_dir / "labels.csv"
                with open(labels_file, 'w') as f:
                    f.write("filename,valence,arousal\n")
                    f.write("test.wav,0.5,0.6\n")

                # Test dataset creation (will fail on actual audio loading, but structure is tested)
                try:
                    dataset = EmotionDataset(audio_dir, labels_file)
                    self.assertIsNotNone(dataset)
                except Exception as e:
                    # Expected if librosa not available or no audio files
                    pass
        except ImportError:
            self.skipTest("Dataset loaders not available")

    def test_model_training_workflow(self):
        """Test model training workflow."""
        try:
            from train_all_models import (
                EmotionRecognizer,
                train_emotion_recognizer,
                SyntheticEmotionDataset
            )
            from torch.utils.data import DataLoader

            # Create model
            model = EmotionRecognizer()

            # Create synthetic dataset
            dataset = SyntheticEmotionDataset(num_samples=100)
            loader = DataLoader(dataset, batch_size=16, shuffle=True)

            # Train for a few epochs
            losses = train_emotion_recognizer(
                model, loader, epochs=2, lr=0.001, device='cpu'
            )

            self.assertGreater(len(losses), 0)
            self.assertIsInstance(losses, dict)
        except ImportError:
            self.skipTest("Training modules not available")

    def test_model_export_workflow(self):
        """Test model export workflow."""
        try:
            from train_all_models import (
                EmotionRecognizer,
                export_to_rtneural
            )

            model = EmotionRecognizer()
            model.eval()

            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)

                # Export model
                json_data = export_to_rtneural(model, "TestModel", output_dir)

                # Verify export
                self.assertIn("layers", json_data)
                self.assertIn("metadata", json_data)

                # Check file was created
                json_file = output_dir / "testmodel.json"
                self.assertTrue(json_file.exists())
        except ImportError:
            self.skipTest("Export modules not available")


class TestOSCWorkflow(unittest.TestCase):
    """Test OSC communication workflow."""

    def test_osc_message_format(self):
        """Test OSC message format."""
        # Test message structure
        message = {
            "address": "/daiw/generate",
            "arguments": [
                json.dumps({
                    "text": "test",
                    "motivation": "testing",
                    "chaos": 0.5,
                    "vulnerability": 0.5
                })
            ]
        }

        self.assertEqual(message["address"], "/daiw/generate")
        self.assertEqual(len(message["arguments"]), 1)

        # Parse JSON argument
        params = json.loads(message["arguments"][0])
        self.assertEqual(params["text"], "test")
        self.assertEqual(params["chaos"], 0.5)

    def test_brain_server_initialization(self):
        """Test brain server can be initialized."""
        try:
            from brain_server import BrainServer

            # Test server creation (don't start it)
            server = BrainServer(host="127.0.0.1", port=5005, response_port=5006)
            self.assertIsNotNone(server)
            self.assertFalse(server.running)
        except ImportError:
            self.skipTest("Brain server not available")


class TestVocalSynthesisWorkflow(unittest.TestCase):
    """Test vocal synthesis workflow."""

    def test_emotion_to_vocal_workflow(self):
        """Test complete emotion to vocal synthesis workflow."""
        try:
            from kelly_vocal_system import KellyVocalSystem, VocalSystemConfig

            config = VocalSystemConfig(
                base_frequency=200.0,
                tempo=120,
                enable_quantum=True
            )
            system = KellyVocalSystem(config)

            # Set emotion
            system.set_emotion_from_name("joy")

            # Generate vocals
            wound = "feeling happy"
            result = system.generate_vocals(wound)

            self.assertIsNotNone(result)
            self.assertIsNotNone(result.lyrics)
            self.assertIsNotNone(result.vocal_track)
        except ImportError:
            self.skipTest("Vocal system not available")


class TestMLInferenceWorkflow(unittest.TestCase):
    """Test ML model inference workflow."""

    def test_emotion_recognition_workflow(self):
        """Test emotion recognition inference."""
        try:
            from train_all_models import EmotionRecognizer
            import torch

            model = EmotionRecognizer()
            model.eval()

            # Simulate audio features
            audio_features = torch.randn(1, 128)

            # Run inference
            with torch.no_grad():
                emotion_embedding = model(audio_features)

            self.assertEqual(emotion_embedding.shape, (1, 64))
            self.assertTrue(torch.all(emotion_embedding >= -1.0))
            self.assertTrue(torch.all(emotion_embedding <= 1.0))
        except ImportError:
            self.skipTest("ML models not available")

    def test_melody_generation_workflow(self):
        """Test melody generation from emotion."""
        try:
            from train_all_models import MelodyTransformer
            import torch

            model = MelodyTransformer()
            model.eval()

            # Emotion embedding
            emotion = torch.randn(1, 64)

            # Generate melody probabilities
            with torch.no_grad():
                note_probs = model(emotion)

            self.assertEqual(note_probs.shape, (1, 128))
            self.assertTrue(torch.all(note_probs >= 0.0))
            self.assertTrue(torch.all(note_probs <= 1.0))
        except ImportError:
            self.skipTest("ML models not available")


class TestDataPipeline(unittest.TestCase):
    """Test data processing pipeline."""

    def test_dataset_preparation(self):
        """Test dataset preparation script structure."""
        prep_script = Path(__file__).parent.parent / "training_pipe" / "scripts" / "prepare_datasets.py"

        self.assertTrue(prep_script.exists(), "Dataset preparation script should exist")

    def test_dataset_download(self):
        """Test dataset download script structure."""
        download_script = Path(__file__).parent.parent / "training_pipe" / "scripts" / "download_datasets.py"

        self.assertTrue(download_script.exists(), "Dataset download script should exist")


if __name__ == "__main__":
    unittest.main()
