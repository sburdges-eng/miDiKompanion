#!/usr/bin/env python3
"""
Advanced Training Utilities for Kelly MIDI Companion ML Training
================================================================
Provides advanced training features:
- Learning rate scheduling
- Gradient clipping
- Mixed precision training
- Data augmentation
- Training stability features
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    OneCycleLR
)
from typing import Optional, Dict, Callable
from pathlib import Path
import numpy as np


class LearningRateScheduler:
    """Wrapper for learning rate schedulers with logging."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'plateau',
        **kwargs
    ):
        """
        Initialize learning rate scheduler.

        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('plateau', 'cosine', 'step', 'exp', 'onecycle')
            **kwargs: Additional arguments for scheduler
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.lr_history = [self.initial_lr]

        if scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 5),
                verbose=kwargs.get('verbose', True),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'exp':
            self.scheduler = ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', self.initial_lr * 10),
                epochs=kwargs.get('epochs', 50),
                steps_per_epoch=kwargs.get('steps_per_epoch', 100)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def step(self, metrics: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler_type == 'plateau':
            if metrics is None:
                raise ValueError("Plateau scheduler requires metrics")
            self.scheduler.step(metrics)
        elif self.scheduler_type == 'onecycle':
            self.scheduler.step()
        else:
            self.scheduler.step()

        # Track learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(current_lr)

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def state_dict(self):
        """Get scheduler state dict."""
        return {
            'scheduler': self.scheduler.state_dict(),
            'lr_history': self.lr_history,
            'scheduler_type': self.scheduler_type
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state dict."""
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.lr_history = state_dict.get('lr_history', [self.initial_lr])


class GradientClipper:
    """Gradient clipping utility."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        """
        Initialize gradient clipper.

        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2.0 for L2, float('inf') for L-inf)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.clip_history = []

    def clip(self, model: nn.Module) -> float:
        """
        Clip gradients of model parameters.

        Returns:
            Gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )
        self.clip_history.append(total_norm.item())
        return total_norm.item()

    def get_stats(self) -> Dict:
        """Get clipping statistics."""
        if not self.clip_history:
            return {}
        return {
            'mean_norm': np.mean(self.clip_history),
            'max_norm': np.max(self.clip_history),
            'min_norm': np.min(self.clip_history),
            'clips': len(self.clip_history)
        }


class MixedPrecisionTrainer:
    """Mixed precision training wrapper."""

    def __init__(self, enabled: bool = True, device: str = 'cuda'):
        """
        Initialize mixed precision trainer.

        Args:
            enabled: Whether to use mixed precision
            device: Device type ('cuda' for GPU, 'cpu' for CPU)
        """
        self.enabled = enabled and device == 'cuda' and torch.cuda.is_available()
        self.scaler = GradScaler() if self.enabled else None

    def autocast(self):
        """Get autocast context manager."""
        if self.enabled:
            return autocast()
        return torch.enable_grad()  # No-op context manager

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with scaling."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def state_dict(self):
        """Get scaler state dict."""
        if self.scaler:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict):
        """Load scaler state dict."""
        if self.scaler and state_dict:
            self.scaler.load_state_dict(state_dict)


class DataAugmentation:
    """Data augmentation utilities for audio and MIDI."""

    @staticmethod
    def augment_audio_features(
        mel_features: torch.Tensor,
        noise_level: float = 0.01,
        time_mask: bool = True,
        freq_mask: bool = True
    ) -> torch.Tensor:
        """
        Augment mel-spectrogram features.

        Args:
            mel_features: Mel-spectrogram tensor (128-dim)
            noise_level: Amount of noise to add
            time_mask: Whether to apply time masking
            freq_mask: Whether to apply frequency masking
        """
        features = mel_features.clone()

        # Add noise
        if noise_level > 0:
            noise = torch.randn_like(features) * noise_level
            features = features + noise

        # Frequency masking (simplified for 1D features)
        if freq_mask and len(features.shape) == 1:
            mask_size = int(len(features) * 0.1)  # Mask 10% of frequencies
            if mask_size > 0:
                start = np.random.randint(0, len(features) - mask_size)
                features[start:start + mask_size] = 0

        # Normalize
        features = (features - features.mean()) / (features.std() + 1e-8)

        return features

    @staticmethod
    def augment_emotion_embedding(
        emotion: torch.Tensor,
        noise_level: float = 0.05
    ) -> torch.Tensor:
        """
        Augment emotion embedding with small noise.

        Args:
            emotion: Emotion embedding tensor (64-dim)
            noise_level: Amount of noise to add
        """
        noise = torch.randn_like(emotion) * noise_level
        augmented = emotion + noise
        # Clip to valid range
        augmented = torch.clamp(augmented, -1.0, 1.0)
        return augmented

    @staticmethod
    def augment_note_probabilities(
        notes: torch.Tensor,
        dropout_prob: float = 0.1,
        noise_level: float = 0.05
    ) -> torch.Tensor:
        """
        Augment MIDI note probabilities.

        Args:
            notes: Note probability tensor (128-dim)
            dropout_prob: Probability of dropping notes
            noise_level: Amount of noise to add
        """
        augmented = notes.clone()

        # Random dropout
        if dropout_prob > 0:
            mask = torch.rand_like(augmented) > dropout_prob
            augmented = augmented * mask

        # Add noise
        if noise_level > 0:
            noise = torch.randn_like(augmented) * noise_level
            augmented = augmented + noise

        # Renormalize
        augmented = torch.clamp(augmented, 0.0, 1.0)
        augmented = augmented / (augmented.sum() + 1e-8)

        return augmented


class TrainingStability:
    """Training stability utilities."""

    @staticmethod
    def check_gradients(model: nn.Module, verbose: bool = False) -> Dict:
        """
        Check gradient statistics.

        Returns:
            Dictionary with gradient statistics
        """
        stats = {
            'total_params': 0,
            'params_with_grad': 0,
            'zero_grads': 0,
            'nan_grads': 0,
            'inf_grads': 0,
            'max_grad': 0.0,
            'mean_grad': 0.0
        }

        total_grad_norm = 0.0
        grad_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                stats['params_with_grad'] += 1
                grad = param.grad.data

                # Check for NaN/Inf
                if torch.isnan(grad).any():
                    stats['nan_grads'] += 1
                    if verbose:
                        print(f"NaN gradient in {name}")
                if torch.isinf(grad).any():
                    stats['inf_grads'] += 1
                    if verbose:
                        print(f"Inf gradient in {name}")

                # Check for zero gradients
                if (grad == 0).all():
                    stats['zero_grads'] += 1

                # Statistics
                grad_norm = grad.norm().item()
                total_grad_norm += grad_norm
                grad_count += 1
                stats['max_grad'] = max(stats['max_grad'], grad_norm)

            stats['total_params'] += param.numel()

        if grad_count > 0:
            stats['mean_grad'] = total_grad_norm / grad_count

        return stats

    @staticmethod
    def check_weights(model: nn.Module, verbose: bool = False) -> Dict:
        """
        Check weight statistics.

        Returns:
            Dictionary with weight statistics
        """
        stats = {
            'total_params': 0,
            'nan_weights': 0,
            'inf_weights': 0,
            'max_weight': 0.0,
            'min_weight': 0.0,
            'mean_weight': 0.0
        }

        total_weight = 0.0
        weight_count = 0

        for name, param in model.named_parameters():
            weight = param.data

            # Check for NaN/Inf
            if torch.isnan(weight).any():
                stats['nan_weights'] += 1
                if verbose:
                    print(f"NaN weight in {name}")
            if torch.isinf(weight).any():
                stats['inf_weights'] += 1
                if verbose:
                    print(f"Inf weight in {name}")

            # Statistics
            weight_norm = weight.norm().item()
            total_weight += weight_norm
            weight_count += 1
            stats['max_weight'] = max(stats['max_weight'], weight_norm)
            stats['min_weight'] = min(stats['min_weight'], weight_norm) if weight_count == 1 else min(stats['min_weight'], weight_norm)

            stats['total_params'] += param.numel()

        if weight_count > 0:
            stats['mean_weight'] = total_weight / weight_count

        return stats


def create_advanced_training_config(
    learning_rate: float = 0.001,
    scheduler_type: str = 'plateau',
    gradient_clip: Optional[float] = 1.0,
    mixed_precision: bool = False,
    device: str = 'cpu',
    **kwargs
) -> Dict:
    """
    Create advanced training configuration.

    Returns:
        Dictionary with training configuration
    """
    config = {
        'learning_rate': learning_rate,
        'scheduler': {
            'type': scheduler_type,
            'enabled': scheduler_type is not None,
            **kwargs.get('scheduler_kwargs', {})
        },
        'gradient_clipping': {
            'enabled': gradient_clip is not None,
            'max_norm': gradient_clip
        },
        'mixed_precision': {
            'enabled': mixed_precision and device == 'cuda',
        },
        'data_augmentation': {
            'enabled': kwargs.get('augment', False),
            'noise_level': kwargs.get('augment_noise', 0.01)
        }
    }
    return config
