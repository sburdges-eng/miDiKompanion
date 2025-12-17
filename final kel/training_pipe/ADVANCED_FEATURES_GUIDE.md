# Advanced Training Features Guide

This guide explains how to use the advanced training features for improved model training.

## Overview

The advanced training utilities provide:

1. **Learning Rate Scheduling** - Adaptive learning rate adjustment
2. **Gradient Clipping** - Prevent exploding gradients
3. **Mixed Precision Training** - Faster training on GPU with less memory
4. **Data Augmentation** - Improve generalization
5. **Training Stability Checks** - Monitor training health

## Learning Rate Scheduling

### Available Schedulers

1. **ReduceLROnPlateau** - Reduce LR when validation loss plateaus
2. **CosineAnnealingLR** - Cosine annealing schedule
3. **StepLR** - Step-wise reduction
4. **ExponentialLR** - Exponential decay
5. **OneCycleLR** - One cycle policy (fast training)

### Usage

```python
from advanced_training_utils import LearningRateScheduler

# Initialize scheduler
scheduler = LearningRateScheduler(
    optimizer,
    scheduler_type='plateau',  # or 'cosine', 'step', 'exp', 'onecycle'
    patience=5,  # For plateau
    factor=0.5,  # For plateau/step
    T_max=50,    # For cosine
    gamma=0.95   # For step/exp
)

# In training loop
for epoch in range(epochs):
    # Train...
    train_loss = train_one_epoch(...)
    
    # Validate
    val_loss = validate(...)
    
    # Step scheduler
    if scheduler_type == 'plateau':
        scheduler.step(val_loss)  # Pass metric for plateau
    else:
        scheduler.step()  # No metric needed
    
    # Get current LR
    current_lr = scheduler.get_lr()
    print(f"Learning rate: {current_lr}")
```

### Recommendations

- **Plateau**: Best for most cases, reduces LR when stuck
- **Cosine**: Good for fixed epoch training
- **OneCycle**: Fastest convergence, use for quick experiments

## Gradient Clipping

Prevents exploding gradients by clipping gradient norms.

### Usage

```python
from advanced_training_utils import GradientClipper

# Initialize clipper
grad_clipper = GradientClipper(max_norm=1.0, norm_type=2.0)

# In training loop
for batch in train_loader:
    loss.backward()
    
    # Clip gradients
    grad_norm = grad_clipper.clip(model)
    
    optimizer.step()

# Get statistics
stats = grad_clipper.get_stats()
print(f"Mean gradient norm: {stats['mean_norm']}")
```

### When to Use

- Training with RNNs/LSTMs (prone to exploding gradients)
- Very deep networks
- Unstable training (loss spikes)
- Recommended: `max_norm=1.0` for most cases

## Mixed Precision Training

Uses FP16 for faster training and less memory on GPU.

### Usage

```python
from advanced_training_utils import MixedPrecisionTrainer

# Initialize
mp_trainer = MixedPrecisionTrainer(enabled=True, device='cuda')

# In training loop
for batch in train_loader:
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with mp_trainer.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Backward with scaling
    scaled_loss = mp_trainer.scale_loss(loss)
    scaled_loss.backward()
    
    # Optimizer step with scaling
    mp_trainer.step_optimizer(optimizer)
```

### Benefits

- **2x faster** training on modern GPUs
- **50% less memory** usage
- Automatic loss scaling prevents underflow

### Requirements

- CUDA GPU (not available on CPU)
- PyTorch 1.6+
- Modern GPU (Volta/Turing/Ampere or newer)

## Data Augmentation

Improves generalization by adding variations to training data.

### Audio Features Augmentation

```python
from advanced_training_utils import DataAugmentation

augmenter = DataAugmentation()

# Augment mel-spectrogram
augmented_mel = augmenter.augment_audio_features(
    mel_features,
    noise_level=0.01,
    time_mask=True,
    freq_mask=True
)
```

### Emotion Embedding Augmentation

```python
# Add small noise to emotion embeddings
augmented_emotion = augmenter.augment_emotion_embedding(
    emotion,
    noise_level=0.05
)
```

### MIDI Note Augmentation

```python
# Augment note probabilities
augmented_notes = augmenter.augment_note_probabilities(
    notes,
    dropout_prob=0.1,
    noise_level=0.05
)
```

### When to Use

- Small datasets (helps prevent overfitting)
- Improving generalization
- Regularization alternative
- Use moderate augmentation (don't overdo it)

## Training Stability Checks

Monitor training health to catch issues early.

### Gradient Checking

```python
from advanced_training_utils import TrainingStability

# Check gradients
grad_stats = TrainingStability.check_gradients(model, verbose=True)

print(f"Max gradient: {grad_stats['max_grad']}")
print(f"NaN gradients: {grad_stats['nan_grads']}")
print(f"Inf gradients: {grad_stats['inf_grads']}")
```

### Weight Checking

```python
# Check weights
weight_stats = TrainingStability.check_weights(model, verbose=True)

print(f"Max weight: {weight_stats['max_weight']}")
print(f"NaN weights: {weight_stats['nan_weights']}")
```

### When to Use

- Debugging training issues
- Monitoring training health
- Catching NaN/Inf early
- Recommended: Check every 10 epochs

## Complete Example

Here's a complete example using all features:

```python
from enhanced_training_function import train_model_advanced

# Train with all advanced features
results = train_model_advanced(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    epochs=100,
    learning_rate=0.001,
    device='cuda',
    checkpoint_dir=Path("./checkpoints"),
    model_name="emotion_recognizer",
    # Advanced features
    use_scheduler=True,
    scheduler_type='plateau',
    gradient_clip=1.0,
    use_mixed_precision=True,
    use_augmentation=True,
    early_stop_patience=10,
    check_stability=True,
    verbose=True
)

print(f"Best validation loss: {results['best_val_loss']}")
print(f"Best epoch: {results['best_epoch']}")
```

## Configuration Presets

### Fast Training (Quick Experiments)

```python
config = {
    'use_scheduler': True,
    'scheduler_type': 'onecycle',
    'gradient_clip': None,
    'use_mixed_precision': True,
    'use_augmentation': False,
    'early_stop_patience': 5
}
```

### Stable Training (Production)

```python
config = {
    'use_scheduler': True,
    'scheduler_type': 'plateau',
    'gradient_clip': 1.0,
    'use_mixed_precision': False,  # More stable
    'use_augmentation': True,
    'early_stop_patience': 15,
    'check_stability': True
}
```

### Memory-Constrained Training

```python
config = {
    'use_scheduler': True,
    'scheduler_type': 'cosine',
    'gradient_clip': 0.5,
    'use_mixed_precision': True,  # Saves memory
    'use_augmentation': False,
    'early_stop_patience': 10
}
```

## Best Practices

1. **Start Simple**: Begin without advanced features, add as needed
2. **Monitor Metrics**: Watch for improvements when adding features
3. **Gradient Clipping**: Use when training is unstable
4. **Mixed Precision**: Use on GPU for speed, disable if issues occur
5. **Scheduling**: Plateau scheduler works well for most cases
6. **Augmentation**: Use moderate amounts, too much can hurt
7. **Stability Checks**: Run periodically, not every batch

## Troubleshooting

### Training Diverges

- Reduce learning rate
- Enable gradient clipping (max_norm=1.0)
- Disable mixed precision
- Check for NaN/Inf with stability checks

### Slow Training

- Enable mixed precision (if GPU available)
- Use OneCycle scheduler for faster convergence
- Reduce batch size if memory limited

### Overfitting

- Increase data augmentation
- Use plateau scheduler (helps generalization)
- Increase early stopping patience

### Memory Issues

- Enable mixed precision
- Reduce batch size
- Disable augmentation during training

## Integration with Existing Code

The advanced features are optional and can be added incrementally:

1. **Start with scheduler** - Easiest to add, good benefits
2. **Add gradient clipping** - If training is unstable
3. **Try mixed precision** - If you have GPU and want speed
4. **Add augmentation** - If overfitting or small dataset
5. **Enable stability checks** - For debugging or monitoring

All features work independently and can be mixed and matched.
