# ML API Documentation

Complete API reference for Kelly MIDI Companion ML components.

## C++ API

### MultiModelProcessor

**Location**: `src/ml/MultiModelProcessor.h`, `src/ml/MultiModelProcessor.cpp`

**Namespace**: `Kelly::ML`

#### Initialization

```cpp
#include "ml/MultiModelProcessor.h"

using namespace Kelly::ML;

// Create processor
MultiModelProcessor processor;

// Initialize with models directory
juce::File modelsDir = juce::File::getSpecialLocation(
    juce::File::currentExecutableFile
).getParentDirectory().getChildFile("Resources").getChildFile("models");

bool success = processor.initialize(modelsDir);
```

#### Model Types

```cpp
enum class ModelType : size_t {
    EmotionRecognizer = 0,  // Audio → Emotion
    MelodyTransformer = 1,  // Emotion → MIDI
    HarmonyPredictor  = 2,  // Context → Chords
    DynamicsEngine    = 3,  // Context → Expression
    GroovePredictor   = 4,  // Emotion → Groove
    COUNT             = 5
};
```

#### Single Model Inference

```cpp
// Run single model
std::vector<float> input(128, 0.0f);  // 128-dim audio features
std::vector<float> output = processor.infer(
    ModelType::EmotionRecognizer,
    input
);
// output.size() == 64 (emotion embedding)
```

#### Full Pipeline Inference

```cpp
// Run all models in sequence
std::array<float, 128> audioFeatures{};
// ... fill audioFeatures with mel-spectrogram features ...

InferenceResult result = processor.runFullPipeline(audioFeatures);

// Access results
std::array<float, 64> emotion = result.emotionEmbedding;
std::array<float, 128> melody = result.melodyProbabilities;
std::array<float, 64> harmony = result.harmonyPrediction;
std::array<float, 16> dynamics = result.dynamicsOutput;
std::array<float, 32> groove = result.grooveParameters;
bool valid = result.valid;
```

#### Model Management

```cpp
// Enable/disable specific models
processor.setModelEnabled(ModelType::EmotionRecognizer, true);
bool enabled = processor.isModelEnabled(ModelType::EmotionRecognizer);

// Reload a model after training
juce::File modelFile = modelsDir.getChildFile("emotionrecognizer.json");
processor.reloadModel(ModelType::EmotionRecognizer, modelFile);

// Reload all models
processor.reloadAllModels(modelsDir);

// Get statistics
size_t totalParams = processor.getTotalParams();
size_t memoryKB = processor.getTotalMemoryKB();
bool initialized = processor.isInitialized();
```

#### Async Inference (Audio Thread Safe)

```cpp
#include "ml/MultiModelProcessor.h"

// Create async pipeline
AsyncMLPipeline asyncPipeline(processor);
asyncPipeline.start();

// Submit features (non-blocking, audio thread safe)
std::array<float, 128> features{};
asyncPipeline.submitFeatures(features);

// Check for results (non-blocking)
if (asyncPipeline.hasResult()) {
    InferenceResult result = asyncPipeline.getResult();
    // Use result...
}

// Cleanup
asyncPipeline.stop();
```

### ModelWrapper

**Location**: `src/ml/MultiModelProcessor.h` (private, used internally)

Direct model access (for advanced use):

```cpp
// Access individual model wrapper
// Note: This is internal, use MultiModelProcessor API instead
```

### RTNeuralProcessor

**Location**: `src/ml/RTNeuralProcessor.h`, `src/ml/RTNeuralProcessor.cpp`

**Namespace**: `kelly`

Legacy single-model processor (use MultiModelProcessor for new code):

```cpp
#include "ml/RTNeuralProcessor.h"

kelly::RTNeuralProcessor processor;

// Load model
juce::File modelFile("emotion_model.json");
bool loaded = processor.loadModel(modelFile);

// Infer emotion from features
std::array<float, 128> features{};
std::array<float, 64> emotion = processor.inferEmotion(features);

// Check status
bool isLoaded = processor.isModelLoaded();
std::string modelPath = processor.getModelPath();
```

## Python API

### Training Script

**Location**: `ml_training/train_all_models.py`

#### Command-Line Interface

```bash
python ml_training/train_all_models.py [OPTIONS]
```

**Options**:

- `--config`, `-c`: Path to config.json (default: ml_training/config.json)
- `--output`, `-o`: Output directory for models (default: from config or ./trained_models)
- `--epochs`, `-e`: Number of epochs (default: from config or 50)
- `--batch-size`, `-b`: Batch size (default: from config or 64)
- `--device`, `-d`: Device: cpu, cuda, mps (default: auto-detect)
- `--learning-rate`, `-lr`: Learning rate (default: from config or 0.001)
- `--validation-split`, `-v`: Validation split ratio (default: 0.2)
- `--early-stopping-patience`: Patience for early stopping (default: 10)
- `--datasets-dir`: Directory containing datasets
- `--use-synthetic`: Force synthetic data
- `--no-history`: Don't save training history
- `--no-plots`: Don't generate plots

#### Programmatic API

```python
from pathlib import Path
from train_all_models import train_all_models

# Train all models
train_all_models(
    output_dir=Path("./trained_models"),
    epochs=50,
    batch_size=64,
    device="cuda",  # or "cpu", "mps"
    validation_split=0.2,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    learning_rate=0.001,
    resume_from=None,  # Model name to resume from
    save_history=True,
    plot_curves=True,
    datasets_dir=Path("./datasets"),
    use_real_data=True
)
```

### Model Definitions

**Location**: `ml_training/train_all_models.py`

```python
from train_all_models import (
    EmotionRecognizer,
    MelodyTransformer,
    HarmonyPredictor,
    DynamicsEngine,
    GroovePredictor
)

# Create models
emotion_model = EmotionRecognizer()
melody_model = MelodyTransformer()
harmony_model = HarmonyPredictor()
dynamics_model = DynamicsEngine()
groove_model = GroovePredictor()

# Forward pass
import torch
input_features = torch.randn(1, 128)  # For EmotionRecognizer
emotion_output = emotion_model(input_features)  # Shape: (1, 64)
```

### RTNeural Export

**Location**: `ml_training/train_all_models.py`

```python
from train_all_models import export_to_rtneural
from pathlib import Path

# Export model to RTNeural JSON
model = EmotionRecognizer()
output_dir = Path("./models")
export_to_rtneural(model, "EmotionRecognizer", output_dir)
# Creates: ./models/emotionrecognizer.json
```

### Training Utilities

**Location**: `ml_training/training_utils.py`

#### TrainingMetrics

```python
from training_utils import TrainingMetrics

metrics = TrainingMetrics()

# Update metrics
metrics.update(
    epoch=1,
    train_loss=0.5,
    val_loss=0.4,
    train_metric=0.8,
    val_metric=0.85
)

# Save metrics
metrics.save(Path("metrics.json"))  # Dataclass format
metrics.save_json(Path("metrics_history.json"))  # History dict format
metrics.save_csv(Path("metrics.csv"))

# Plot curves
metrics.plot_curves(Path("./plots"), "ModelName")

# Get best epoch
best_epoch = metrics.get_best_epoch('val_loss', 'min')
```

#### EarlyStopping

```python
from training_utils import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    mode='min',  # 'min' for loss, 'max' for accuracy
    restore_best_weights=True
)

# In training loop
for epoch in range(epochs):
    val_loss = validate(model, val_loader)
    if early_stopping(val_loss, model):
        print("Early stopping triggered")
        break
```

#### CheckpointManager

```python
from training_utils import CheckpointManager

checkpoint_manager = CheckpointManager(
    checkpoint_dir=Path("./checkpoints"),
    max_checkpoints=5
)

# Save checkpoint
checkpoint_manager.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics=metrics,
    model_name="EmotionRecognizer",
    is_best=True
)

# Load checkpoint
model, optimizer, epoch, metrics = checkpoint_manager.load_latest(
    model=model,
    optimizer=optimizer,
    device="cpu"
)

# Load best model
model, optimizer, epoch, metrics = checkpoint_manager.load_best(
    model=model,
    optimizer=optimizer,
    device="cpu"
)
```

#### LearningRateScheduler

```python
from training_utils import LearningRateScheduler

lr_scheduler = LearningRateScheduler(
    optimizer=optimizer,
    mode='reduce_on_plateau',  # or 'step', 'cosine', 'exponential'
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# In training loop
for epoch in range(epochs):
    train(model, train_loader)
    val_loss = validate(model, val_loader)
    lr_scheduler.step(val_loss)  # For reduce_on_plateau
    # or lr_scheduler.step()  # For other modes
```

#### Model Evaluation

```python
from training_utils import evaluate_model, compute_cosine_similarity
import torch.nn as nn

# Evaluate model
criterion = nn.MSELoss()
results = evaluate_model(
    model=model,
    data_loader=val_loader,
    criterion=criterion,
    device="cpu",
    metric_fn=compute_cosine_similarity
)

# results['loss'] - average loss
# results['metric'] - average metric (if metric_fn provided)
```

### Dataset Loaders

**Location**: `ml_training/dataset_loaders.py`

```python
from dataset_loaders import (
    create_dataset,
    DEAMDataset,
    LakhMIDIDataset,
    MAESTRODataset,
    GrooveMIDIDataset,
    HarmonyDataset,
    create_data_loaders
)

# Create dataset
emotion_dataset = create_dataset('deam', Path("./datasets/deam"))

# Or use specific dataset class
emotion_dataset = DEAMDataset(
    deam_dir=Path("./datasets/deam"),
    annotations_file=Path("./datasets/deam/annotations.csv"),
    sample_rate=22050,
    n_mels=128
)

# Create data loaders for all models
loaders = create_data_loaders(
    datasets_dir=Path("./datasets"),
    batch_size=64,
    val_split=0.2,
    use_synthetic=True,
    num_workers=0
)

# Access loaders
emotion_train, emotion_val = loaders['EmotionRecognizer']
melody_train, melody_val = loaders['MelodyTransformer']
# etc.
```

### Model Validation

**Location**: `ml_training/validate_models.py`

```python
from validate_models import validate_model_file
from pathlib import Path

# Validate single model
is_valid, errors, report = validate_model_file(
    Path("./models/emotionrecognizer.json")
)

if is_valid:
    print("Model is valid")
else:
    print("Errors:", errors)
    print("Report:", report)
```

**Command-line**:

```bash
# Validate all models in directory
python ml_training/validate_models.py ./models

# Validate specific model
python ml_training/validate_models.py ./models/emotionrecognizer.json

# Save validation report
python ml_training/validate_models.py ./models --json report.json
```

### Architecture Verification

**Location**: `ml_training/verify_model_architectures.py`

```python
from verify_model_architectures import verify_model
from train_all_models import EmotionRecognizer

# Verify model matches C++ specs
is_valid = verify_model(EmotionRecognizer, "EmotionRecognizer")
```

**Command-line**:

```bash
python ml_training/verify_model_architectures.py
```

## Integration Examples

### Python Training → C++ Loading

```python
# 1. Train model
from train_all_models import train_all_models
train_all_models(output_dir=Path("./models"), epochs=50)

# 2. Validate export
from validate_models import validate_all_models
results = validate_all_models(Path("./models"))

# 3. Models ready for C++ loading
# Copy to: plugin/Resources/models/
```

```cpp
// 4. Load in C++
MultiModelProcessor processor;
juce::File modelsDir = getModelsDirectory();
processor.initialize(modelsDir);

// 5. Use for inference
std::array<float, 128> features{};
InferenceResult result = processor.runFullPipeline(features);
```

### Model Reload After Training

```python
# Python: Train and export new model
train_all_models(output_dir=Path("./models"), epochs=50)
```

```cpp
// C++: Reload updated model
processor.reloadModel(
    ModelType::EmotionRecognizer,
    juce::File("./models/emotionrecognizer.json")
);
```

## Error Handling

### Python

All training functions handle errors gracefully:

- Dataset loading failures → Falls back to synthetic data
- Model export failures → Logs error, continues with other models
- Training failures → Saves checkpoint, allows resume

### C++

Model loading failures use fallback heuristics:

- Missing model file → Uses heuristic-based inference
- Invalid JSON → Logs error, uses fallback
- RTNeural parse failure → Uses fallback

Check logs for specific error messages.

## Performance Considerations

### Inference Latency

- **Target**: <10ms per model
- **Full Pipeline**: <50ms for all 5 models
- **Async**: Use `AsyncMLPipeline` for audio thread safety

### Memory Usage

- **Total**: ~4.6 MB (1,152,280 parameters × 4 bytes)
- **Per Model**: See model specifications in ML_ARCHITECTURE.md
- **Optimization**: Models use float32, no unnecessary allocations

### Training Speed

- **CPU**: ~10-30 minutes per model (depends on dataset size)
- **GPU (CUDA)**: ~2-5 minutes per model
- **GPU (MPS)**: ~3-7 minutes per model

## References

- **C++ Header**: `src/ml/MultiModelProcessor.h`
- **C++ Implementation**: `src/ml/MultiModelProcessor.cpp`
- **Python Training**: `ml_training/train_all_models.py`
- **Training Utils**: `ml_training/training_utils.py`
- **Dataset Loaders**: `ml_training/dataset_loaders.py`
- **Architecture Docs**: `docs/ML_ARCHITECTURE.md`
- **Training Guide**: `docs/ML_TRAINING_GUIDE.md`
