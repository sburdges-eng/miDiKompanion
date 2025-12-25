# miDiKompanion ONNX Models

This directory contains the trained ONNX models for the ML pipeline.

## Model Files

| Model | File | Input | Output | Purpose |
|-------|------|-------|--------|---------|
| EmotionRecognizer | `emotionrecognizer.onnx` | 128-dim | 64-dim | Audio → Emotion embedding |
| MelodyTransformer | `melodytransformer.onnx` | 64-dim | 128-dim | Emotion → MIDI probabilities |
| HarmonyPredictor | `harmonypredictor.onnx` | 128-dim | 64-dim | Context → Chord probabilities |
| DynamicsEngine | `dynamicsengine.onnx` | 32-dim | 16-dim | Intensity → Expression params |
| GroovePredictor | `groovepredictor.onnx` | 64-dim | 32-dim | Arousal → Groove params |

## Training Models

Models can be trained using:

### Google Colab (Free GPU)

1. Upload `ml_training/Train_MidiKompanion_Models.ipynb` to Colab
2. Enable GPU runtime
3. Run all cells
4. Download the resulting `midikompanion_models.zip`
5. Extract to this directory

### Docker

```bash
cd ml_training
docker-compose up training
# Models will be saved to this directory
```

### Manual Training

```bash
cd ml_training
pip install -r requirements.txt
python train_all_models.py --epochs 100 --output_dir ../models/onnx
```

## Validation

Validate models after training:

```bash
cd ml_training
python validate_models.py --models_dir ../models/onnx
```

## Performance Targets

- Total size: <5 MB
- Single inference: <10 ms
- Full pipeline: <50 ms

## Stub Mode

If models are not present, miDiKompanion will use stub mode which returns
reproducible random data based on input hash. This allows development and
testing without trained models.

To enable stub mode in code:

```cpp
processor.setStubMode(true);
```

## License

These models are part of the miDiKompanion project and are subject to
the same license as the main project.
