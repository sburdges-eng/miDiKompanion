# Test Model Creation - Setup Instructions

## Current Status

✅ All training scripts are ready
✅ Environment verification script created
⚠️ PyTorch needs to be installed

## Quick Setup (2 minutes)

### Step 1: Install PyTorch

```bash
pip install torch
```

Or if using a virtual environment:

```bash
cd python
source venv/bin/activate  # or: python -m venv venv
pip install torch
```

### Step 2: Verify Installation

```bash
cd training
python3 verify_setup.py
```

You should see all checks passing.

### Step 3: Create Test Model

```bash
python3 create_test_model.py --output test_emotion_model.pt
```

### Step 4: Export to RTNeural Format

```bash
python3 export_to_rtneural.py --model test_emotion_model.pt --output emotion_model.json
```

### Step 5: Deploy to Plugin

```bash
cp emotion_model.json ../../data/emotion_model.json
```

## What You'll Get

After completing these steps, you'll have:

1. **`test_emotion_model.pt`** - PyTorch model file (for reference)
2. **`emotion_model.json`** - RTNeural format (for plugin)
3. Model deployed to `data/emotion_model.json` (plugin will find it)

## Testing in Plugin

1. Build plugin:
   ```bash
   cmake -B build -DENABLE_RTNEURAL=ON
   cmake --build build
   ```

2. Load plugin in DAW

3. Enable ML inference in plugin settings

4. Process audio - model will run!

## Notes

- The test model has random weights (not trained)
- It will run but won't produce accurate results
- Useful for testing plugin integration
- For production, train with real data using `train_emotion_model.py`

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

Install PyTorch:
```bash
pip install torch
```

### "Permission denied" on scripts

Make scripts executable:
```bash
chmod +x *.py
```

### Virtual environment issues

Create fresh venv:
```bash
cd python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps After Test Model

Once the test model works in the plugin:

1. Collect real audio data with emotion labels
2. Extract features using `MLFeatureExtractor`
3. Train production model with `train_emotion_model.py`
4. Export and deploy trained model

See `README.md` for full training instructions.

