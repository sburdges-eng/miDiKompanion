# 🎵 START HERE - Auto Emotion Sampler

## What You're About to Download

**FREE samples organized by emotion and instrument:**

- ✅ **6 Base Emotions**: HAPPY, SAD, ANGRY, FEAR, SURPRISE, DISGUST
- ✅ **4 Instruments**: Piano, Guitar, Drums, Vocals
- ✅ **24 Combinations** (6 × 4)
- ✅ **~120 samples** (~600MB total)
- ✅ **Auto-syncs** to Google Drive

---

## 🚀 OPTION 1: Interactive Setup (Recommended)

### Step 1: Get API Key (2 minutes)
1. Visit: **https://freesound.org/apiv2/apply/**
2. Create account (if needed)
3. Create API application:
   - Name: `iDAW Sample Fetcher`
   - Description: `Personal sample library`
4. **Copy your API key**

### Step 2: Run Setup
```bash
cd ~/Applications/iDAW
./setup_and_start.sh
```

### Step 3: Paste Key
- Paste your API key when prompted
- Press Enter
- Type `y` to start downloading

---

## ⚡ OPTION 2: Quick Start (If You Have API Key)

```bash
cd ~/Applications/iDAW
./quick_start.sh YOUR_API_KEY_HERE
```

Example:
```bash
./quick_start.sh abc123def456ghi789jkl012mno345pqr678stu901
```

Downloads start immediately!

---

## 📊 What Happens Next

### Phase 1: Base Emotions (~30-60 minutes)
```
Downloading HAPPY samples...
  ⬇ HAPPY + piano (1/24)
  ⬇ HAPPY + guitar (2/24)
  ⬇ HAPPY + drums (3/24)
  ⬇ HAPPY + vocals (4/24)

Downloading SAD samples...
  ⬇ SAD + piano (5/24)
  ⬇ SAD + guitar (6/24)
  ... etc
```

### Phase 2: Sub-Emotions (Automatic)
```
Downloading CONTENTMENT samples...
Downloading JOY samples...
... etc
```

### Final: Sync to Google Drive
```
✓ Synced 120 files to Google Drive
Location: ~/sburdges@gmail.com - Google Drive/My Drive/iDAW_Samples/Emotion_Instrument_Library/
```

---

## 📁 Where Your Samples Go

### Google Drive (Permanent):
```
iDAW_Samples/Emotion_Instrument_Library/
├── base/
│   ├── HAPPY/
│   │   ├── piano/     (5 files, ~25MB)
│   │   ├── guitar/    (5 files, ~25MB)
│   │   ├── drums/     (5 files, ~25MB)
│   │   └── vocals/    (5 files, ~25MB)
│   ├── SAD/
│   ├── ANGRY/
│   ├── FEAR/
│   ├── SURPRISE/
│   └── DISGUST/
└── sub/
    ├── CONTENTMENT/
    ├── JOY/
    └── ...
```

---

## 🎯 Example Output

```
======================================================================
[BASE] HAPPY + PIANO
Progress: 0.00MB / 25MB
======================================================================

Searching: happy piano
  ⬇ happy_piano_melody.mp3...
    ✓ Downloaded 5.20MB (Total: 5.20MB)
  ⬇ joyful_keys_loop.mp3...
    ✓ Downloaded 4.80MB (Total: 10.00MB)
  ⬇ bright_chords.mp3...
    ✓ Downloaded 5.10MB (Total: 15.10MB)
  ⬇ uplifting_progression.mp3...
    ✓ Downloaded 4.90MB (Total: 20.00MB)
  ⬇ cheerful_piano.mp3...
    ✓ Downloaded 5.00MB (Total: 25.00MB)

✓ Completed HAPPY/piano: 5 files, 25.00MB

======================================================================
[BASE] HAPPY + GUITAR
Progress: 0.00MB / 25MB
======================================================================
...
```

---

## 🎮 Control Commands

### Check Progress:
```bash
./auto_emotion_sampler.py stats
```

Shows:
```
Statistics:
  Total Combinations: 12
  Total Files: 60
  Total Size: 300.00MB
  Google Drive: ~/sburdges@gmail.com.../Emotion_Instrument_Library/
```

### Sync to Google Drive:
```bash
./auto_emotion_sampler.py sync
```

### Resume Downloads:
```bash
./auto_emotion_sampler.py start
```

(Automatically skips completed combinations)

---

## ⏸️ Pause/Resume

### To Pause:
Press `Ctrl+C`

### To Resume:
```bash
./auto_emotion_sampler.py start
```

Progress is automatically saved!

---

## ❓ Troubleshooting

### "API key required" error:
Run setup again:
```bash
./setup_and_start.sh
```

### No samples found:
- Some emotion-instrument combos have few results
- Script automatically tries multiple searches
- Will skip and move to next combination

### Slow downloads:
- Normal! Freesound has rate limits (1 per second)
- 120 files = ~2-3 minutes
- Leave it running in background

---

## 🎵 Next Steps After Download

1. ✅ Browse samples in Google Drive
2. ✅ Import into your DAW (Ableton, Logic, FL Studio, etc.)
3. ✅ Create emotion-based playlists
4. ✅ Build custom instrument racks
5. ✅ Use with fetch_musical tool for scale recommendations

---

## 🔗 Quick Links

- **Get API Key**: https://freesound.org/apiv2/apply/
- **Freesound Home**: https://freesound.org/
- **Documentation**: AUTO_SAMPLER_README.md

---

## 🎯 Ready to Start?

Choose your option:

### OPTION 1: Interactive
```bash
cd ~/Applications/iDAW
./setup_and_start.sh
```

### OPTION 2: Quick Start
```bash
cd ~/Applications/iDAW
./quick_start.sh YOUR_API_KEY
```

Enjoy your organized sample library! 🎹🎸🥁🎤
