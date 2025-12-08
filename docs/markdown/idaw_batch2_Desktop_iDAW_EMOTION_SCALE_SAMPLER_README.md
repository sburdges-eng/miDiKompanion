# Emotion-Scale Sample Fetcher

Intelligent sample downloader that uses your **74 emotions × 52 scales** database to organize free .wav samples from Freesound.org.

## 🎯 Features

- **3,848 Possible Combinations**: 74 emotions × 52 musical scales
- **25MB per Combination**: Automatic size management
- **Google Drive Sync**: Organizes into `iDAW_Samples/Emotion_Scale_Library/`
- **Smart Search**: Uses emotion + scale combinations for intelligent queries
- **Progress Tracking**: JSON-based download history
- **Free Samples**: Uses Freesound.org API (100% free)

## 📊 Current Database

- **74 Unique Emotions**: happy, melancholy, jazzy, exotic, tense, peaceful, etc.
- **52 Musical Scales**: Dorian, Phrygian Dominant, Lydian, Blues, Pentatonic, etc.
- **8 Scale Categories**: Major Modes, Harmonic Minor, Melodic Minor, Pentatonic, Blues, Symmetric, Bebop, World/Exotic

## 🚀 Setup (One-time)

### 1. Get Freesound API Key (Free)

1. Go to https://freesound.org/
2. Create a free account (takes 2 minutes)
3. Visit https://freesound.org/apiv2/apply/
4. Create an API application:
   - Name: "iDAW Sample Fetcher"
   - Description: "Personal sample library organizer"
5. Copy your API key

### 2. Configure the Tool

```bash
cd ~/Applications/iDAW
./emotion_scale_sampler.py setup
```

Paste your API key when prompted.

### 3. Install Dependencies (if needed)

```bash
pip3 install requests
```

## 📖 Usage

### List Available Options

```bash
./emotion_scale_sampler.py list
```

Shows all 74 emotions and 52 scales.

### Fetch Specific Combination

```bash
./emotion_scale_sampler.py fetch melancholy dorian
```

Downloads up to 25MB of samples for "melancholy + dorian scale".

### Batch Download (Random Combinations)

```bash
./emotion_scale_sampler.py batch 10
```

Fetches 10 random emotion-scale combinations (5 files each).

### Sync to Google Drive

```bash
./emotion_scale_sampler.py sync
```

Copies all local files to Google Drive.

### Show Statistics

```bash
./emotion_scale_sampler.py stats
```

Displays:
- Total combinations downloaded
- Total files and size
- Top 10 combinations by size

## 📁 File Organization

### Local Staging (Temporary)
```
~/Applications/iDAW/emotion_scale_staging/
├── melancholy/
│   ├── dorian/
│   │   ├── 12345_ambient_loop.wav
│   │   └── 67890_pad_sound.wav
│   └── minor_pentatonic/
│       └── ...
└── happy/
    ├── ionian/
    └── ...
```

### Google Drive (Permanent)
```
~/sburdges@gmail.com - Google Drive/My Drive/iDAW_Samples/Emotion_Scale_Library/
├── melancholy/
│   ├── dorian/
│   └── minor_pentatonic/
└── happy/
    └── ionian/
```

## 🎵 Example Combinations

| Emotion | Scale | Use Case |
|---------|-------|----------|
| melancholy | Dorian | Jazz/fusion melancholic loops |
| happy | Ionian | Bright, uplifting samples |
| exotic | Phrygian Dominant | Spanish/flamenco sounds |
| tense | Altered Scale | Dissonant, anxious textures |
| peaceful | Lydian | Dreamy, floating pads |
| jazzy | Bebop Dominant | Swinging jazz samples |
| dark | Locrian | Unstable, ominous sounds |

## 📊 Size Management

- **Per Combination**: 25MB max (automatically enforced)
- **Total Library**: 74 × 52 × 25MB = ~9.6GB max (if all filled)
- **Smart Caching**: Only downloads new files
- **Progress Tracking**: `emotion_scale_downloads.json` tracks everything

## 🔍 Search Algorithm

The tool creates multiple search queries per combination:

```python
emotion + scale     # "melancholy dorian"
emotion + "music"   # "melancholy music"
emotion + "ambient" # "melancholy ambient"
scale + "scale"     # "dorian scale"
```

Then filters for:
- ✅ .wav files only
- ✅ High-quality previews
- ✅ Freesound-licensed (free to use)

## 💡 Pro Tips

### 1. Start with Popular Combinations
```bash
./emotion_scale_sampler.py fetch melancholy dorian
./emotion_scale_sampler.py fetch happy ionian
./emotion_scale_sampler.py fetch jazzy bebop
```

### 2. Fill Your Library Gradually
```bash
# Download 5 random combos per day
./emotion_scale_sampler.py batch 5
```

### 3. Check Progress Regularly
```bash
./emotion_scale_sampler.py stats
```

### 4. Sync After Each Session
```bash
./emotion_scale_sampler.py sync
```

## 🛠 Advanced Usage

### Custom Search (Edit the Script)

Open `emotion_scale_sampler.py` and modify `create_search_query()`:

```python
def create_search_query(self, emotion, scale):
    queries = [
        f"{emotion} {scale.lower()}",
        f"{emotion} synth",        # Add synth sounds
        f"{emotion} guitar",       # Add guitar samples
        f"{scale.lower()} melody", # Add melodic content
    ]
    return queries
```

### Integration with fetch_musical

The sampler works alongside `fetch_musical`:

```bash
# Find scales for an emotion
./fetch_musical scale emotion melancholy

# Then download samples for those scales
./emotion_scale_sampler.py fetch melancholy dorian
./emotion_scale_sampler.py fetch melancholy aeolian
```

## 📈 Roadmap

- [ ] Multi-threaded downloads (faster)
- [ ] Alternative sources (Archive.org, BBC Sound Effects)
- [ ] BPM detection and tagging
- [ ] Key detection for samples
- [ ] Integration with Streamlit UI
- [ ] Playlist generation by emotion/scale

## ❓ Troubleshooting

### "API key required" Error
Run setup again:
```bash
./emotion_scale_sampler.py setup
```

### No Results Found
Try different emotions/scales:
```bash
./emotion_scale_sampler.py list
```

### Download Fails
- Check internet connection
- Verify API key is valid
- Try again later (rate limiting)

### Google Drive Not Found
Update path in script if your Google Drive is in different location:
```python
GDRIVE_ROOT = Path.home() / "YOUR_EMAIL@gmail.com - Google Drive" / "My Drive"
```

## 📝 License

Uses Freesound.org samples - check individual sample licenses.
Tool code: Use freely for personal projects.

## 🎨 Created By

Claude Code + DAiW Music-Brain emotion taxonomy
