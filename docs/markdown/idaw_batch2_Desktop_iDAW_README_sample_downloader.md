# iDAW Sample Downloader - Quick Start

## 🎯 Smart Local Cache System

The downloader automatically manages your local disk space while keeping samples in Google Drive:

### How It Works

1. **Download** → Samples saved to local staging (`~/.idaw_sample_staging`)
2. **Sync** → Automatically copied to Google Drive (`sburdges@gmail.com/My Drive/iDAW_Samples`)
3. **Smart Cleanup** → Keeps most recent **5GB per source** locally, deletes older files
4. **Cloud Backup** → All samples remain in Google Drive (1TB available)

### Why This Matters

- **Offline Access**: Most recent 5GB from each source available offline
- **Save Disk Space**: Older files auto-deleted from local disk
- **Cloud Security**: Everything backed up to Google Drive automatically
- **Multiple Sources**: 5GB cache *per source* (BBC, Freesound, etc.)

**Example:** If you download from BBC (10GB) and Freesound (8GB):
- **Local disk**: 5GB from BBC + 5GB from Freesound = 10GB total
- **Google Drive**: 10GB from BBC + 8GB from Freesound = 18GB total
- **Saved space**: 8GB freed on local disk

---

## 📥 Basic Usage

### 1. Download with Default Settings (5GB local cache per source)

```bash
cd ~/Applications/iDAW

# Download from URLs
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls https://example.com/sample1.wav https://example.com/sample2.wav

# Download from file
python3 sample_downloader.py \
  --source freesound \
  --category rhythm_core \
  --urls-file my_urls.txt
```

**What happens:**
- Downloads to local staging
- Syncs to Google Drive
- Keeps newest 5GB locally
- Deletes older files from disk

### 2. Custom Local Cache Size

Want to keep 10GB per source instead of 5GB?

```bash
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls-file bbc_nature_sounds.txt \
  --local-cache-gb 10
```

### 3. Keep Everything Locally

Don't want any cleanup? Keep all files:

```bash
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls-file urls.txt \
  --keep-local-files
```

**Warning:** This disables cleanup and may fill your disk!

---

## 📊 Check Storage

```bash
python3 sample_downloader.py --check-storage
```

**Output:**
```
=== Storage Usage ===
Google Drive: 24.50 GB
Local Staging: 15.00 GB
Total Used: 39.50 GB / 1000.00 GB
Available: 960.50 GB
Usage: 4.0%
```

---

## 🗂️ Categories

Map your downloads to iDAW categories:

| Category Key | Folder Name | Best For |
|-------------|------------|----------|
| `cinema_fx` | Cinema FX & Foley | Sound effects, real-world sounds |
| `rhythm_core` | Rhythm Core | Drums, percussion, loops |
| `lo_fi_dreams` | Lo-Fi Dreams | Degraded, warm, nostalgic |
| `velvet_noir` | Velvet Noir | Dark, moody, intimate |
| `brass_soul` | Brass & Soul | Horns, orchestral |
| `organic_textures` | Organic Textures | Field recordings, acoustic |

---

## 🌐 Sample Sources

```bash
python3 sample_downloader.py --list-sources
```

Available sources:
- **bbc** - BBC Sound Effects (30,000+ clips)
- **freesound** - Public domain sound effects
- **sampleswap** - Royalty-free loops (7.5GB)
- **sampleradar** - MusicRadar samples (64,000+)
- **looperman** - Community loops & vocals
- **bedroom_producers** - Curated sample packs

---

## 💡 Real-World Examples

### Example 1: Download BBC Nature Sounds

```bash
# Create URL file (bbc_nature.txt):
# https://sound-effects.bbcrewind.co.uk/samples/rain.wav
# https://sound-effects.bbcrewind.co.uk/samples/thunder.wav
# https://sound-effects.bbcrewind.co.uk/samples/wind.wav

python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls-file bbc_nature.txt
```

**Result:**
- Downloads 3 files to local staging
- Syncs to Google Drive
- All 3 kept locally (under 5GB limit)
- Available offline for sampling

### Example 2: Large BBC Download (20GB)

```bash
python3 sample_downloader.py \
  --source bbc \
  --category organic_textures \
  --urls-file bbc_all_field_recordings.txt
```

**Result:**
- Downloads 20GB to local staging
- Syncs to Google Drive (20GB)
- Keeps newest 5GB locally
- Deletes 15GB from local disk
- All 20GB still in Google Drive

### Example 3: Multiple Sources

```bash
# Download from BBC
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls-file bbc_urls.txt

# Download from Freesound
python3 sample_downloader.py \
  --source freesound \
  --category cinema_fx \
  --urls-file freesound_urls.txt

# Check storage
python3 sample_downloader.py --check-storage
```

**Result:**
- 5GB from BBC locally
- 5GB from Freesound locally
- 10GB total local disk usage
- Full libraries in Google Drive

---

## 🔧 Advanced Options

### All Options

```bash
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls-file urls.txt \
  --max-storage-gb 1000 \
  --local-cache-gb 5 \
  --keep-local-files
```

**Options:**
- `--max-storage-gb`: Total cloud storage limit (default: 1000)
- `--local-cache-gb`: GB to keep per source (default: 5)
- `--keep-local-files`: Disable cleanup, keep everything

---

## 📁 File Locations

### Local Staging
```
~/.idaw_sample_staging/
├── Cinema FX & Foley/
│   ├── rain.wav
│   └── thunder.wav
├── Rhythm Core/
│   ├── drum_loop_1.wav
│   └── drum_loop_2.wav
└── download_log.json
```

### Google Drive
```
~/sburdges@gmail.com - Google Drive/My Drive/iDAW_Samples/
├── Cinema FX & Foley/
│   ├── rain.wav
│   ├── thunder.wav
│   └── wind.wav (older, deleted locally)
└── Rhythm Core/
    ├── drum_loop_1.wav
    └── drum_loop_2.wav
```

---

## 🚀 Pro Tips

1. **Start Small**: Download a test batch to verify everything works
   ```bash
   python3 sample_downloader.py \
     --source bbc \
     --category cinema_fx \
     --urls https://example.com/test.wav
   ```

2. **Monitor Storage**: Check regularly to track usage
   ```bash
   python3 sample_downloader.py --check-storage
   ```

3. **Increase Cache**: Working on a project? Bump local cache temporarily
   ```bash
   python3 sample_downloader.py \
     --source bbc \
     --category cinema_fx \
     --urls-file project_sounds.txt \
     --local-cache-gb 20
   ```

4. **Batch by Category**: Download similar sounds together
   ```bash
   # All drums to Rhythm Core
   python3 sample_downloader.py \
     --source sampleswap \
     --category rhythm_core \
     --urls-file drums.txt
   ```

5. **Organize URLs**: Create URL files by theme
   ```
   bbc_nature.txt
   bbc_urban.txt
   bbc_household.txt
   freesound_foley.txt
   ```

---

## 🐛 Troubleshooting

### "Operation not permitted" on Google Drive
- Make sure Google Drive app is running
- Check that the folder exists: `ls -la ~/sburdges@gmail.com\ -\ Google\ Drive/My\ Drive/`

### Downloads not syncing
- Check Google Drive is connected
- Verify sufficient Google Drive storage
- Run `--check-storage` to see usage

### Local disk full
- Decrease `--local-cache-gb` to free space
- Or manually clean: `rm -rf ~/.idaw_sample_staging/*` (files still in cloud)

---

## 📈 Storage Planning

With 1TB Google Drive and smart local cache:

| Source | Download Size | Local Cache | Cloud Storage | Local Saved |
|--------|--------------|-------------|---------------|-------------|
| BBC Sound Effects | 16 GB | 5 GB | 16 GB | 11 GB |
| SampleSwap | 7.5 GB | 5 GB | 7.5 GB | 2.5 GB |
| Freesound (custom) | 30 GB | 5 GB | 30 GB | 25 GB |
| **Total** | **53.5 GB** | **15 GB** | **53.5 GB** | **38.5 GB** |

You can download **hundreds of GB** while using minimal local disk space!

---

## ✅ Quick Reference

```bash
# Check storage
python3 sample_downloader.py --check-storage

# List sources
python3 sample_downloader.py --list-sources

# List categories
python3 sample_downloader.py --list-categories

# Download (default: 5GB cache per source)
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls-file urls.txt

# Download (custom cache)
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls-file urls.txt \
  --local-cache-gb 10

# Download (keep everything)
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls-file urls.txt \
  --keep-local-files
```

Happy sampling! 🎵
