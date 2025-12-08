# iDAW Sample Library Sources

Quick reference for downloading samples to your Google Drive (up to 1TB).

## How to Download Samples

### 1. Check Your Storage
```bash
cd ~/Applications/iDAW
python3 sample_downloader.py --check-storage
```

### 2. Download from Direct URLs
```bash
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls https://example.com/sample1.wav https://example.com/sample2.wav
```

### 3. Download from URL File
Create a text file with URLs (one per line), then:
```bash
python3 sample_downloader.py \
  --source freesound \
  --category rhythm_core \
  --urls-file my_urls.txt
```

## Sample Sources

### 1. Freesound (https://freesound.org)
**Best for:** Cinema FX, Organic Textures, Foley
**Size:** Varies (individual downloads)
**API Required:** Yes (free account)

**How to download:**
1. Browse Freesound.org for sounds
2. Copy direct download URLs
3. Use the downloader with `--source freesound --category cinema_fx`

### 2. BBC Sound Effects (http://bbcsfx.acropolis.org.uk)
**Best for:** Cinema FX, Sound Design, Foley
**Size:** 30,000+ clips (16GB+ total)
**API Required:** No

**How to download:**
1. Browse the BBC archive
2. Right-click on WAV links to copy URLs
3. Download in batches to avoid filling local disk

**Popular categories:**
- Nature sounds
- Urban environments
- Household items
- Transportation
- Weather effects

### 3. SampleSwap (https://sampleswap.org)
**Best for:** Rhythm Core, Lo-Fi Dreams, Musical Loops
**Size:** 7.5GB+ of samples

**Categories:**
- Drum Loops
- Bass Loops
- Melodic Loops
- Full Drum Kits

### 4. SampleRadar - MusicRadar (https://www.musicradar.com/news/tech/free-music-samples-royalty-free-loops-hits-and-multis-to-download)
**Best for:** Rhythm Core, Brass & Soul, Lo-Fi Dreams
**Size:** 64,000+ samples

**Sample packs available:**
- Acoustic drums
- Electronic drums
- Bass (electric & synth)
- Guitars
- Piano & keys
- Brass & orchestral

### 5. Looperman (https://www.looperman.com)
**Best for:** Rhythm Core, Velvet Noir, Hip-Hop
**Size:** Community-driven, constantly updated

**Categories:**
- Loops
- Acapellas
- Vocals
- MIDI patterns

### 6. Bedroom Producers Blog (https://bedroomproducersblog.com/free-samples/)
**Best for:** Lo-Fi Dreams, Rhythm Core, Curated Packs
**Size:** Gigabytes of organized sample packs

**Popular downloads:**
- Lo-fi piano
- Bedroom drums
- Tape textures
- Vintage synths

## iDAW Categories

Map downloads to these folders in your Google Drive:

- `velvet_noir` → Velvet Noir (dark, moody, intimate sounds)
- `rhythm_core` → Rhythm Core (drums, percussion, rhythmic elements)
- `cinema_fx` → Cinema FX & Foley (sound effects, real-world sounds)
- `lo_fi_dreams` → Lo-Fi Dreams (degraded, warm, nostalgic)
- `brass_soul` → Brass & Soul (horns, orchestral, organic)
- `organic_textures` → Organic Textures (field recordings, acoustic)

## Download Strategy

### For BBC Sound Effects (30,000+ files):
Download in themed batches to match your iDAW categories:

```bash
# Cinema FX batch
python3 sample_downloader.py \
  --source bbc \
  --category cinema_fx \
  --urls-file bbc_nature_sounds.txt

# Organic Textures batch
python3 sample_downloader.py \
  --source bbc \
  --category organic_textures \
  --urls-file bbc_field_recordings.txt
```

### For SampleRadar Packs:
Download complete packs and organize by category:

```bash
# Drum pack → Rhythm Core
python3 sample_downloader.py \
  --source sampleradar \
  --category rhythm_core \
  --urls https://example.com/drum_pack.zip

# Piano pack → Lo-Fi Dreams
python3 sample_downloader.py \
  --source sampleradar \
  --category lo_fi_dreams \
  --urls https://example.com/lofi_piano.zip
```

## Tips

1. **Storage Management:** Run `--check-storage` regularly to monitor your 1TB limit

2. **Batch Downloads:** Create text files with URLs for bulk downloads
   ```
   # cinema_fx_urls.txt
   https://example.com/sound1.wav
   https://example.com/sound2.wav
   https://example.com/sound3.wav
   ```

3. **Automatic Sync:** Files download to local staging, then auto-sync to Google Drive

4. **Resume Downloads:** The script tracks what's been downloaded to avoid duplicates

5. **Organize as You Go:** Use the correct `--category` flag to keep your library organized

## Finding Direct Download URLs

### Method 1: Browser Dev Tools
1. Open the sample site in Chrome
2. Right-click → Inspect
3. Go to Network tab
4. Click download button
5. Find the .wav/.mp3 file in network requests
6. Copy the URL

### Method 2: Right-Click
1. Right-click on download link
2. Select "Copy Link Address"
3. Paste into your URL file

### Method 3: Bulk Scraping (Advanced)
For sites like BBC, you can write a simple scraper:
```bash
# Example: Get all WAV links from a BBC page
curl "http://bbcsfx.acropolis.org.uk/" | grep -o 'http[s]*://[^"]*\.wav' > bbc_urls.txt
```

## Storage Planning

With 1TB available:
- **BBC Sound Effects (16GB):** ~1.6% of storage
- **SampleRadar (varies):** Individual packs typically 100MB-2GB
- **SampleSwap (7.5GB):** ~0.75% of storage
- **Freesound:** Individual files, negligible storage per download

**Recommended allocation:**
- 100GB: Cinema FX & Foley
- 200GB: Rhythm Core
- 200GB: Lo-Fi Dreams
- 100GB: Velvet Noir
- 200GB: Brass & Soul
- 200GB: Organic Textures

## Next Steps

1. Check storage: `python3 sample_downloader.py --check-storage`
2. List categories: `python3 sample_downloader.py --list-categories`
3. Start with a small test batch to verify everything works
4. Scale up to larger downloads

Happy sampling! 🎵
