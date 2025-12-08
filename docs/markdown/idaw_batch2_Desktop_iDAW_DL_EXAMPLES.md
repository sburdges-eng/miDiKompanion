# DL Script - Quick Reference

The `dl` script makes batch downloading super easy with number shortcuts and auto-cycling.

## 🔢 Number Reference

### Sources (1-6)
```
1 = bbc              (BBC Sound Effects - 30K+ clips)
2 = freesound        (Public domain FX)
3 = sampleswap       (Royalty-free loops)
4 = sampleradar      (MusicRadar 64K+ samples)
5 = looperman        (Community loops)
6 = bedroom_producers (Curated packs)
```

### Categories (1-6)
```
1 = cinema_fx        (Cinema FX & Foley)
2 = rhythm_core      (Rhythm Core)
3 = lo_fi_dreams     (Lo-Fi Dreams)
4 = velvet_noir      (Velvet Noir)
5 = brass_soul       (Brass & Soul)
6 = organic_textures (Organic Textures)
```

---

## 📖 Usage Patterns

### Basic: Single Download
```bash
./dl 1 1 urls.txt
# Downloads: bbc → cinema_fx from urls.txt
```

### Multiple Sources, Same Category
```bash
./dl 1,2,3 2 drums1.txt drums2.txt drums3.txt
# Job 1: bbc → rhythm_core (drums1.txt)
# Job 2: freesound → rhythm_core (drums2.txt)
# Job 3: sampleswap → rhythm_core (drums3.txt)
```

### Auto-Cycling Sources & Categories
```bash
./dl 1,2,3 1,2,3 file1.txt file2.txt file3.txt
# Job 1: bbc → cinema_fx (file1.txt)
# Job 2: freesound → rhythm_core (file2.txt)
# Job 3: sampleswap → lo_fi_dreams (file3.txt)
```

### Download from All Sources
```bash
./dl all 2 drums1.txt drums2.txt drums3.txt drums4.txt drums5.txt drums6.txt
# Downloads from all 6 sources to rhythm_core
```

### Download to All Categories
```bash
./dl 1 all nature1.txt nature2.txt nature3.txt nature4.txt nature5.txt nature6.txt
# Downloads BBC sounds to all 6 categories
```

### Download Everything Everywhere
```bash
./dl all all file1.txt file2.txt ... file36.txt
# Cycles through all sources and categories
# 6 sources × 6 categories = needs 36 files for full coverage
```

---

## 💡 Real-World Examples

### Example 1: BBC Nature Sounds to Multiple Categories
```bash
# Create URL files
echo "https://example.com/rain.wav" > nature1.txt
echo "https://example.com/thunder.wav" > nature2.txt
echo "https://example.com/wind.wav" > nature3.txt

# Download to cinema_fx, organic_textures, lo_fi_dreams
./dl 1 1,6,3 nature1.txt nature2.txt nature3.txt
```

**Result:**
- Job 1: bbc → cinema_fx (rain.wav)
- Job 2: bbc → organic_textures (thunder.wav)
- Job 3: bbc → lo_fi_dreams (wind.wav)

### Example 2: Drums from Multiple Sources
```bash
# Organize drum URLs by source
echo "https://bbc.com/kick.wav" > bbc_drums.txt
echo "https://freesound.org/snare.wav" > freesound_drums.txt
echo "https://sampleswap.org/hihat.wav" > sampleswap_drums.txt

# All go to rhythm_core
./dl 1,2,3 2 bbc_drums.txt freesound_drums.txt sampleswap_drums.txt
```

**Result:**
- Job 1: bbc → rhythm_core
- Job 2: freesound → rhythm_core
- Job 3: sampleswap → rhythm_core

### Example 3: Build Complete Library
```bash
# Download from all sources, cycling through categories
./dl all 1,2,3,4,5,6 \
  fx1.txt fx2.txt drums1.txt drums2.txt lofi1.txt lofi2.txt \
  velvet1.txt velvet2.txt brass1.txt brass2.txt organic1.txt organic2.txt

# Creates 12 jobs cycling through:
# - All 6 sources
# - All 6 categories
```

### Example 4: Themed Collection
```bash
# Dark/moody sounds to appropriate categories
./dl 1,2 4,3 dark1.txt dark2.txt dark3.txt dark4.txt

# Job 1: bbc → velvet_noir (dark1.txt)
# Job 2: freesound → lo_fi_dreams (dark2.txt)
# Job 3: bbc → velvet_noir (dark3.txt)  ← cycles back
# Job 4: freesound → lo_fi_dreams (dark4.txt)
```

---

## 🎯 Pro Tips

### 1. **Preview Before Download**
The script shows a plan and asks for confirmation:
```
=========================================
iDAW DOWNLOAD PLAN (3 jobs)
=========================================

Job 1:
  Source:   bbc
  Category: cinema_fx
  URLs:     nature1.txt

Job 2:
  Source:   freesound
  Category: rhythm_core
  URLs:     drums.txt
...

Proceed with download? [y/N]
```

### 2. **Quick Storage Check**
```bash
./dl check
```

### 3. **List Sources/Categories**
```bash
./dl sources
./dl categories
```

### 4. **Single Source, Multiple Categories**
```bash
# Download BBC sounds across all your categories
./dl 1 all bbc_pack1.txt bbc_pack2.txt bbc_pack3.txt bbc_pack4.txt bbc_pack5.txt bbc_pack6.txt
```

### 5. **Organize URL Files by Theme**
```
urls/
├── bbc_nature.txt
├── bbc_urban.txt
├── freesound_foley.txt
├── sampleswap_drums.txt
├── looperman_vocals.txt
└── bedroom_lofi.txt
```

Then batch download:
```bash
./dl 1,1,2,3,5,6 1,6,1,2,4,3 urls/*.txt
```

---

## 📋 Command Cheat Sheet

```bash
# Single download
./dl <source> <category> <file>

# Multiple sources, one category
./dl <s1,s2,s3> <cat> <file1> <file2> <file3>

# One source, multiple categories
./dl <source> <c1,c2,c3> <file1> <file2> <file3>

# Auto-cycle both
./dl <s1,s2> <c1,c2> <f1> <f2> <f3> <f4>

# All sources
./dl all <category> <files...>

# All categories
./dl <source> all <files...>

# Everything
./dl all all <files...>

# Utilities
./dl check       # Storage
./dl sources     # List sources
./dl categories  # List categories
./dl help        # Show help
```

---

## 🔄 How Cycling Works

The script cycles through **both** sources and categories independently:

### Example: 2 Sources, 3 Categories, 5 Files
```bash
./dl 1,2 1,2,3 a.txt b.txt c.txt d.txt e.txt
```

**Jobs Created:**
```
File 1 (a.txt): source[0]=1 (bbc)       category[0]=1 (cinema_fx)
File 2 (b.txt): source[1]=2 (freesound) category[1]=2 (rhythm_core)
File 3 (c.txt): source[0]=1 (bbc)       category[2]=3 (lo_fi_dreams)
File 4 (d.txt): source[1]=2 (freesound) category[0]=1 (cinema_fx) ← cycled
File 5 (e.txt): source[0]=1 (bbc)       category[1]=2 (rhythm_core)
```

Both indices cycle independently!

---

## ⚡ Quick Workflow

### Daily Download Routine
```bash
# 1. Collect URLs in themed files
echo "url1" > bbc_today.txt
echo "url2" >> bbc_today.txt

# 2. Quick download
./dl 1 1 bbc_today.txt

# 3. Check storage
./dl check
```

### Batch Library Building
```bash
# Prepare URL files
ls urls/
# bbc_1.txt bbc_2.txt free_1.txt free_2.txt swap_1.txt swap_2.txt

# One command for all
./dl 1,2,3 1,2 urls/*.txt

# Check results
./dl check
```

---

## 🎵 Smart Features

✅ **Auto 5GB local cache per source** - Most recent files kept offline
✅ **Confirmation prompt** - Preview before downloading
✅ **Progress tracking** - See each job complete
✅ **Error handling** - Failed jobs reported
✅ **Storage summary** - Auto-check after completion

Happy sampling! 🎵
