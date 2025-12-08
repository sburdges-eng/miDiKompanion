# Lariat Bible - Official Release v1.0

**Complete Restaurant Management System**
**Release Date**: November 18, 2025
**Version**: 1.0.0
**Status**: Production Ready

---

## 🚀 Quick Start

### Option 1: Double-Click Launcher (Easiest)

**Simply double-click:** `START_LARIAT_BIBLE.command`

### Option 2: Terminal

```bash
cd "/Users/seanburdges/Desktop/LARIAT-BIBLE-OFFICIAL-V.01.00"
python3 lariat_bible_app.py
```

---

## ✨ What's New in v1.0

### Official Order Guide Integration
✅ **CORRECT product codes** for Sysco and Shamrock
✅ **156 pre-matched product pairs** (100% confidence)
✅ **Side-by-side vendor comparison** format
✅ **Optimized descriptions** for fuzzy matching algorithm

### Complete Feature Set
- 📖 **53 Recipes** from Lariat Recipe Book.docx
- 🏪 **332 Vendor Products** (175 Shamrock + 157 Sysco)
- 🔗 **156 Matched Pairs** from official guide
- 📋 **Automated Order Generation** from recipes
- 📈 **Purchase Analysis** and insights

---

## 📊 System Overview

### Dashboard
```
┌──────────────────────────────────────────┐
│  LARIAT BIBLE - DASHBOARD                │
├──────────────────────────────────────────┤
│  Recipes: 53                             │
│  Vendor Products: 332                    │
│  Product Matches: 156                    │
│  Orders Generated: 0                     │
└──────────────────────────────────────────┘
```

### 5 Main Tabs:

1. **📊 Dashboard** - System overview and quick stats
2. **📖 Recipes** - Browse 53 recipes with ingredients
3. **🏪 Vendors** - 332 products with price comparison
4. **📋 Orders** - Automated order generation
5. **📈 Analysis** - Purchase history insights

---

## 🎯 Key Features

### 1. Recipe Management
- **53 recipes** from official Lariat Recipe Book
- Search and filter by category
- View ingredients with pricing
- Export to JSON

### 2. Vendor Management
- **175 Shamrock products** with correct codes
- **157 Sysco products** with correct codes
- **156 pre-matched pairs** from official guide
- Price comparison tool
- Product matching across vendors

### 3. Order Generation
- Select multiple recipes
- Set servings multiplier (2x, 3x, etc.)
- Automatic vendor consolidation
- Cost estimation
- Export to JSON

### 4. Purchase Analysis
- Analyze historical purchase data
- Identify top purchased products
- Generate recommendations
- Export reports

---

## 📁 Package Contents

```
LARIAT-BIBLE-OFFICIAL-V.01.00/
├── START_LARIAT_BIBLE.command    ← DOUBLE-CLICK TO LAUNCH
├── lariat_bible_app.py            ← Main application
├── __init__.py
│
├── features/
│   ├── recipe_manager.py          ← Recipe management system
│   └── vendor_manager.py          ← Vendor management system
│
├── data_importers/
│   ├── docx_importer.py           ← Word document recipe parser
│   ├── order_guide_parser.py     ← Official order guide parser
│   └── excel_importer.py          ← Excel data importer
│
├── SMART COSTING 1/
│   └── LARIAT_SMART_COSTING_COMPLETE_1.xlsx
│
└── documentation/
    ├── README_APP.md              ← Complete user guide
    ├── OFFICIAL_ORDER_GUIDE_UPDATE.md
    ├── VENDOR_MANAGEMENT_COMPLETE.md
    ├── DOCX_INTEGRATION_COMPLETE.md
    └── QUICK_START_VENDOR_SYSTEM.md
```

---

## 💾 Required Data Files

### Must Be Available on Your System:

1. **Lariat Recipe Book**
   - Location: `/Users/seanburdges/Desktop/LARIAT/Lariat Recipe Book.docx`
   - Contains: 53 recipes
   - Format: Microsoft Word document

2. **Official Order Guide**
   - Location: `/Users/seanburdges/Desktop/LARIAT ORDER GUIDE OFFICIAL 8-28-25 (1).xlsx`
   - Contains: 332 products with correct codes
   - Format: Excel with side-by-side vendor comparison

3. **Smart Costing Data**
   - Included: `SMART COSTING 1/LARIAT_SMART_COSTING_COMPLETE_1.xlsx`
   - Contains: 139 ingredients with pricing

---

## ⚙️ System Requirements

### Required:
- **Python 3.7+** (Python 3.13 tested and working)
- **macOS** (tested on macOS with Apple Silicon)

### Python Libraries:
```bash
pip3 install openpyxl python-docx
```

### Included Libraries (standard):
- tkinter (GUI)
- pathlib, json, re, logging
- difflib (fuzzy matching)

---

## 🎨 How to Use

### View Recipes:
1. Click **"📖 Recipes"** tab
2. Browse recipe list or search
3. Select a recipe to view details
4. See ingredients with pricing

### Compare Vendor Prices:
1. Click **"🏪 Vendors"** tab
2. Click **"Compare Prices"** button
3. View 156 pre-matched product pairs
4. See which vendor is cheaper

### Generate Orders:
1. Click **"📋 Orders"** tab
2. Select recipes (Cmd/Ctrl for multiple)
3. Set servings multiplier (e.g., 2.0 for double)
4. Click **"Generate Order"**
5. Review consolidated order
6. Export to JSON

### Analyze Purchases:
1. Click **"📈 Analysis"** tab
2. Click **"Run Analysis"**
3. View top purchased products
4. Get recommendations
5. Export report

---

## 🔧 Configuration

### Change Order Guide:
- **File** → **Load Order Guide**
- Select your Excel order guide
- Auto-detects format (official vs standard)

### Export Options:
- **Orders**: Export to JSON
- **Analysis**: Export to TXT
- **Recipes**: Export to JSON

---

## 📊 What's Loaded on Startup

```
INFO: Imported 53 recipes from Lariat Recipe Book.docx
INFO: Imported 139 ingredients from database
INFO: Parsed 175 Shamrock products
INFO: Parsed 157 Sysco products
INFO: Found 156 pre-matched product pairs

✓ Loaded 53 recipes
✓ Loaded 332 vendor products (official format)
✓ Loaded 156 pre-matched pairs (official guide)
✓ All data loaded successfully!
```

---

## 🐛 Troubleshooting

### App Won't Launch:

**Check Python version:**
```bash
python3 --version  # Should be 3.7 or higher
```

**Install dependencies:**
```bash
pip3 install openpyxl python-docx
```

**Run manually:**
```bash
cd "/Users/seanburdges/Desktop/LARIAT-BIBLE-OFFICIAL-V.01.00"
python3 lariat_bible_app.py
```

### Data Not Loading:

**Check file paths exist:**
```bash
# Recipe file
ls "/Users/seanburdges/Desktop/LARIAT/Lariat Recipe Book.docx"

# Order guide
ls "/Users/seanburdges/Desktop/LARIAT ORDER GUIDE OFFICIAL 8-28-25 (1).xlsx"
```

**If files moved**, update paths in `lariat_bible_app.py` line 33-34

### Order Generation Issues:
- Ensure recipes are selected (check boxes)
- Verify servings multiplier is valid number
- Check that vendor data is loaded

---

## 📋 Menu Bar Functions

### File Menu:
- **Load Order Guide** - Import new vendor data
- **Export Data** - Save current data
- **Exit** - Close application

### Tools Menu:
- **Generate Price Report** - Create price comparison report
- **Run Purchase Analysis** - Analyze buying patterns
- **Refresh Data** - Reload all data

### Help Menu:
- **About** - Application info
- **Quick Start** - Usage guide

---

## 📈 Official Order Guide Format

### What Makes It "Official":

1. **Correct Product Codes**
   - Shamrock: 0039261, 4467141, 4184981, etc.
   - Sysco: 1356254, 4689212, 6395974, etc.

2. **Pre-Matched Pairs**
   - 156 products manually matched
   - 100% confidence (no fuzzy matching)
   - Ready for ordering

3. **Side-by-Side Format**
   - Column 4: Shamrock Product #
   - Column 6: Shamrock Description
   - Column 12: Sysco Product #
   - Column 14: Sysco Description

4. **Optimized Descriptions**
   - Full product descriptions
   - Assists fuzzy matching algorithm
   - Better product identification

---

## 🎓 Documentation

### Quick Reference:
- **README_APP.md** - Complete user guide with screenshots
- **QUICK_START_VENDOR_SYSTEM.md** - 5-minute quick start

### Technical Documentation:
- **VENDOR_MANAGEMENT_COMPLETE.md** - System architecture
- **OFFICIAL_ORDER_GUIDE_UPDATE.md** - Official format details
- **DOCX_INTEGRATION_COMPLETE.md** - Recipe system

---

## ✅ Production Status

**Application**: ✅ Working
**Data Loaded**: ✅ Complete
**All Features**: ✅ Functional
**Production Ready**: ✅ Yes

### Tested and Working:
- ✅ Recipe import from Word document
- ✅ Vendor product parsing (official format)
- ✅ Product matching (156 pairs)
- ✅ Order generation from recipes
- ✅ Purchase analysis
- ✅ Price comparison
- ✅ Data export (JSON, TXT)

---

## 📊 Statistics

| Component | Count | Status |
|-----------|-------|--------|
| Recipes | 53 | ✅ |
| Vendor Products | 332 | ✅ |
| Pre-Matched Pairs | 156 | ✅ |
| Shamrock Products | 175 | ✅ |
| Sysco Products | 157 | ✅ |
| Ingredients with Pricing | 139 | ✅ |

---

## 🚀 Next Steps

### Immediate Use:
1. Double-click `START_LARIAT_BIBLE.command`
2. Explore the 5 tabs
3. Generate your first order
4. Run purchase analysis

### Future Enhancements:
- Add price data to order guide
- Enable real-time price comparison
- Export orders directly to vendors
- Add inventory tracking
- Multi-location support

---

## 🆘 Support

### For Issues:
- Check **documentation/** folder for guides
- Review **OFFICIAL_ORDER_GUIDE_UPDATE.md** for format details
- Check **Troubleshooting** section above

### For Questions:
- See **README_APP.md** for complete user guide
- See **QUICK_START_VENDOR_SYSTEM.md** for quick reference

---

## 📝 Version History

### v1.0.0 (November 18, 2025)
- ✅ Official order guide integration
- ✅ Correct product codes (Sysco & Shamrock)
- ✅ 156 pre-matched product pairs
- ✅ 53 recipes from official recipe book
- ✅ Complete vendor management system
- ✅ Automated order generation
- ✅ Purchase analysis tools
- ✅ Full GUI application

---

## 🎉 Ready to Use!

**Your Lariat Bible application is ready for production use!**

Double-click **START_LARIAT_BIBLE.command** to launch the app and start managing your restaurant operations.

---

**Version**: 1.0.0
**Release**: Official
**Date**: November 18, 2025
**Status**: ✅ Production Ready
