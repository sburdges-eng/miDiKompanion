# Lariat Banquet Manager - GUI Application Guide
## Beautiful macOS-Native Interface

> **Version 1.0** | Modern, intuitive design inspired by macOS

---

## 🚀 Quick Start

### Launch the Application

**Method 1: Double-Click (Recommended)**

Simply double-click: **`Lariat Manager.command`**

The application will open in a new window with your custom logo!

**Method 2: Command Line**

```bash
cd "/Users/seanburdges/Desktop/BANQUET BEO"
python3 lariat_gui.py
```

---

## 🎨 Interface Overview

### macOS-Inspired Design

The Lariat Manager features a beautiful, modern interface that feels native to macOS:

- **Clean Sidebar Navigation** - Quick access to all features
- **Card-Based Layout** - Information organized in elegant cards
- **macOS Color Palette** - System blue (#007AFF), green (#34C759), orange (#FF9500)
- **Your Custom Logo** - Displays your photo from the BANQUET BEO folder
- **Smooth Interactions** - Hover effects and visual feedback

### Color System

```
Background:       #F5F5F7 (Light gray - macOS style)
Sidebar:          #E8E8EA (Sidebar gray)
Accent:           #007AFF (macOS Blue)
Success:          #34C759 (macOS Green)
Warning:          #FF9500 (macOS Orange)
Danger:           #FF3B30 (macOS Red)
Cards:            #FFFFFF (Pure white)
```

---

## 📊 Features Guide

### 1. Dashboard View

**Access:** Click "📊 Dashboard" in sidebar (default view)

**What You See:**
- **4 Metric Cards:**
  - Total Events (blue)
  - Total Revenue (green)
  - Average Event Size (orange)
  - This Month Revenue (blue)

- **Recent Activity:**
  - Last 5 events with client name
  - Event date, item count, and total
  - Sorted by most recent

- **Quick Actions:**
  - Import New Invoices button
  - Generate Report button
  - View Database button

**Use Case:** Get an at-a-glance view of your business performance

---

### 2. Import Invoices View

**Access:** Click "📥 Import Invoices" in sidebar

**What You See:**
- **New Invoices Card:**
  - Count of unprocessed invoices
  - List of filenames ready to import
  - Green "Import" button

- **How Import Works Card:**
  - Step-by-step explanation
  - File location reminder
  - Process overview

**How to Use:**
1. Save invoice to `/INVOICE SPREADSHEETS/`
2. Click "Import Invoices" in sidebar
3. Review list of new files
4. Click "Import X Invoice(s)" button
5. Wait for success message
6. Database auto-updates!

**Success Message Shows:**
- Number of events imported
- Total line items added
- Confirmation database updated

---

### 3. Generate Reports View

**Access:** Click "📄 Generate Reports" in sidebar

**What You See:**
- **Monthly Report Generator Card:**
  - Year dropdown (2024-2030)
  - Month dropdown (1-12 with names)
  - Green "Generate Report" button

- **Recent Reports Card:**
  - List of previously generated reports
  - Blue "Open" button for each report
  - Shows last 10 reports

**How to Use:**
1. Select year from dropdown
2. Select month from dropdown
3. Click "Generate Report"
4. Wait for success message
5. Report saved to `/reports/` folder
6. Click "Open" to view in TextEdit

**Report Contains:**
- Revenue summary for the month
- Event count and averages
- Top 5 items ordered
- Event-by-event breakdown
- Category analysis (bar, tacos, etc.)

---

### 4. Database View

**Access:** Click "💾 Database" in sidebar

**What You See:**
- **Database Actions Card:**
  - "Open in Excel" button (blue)
  - "Open Folder" button (gray)

- **Database Structure Card:**
  - List of all 7 sheets
  - Description of each sheet:
    1. Event Summary
    2. All Line Items
    3. Menu Item Popularity
    4. Top Revenue Items
    5. Monthly Summary
    6. Pricing History
    7. Executive Summary

**How to Use:**
- **Open in Excel:** Opens `Lariat_BEO_Database_Analysis.xlsx` in Microsoft Excel
- **Open Folder:** Opens BANQUET BEO folder in Finder to see all files

---

### 5. Settings View

**Access:** Click "⚙️ Settings" in sidebar

**What You See:**
- **File Locations Card:**
  - Invoice directory path
  - Database file path
  - Monospace font for easy reading

- **About Card:**
  - Version information
  - Built for The Lariat
  - Location: Fort Collins, CO
  - Release date

**Use Case:** Verify file paths and system information

---

## 🎯 Common Workflows

### After Each Event

```
1. Save invoice → /INVOICE SPREADSHEETS/
2. Open Lariat Manager (double-click)
3. Click "Import Invoices"
4. Click "Import X Invoice(s)"
5. ✅ Database updated!
6. View updated metrics on Dashboard
```

### End of Month

```
1. Open Lariat Manager
2. Click "Generate Reports"
3. Select year and month
4. Click "Generate Report"
5. Click "Open" to view report
6. Review revenue and top items
```

### Check Business Performance

```
1. Open Lariat Manager
2. Dashboard shows automatically:
   - Total events
   - Total revenue
   - This month's revenue
   - Recent activity
3. Click "View Database" for details
```

---

## 🖼️ Logo Display

### Your Custom Logo

The application displays your photo from:
```
/BANQUET BEO/lariat_logo.png
```

**What Happened:**
1. Original photo: `4D9835AA-F07D-4C41-842E-8D889277A1F3.heic`
2. Converted to: `lariat_logo.png` for compatibility
3. Resized to fit sidebar (max 150x150px)
4. Maintains aspect ratio

**To Change Logo:**
1. Replace `lariat_logo.png` with your new image
2. Restart the application
3. Logo updates automatically!

**Supported Formats:**
- PNG (recommended)
- JPG/JPEG
- GIF

---

## ⌨️ Keyboard Shortcuts

Currently, the application uses mouse/click navigation. Potential shortcuts for future:

- **⌘Q** - Quit application (standard macOS)
- **⌘I** - Import invoices
- **⌘R** - Generate report
- **⌘D** - Dashboard view

---

## 🎨 Design Philosophy

### Why This Design?

**1. Familiar macOS Look**
- Users feel immediately comfortable
- Matches system apps like Settings, Finder
- Professional and polished

**2. Sidebar Navigation**
- Quick access to all features
- Clear visual hierarchy
- Active state highlighting (blue)

**3. Card-Based Content**
- Organized, scannable information
- White cards on gray background
- Consistent spacing and padding

**4. Color-Coded Actions**
- Blue: Primary actions (navigate, open)
- Green: Success/Import actions
- Orange: Warning/Attention
- Gray: Secondary actions

**5. Generous White Space**
- Not cluttered or overwhelming
- Easy to focus on one thing
- Modern, clean aesthetic

---

## 🔧 Technical Details

### Built With

```python
# Core Framework
tkinter - Python's built-in GUI library

# Image Handling
Pillow (PIL) - Logo display and resizing

# Backend
lariat_invoice_automation.py - All business logic
```

### Window Specifications

```
Size:        1200x800 pixels
Sidebar:     220px wide
Font:        SF Pro (macOS system font)
Colors:      macOS standard palette
```

### File Structure

```
lariat_gui.py               # Main GUI application (800+ lines)
Lariat Manager.command      # Double-click launcher
lariat_logo.png            # Your custom logo
```

---

## 🐛 Troubleshooting

### Logo Not Showing

**Issue:** Logo image doesn't appear in sidebar

**Solutions:**
1. Check that `lariat_logo.png` exists in BANQUET BEO folder
2. Verify image format (PNG, JPG, GIF)
3. Check console for error messages
4. Try replacing with a different image

### Window Size Issues

**Issue:** Window too large/small for screen

**Solution:**
Edit `lariat_gui.py` line 25:
```python
self.root.geometry("1200x800")  # Change to "1000x700" or other size
```

### Import Button Not Working

**Issue:** Clicking "Import" does nothing or shows error

**Solutions:**
1. Verify invoice files are in `/INVOICE SPREADSHEETS/`
2. Check file naming format (needs client name and date)
3. Ensure database file isn't open in Excel
4. Check console for error messages

### Database Opens in Wrong App

**Issue:** Database opens in Numbers instead of Excel

**Solution:**
1. Right-click `Lariat_BEO_Database_Analysis.xlsx`
2. Choose "Get Info"
3. Under "Open with:", select Microsoft Excel
4. Click "Change All..."

---

## 📚 View-by-View Breakdown

### Dashboard Metrics Calculation

```python
Total Events:      Count of all rows in Event Summary (excluding totals)
Total Revenue:     Sum of all event totals (column 8)
Avg Event Size:    Total Revenue ÷ Total Events
This Month:        Sum of totals for current month (YYYY-MM match)
```

### Import Process Flow

```
1. Click "Import Invoices"
   ↓
2. Scan /INVOICE SPREADSHEETS/ for new files
   ↓
3. Display count and filenames
   ↓
4. Click "Import X Invoice(s)"
   ↓
5. Extract data from each invoice:
   - Client name
   - Event date
   - Menu items & quantities
   - Pricing & totals
   ↓
6. Update database (all 7 sheets)
   ↓
7. Mark invoices as processed (.processed_invoices.json)
   ↓
8. Show success message
```

### Report Generation Flow

```
1. Select year & month
   ↓
2. Click "Generate Report"
   ↓
3. Query database for matching events
   ↓
4. Calculate statistics:
   - Total revenue
   - Event count
   - Top items
   - Category breakdown
   ↓
5. Generate formatted text file
   ↓
6. Save to /reports/ folder
   ↓
7. Show success message
   ↓
8. Click "Open" to view in TextEdit
```

---

## 🎯 Best Practices

### Daily Use

✅ **DO:**
- Launch app to check dashboard before/after events
- Import invoices within 24 hours of event
- Use Quick Actions for common tasks
- Keep app open during busy periods

❌ **DON'T:**
- Have database open in Excel while importing
- Delete invoices after importing (they're archived)
- Modify database structure manually

### Monthly Review

✅ **DO:**
- Generate monthly report at month-end
- Review top revenue items
- Check category breakdown
- Compare to previous months

### Data Integrity

✅ **DO:**
- Backup database monthly
- Keep original invoice files
- Verify import success messages
- Check recent activity after import

---

## 🔮 Future Enhancements

### Planned Features

**Phase 1 (Current):** ✅ Complete
- Dashboard with metrics
- Invoice import
- Report generation
- Database browser
- Custom logo display

**Phase 2 (Next):**
- [ ] Click to view individual events
- [ ] Search functionality
- [ ] Date range filters
- [ ] Export capabilities

**Phase 3 (Future):**
- [ ] Chart visualizations
- [ ] Client management
- [ ] Menu item editor
- [ ] Inventory tracking

---

## 📞 Support

### Getting Help

1. **Check this guide** - Most features explained
2. **Check console output** - Errors shown in Terminal
3. **Verify file paths** - Settings view shows paths
4. **Test with sample data** - Use existing invoices

### Error Messages

**"No new invoices found"**
→ All files in folder already processed

**"Error importing invoices"**
→ Check file format and database access

**"Could not open database"**
→ Make sure Excel isn't already open

**"Could not load logo"**
→ Verify lariat_logo.png exists

---

## 🏆 Success Metrics

### You'll Know It's Working When:

✅ Dashboard shows correct event count
✅ Import adds new rows to database
✅ Reports generate without errors
✅ Logo displays in sidebar
✅ Database opens in Excel
✅ Recent activity updates after import

---

## 📝 Version History

### v1.0 (2024-11-19)

**Initial Release:**
- ✅ 5-view navigation system
- ✅ Dashboard with 4 key metrics
- ✅ Import invoices interface
- ✅ Generate reports interface
- ✅ Database browser
- ✅ Settings panel
- ✅ Custom logo display
- ✅ macOS-inspired design
- ✅ Real-time data from database
- ✅ Background automation integration

---

**🎉 Enjoy your beautiful new Lariat Banquet Manager!**

For technical details, see: `README.md`
For automation details, see: `AUTOMATION_GUIDE.md`
For quick commands, see: `QUICK_REFERENCE.txt`

---

*Last Updated: November 19, 2024*
*GUI Version: 1.0*
*Built for: The Lariat Restaurant, Fort Collins, CO*
