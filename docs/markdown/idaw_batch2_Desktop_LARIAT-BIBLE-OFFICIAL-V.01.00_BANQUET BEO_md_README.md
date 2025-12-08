# The Lariat Banquet Management System
## Complete End-to-End Event Operations & Analytics Platform

> **Version 1.0** | Built for The Lariat Restaurant | Fort Collins, Colorado

---

## 🎯 System Overview

This comprehensive system manages your entire banquet operation from quote to completion, with automated data analysis and reporting.

### What This System Does

✅ **Create client quotes** with auto-calculated pricing
✅ **Plan kitchen prep** with day-by-day assignments
✅ **Schedule production** from T-7 days through event
✅ **Execute events** with hour-by-hour checklists
✅ **Analyze data** automatically from all invoices
✅ **Generate reports** monthly with trends and insights

---

## 📊 System Performance (6 Events Analyzed)

| Metric | Value |
|--------|-------|
| **Total Revenue** | $75,131.07 |
| **Average Event** | $12,521.85 |
| **Top Revenue Item** | OPEN BAR BASIC ($9,810) |
| **Most Popular Item** | Rope Caesar Salad (5 orders) |
| **Bar Revenue %** | 59.2% of subtotal |
| **Taco Buffet Revenue** | $8,850 (14.9%) |

---

## 🗂️ Complete File Structure

```
BANQUET BEO/
│
├── 📋 TEMPLATES (Use these for new events)
│   ├── Invoice_Template_v3.0_FIXED.xlsx
│   ├── Kitchen_Prep_Sheet_Template_v1.0.xlsx
│   ├── Production_Schedule_Template_v1.0.xlsx
│   └── Event_Execution_Plan_Template_v1.0.xlsx
│
├── 🤖 AUTOMATION SYSTEM
│   ├── lariat_invoice_automation.py       ⭐ Main automation script
│   ├── lariat_automation.command          ⭐ Double-click launcher
│   ├── AUTOMATION_GUIDE.md                📖 Complete guide
│   └── QUICK_REFERENCE.txt                📋 Command cheat sheet
│
├── 📊 DATABASE & ANALYSIS
│   ├── Lariat_BEO_Database_Analysis.xlsx  ⭐ Master database (7 sheets)
│   └── .processed_invoices.json           🔒 Auto-tracking file
│
├── 📁 DATA FOLDERS
│   ├── INVOICE SPREADSHEETS/              💾 Archive of all invoices
│   └── reports/                           📄 Monthly reports (auto-generated)
│
├── 📄 DOCUMENTATION
│   ├── README.md                          📖 This file
│   ├── PROJECT_RULES.md                   📋 Business rules & specs
│   └── claude m.g. 1/
│       ├── CLAUDE_CODE_GUIDELINES.md      💻 Coding standards
│       └── INVOICE_TEMPLATE_BUILD_COMPLETE.md
│
└── 📦 SAMPLE FILES
    └── Event_Execution_Plan_Anderson_Wedding_2025-12-15_SAMPLE.xlsx
```

---

## 🚀 Quick Start Guide

### For New Events: 4-Step Workflow

#### 1️⃣ CREATE INVOICE (Client Quote)

**Use:** `Invoice_Template_v3.0_FIXED.xlsx`

1. Create copy: `Invoice_[ClientName]_[MM-DD-YY].xlsx`
2. Fill in client info (rows 2-4)
3. Select menu items from dropdown (column A)
4. Enter quantities (column C)
5. **Auto-calculates:** pricing, tax, service fee, total

**Features:**
- 53 menu items with auto-pricing via VLOOKUP
- Tax (8.15%) and service fee (20%) auto-calculated
- Minimum spend validation ($3,500)
- All formulas working and tested

---

#### 2️⃣ PLAN KITCHEN PREP

**Use:** `Kitchen_Prep_Sheet_Template_v1.0.xlsx`

1. Create copy: `KitchenPrep_[ClientName]_[MM-DD-YY].xlsx`
2. Enter event info (row 2)
3. Assign prep days for each item (dropdown: T-3, T-2, T-1, Day-of)
4. Add plating instructions and notes
5. Use Recipe Scaling Calculator sheet if needed

**Features:**
- 23 menu item rows
- Prep day dropdowns (Thursday/Friday/Saturday/Day-of)
- Recipe scaling calculator (auto-scales 20 ingredients)
- Timing standards and equipment checklist

---

#### 3️⃣ SCHEDULE PRODUCTION (T-7 Days Out)

**Use:** `Production_Schedule_Template_v1.0.xlsx`

1. Create copy: `ProductionSchedule_[ClientName]_[MM-DD-YY].xlsx`
2. Fill in timeline tasks (T-7 through event day)
3. Track ingredient ordering (40 rows with vendor dropdowns)
4. Assign staff roles (20 rows with position dropdowns)
5. Check equipment needs (30 items)

**Features:**
- Complete T-7 day timeline (6 days + event day)
- Ingredient ordering tracker with vendor dropdowns (Shamrock, SYSCO, etc.)
- Staff assignment sheet with role dropdowns
- Equipment checklist with source tracking

---

#### 4️⃣ EXECUTE EVENT (Day-Of)

**Use:** `Event_Execution_Plan_Template_v1.0.xlsx`

1. Create copy: `EventExecution_[ClientName]_[MM-DD-YY].xlsx`
2. Follow hour-by-hour timeline (-3:00 through +1:00)
3. Complete quality control checks (temperature, presentation)
4. Track staff positions and responsibilities
5. Fill out post-event debrief form

**Features:**
- 52 execution tasks with time markers
- Temperature check requirements (hot 140°F+, cold 40°F-)
- 8 pre-filled staff positions + custom rows
- 17 emergency contacts
- 30+ question post-event debrief

---

### Automation: Import & Analyze Data

#### 🤖 DOUBLE-CLICK AUTOMATION

**Easiest Method:**

1. Save invoice to `/INVOICE SPREADSHEETS/` folder
2. **Double-click** `lariat_automation.command`
3. Choose **Option 1: Import New Invoices**
4. Database auto-updates with all event data!

#### 📊 GENERATE MONTHLY REPORTS

1. **Double-click** `lariat_automation.command`
2. Choose **Option 2: Generate Monthly Report**
3. Enter year and month (or press Enter for current)
4. Report saved to `/reports/` folder

**Report includes:**
- Total revenue and event count
- Average event size
- Top 5 menu items
- Event-by-event breakdown
- Category analysis (bar, tacos, etc.)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| **README.md** (this file) | System overview and quick start |
| **AUTOMATION_GUIDE.md** | Complete automation system guide |
| **QUICK_REFERENCE.txt** | Command cheat sheet |
| **PROJECT_RULES.md** | Business rules and specifications |
| **CLAUDE_CODE_GUIDELINES.md** | Excel coding standards |

---

## 💡 System Features

### Templates (Excel)

#### Invoice Template
- ✅ 53 menu items with auto-pricing
- ✅ VLOOKUP formulas for instant pricing
- ✅ Tax (8.15%) auto-calculated
- ✅ Service fee (20%) auto-calculated
- ✅ Minimum spend check ($3,500)
- ✅ Professional formatting with color coding

#### Kitchen Prep Sheet
- ✅ Prep day assignments (T-3, T-2, T-1, Day-of)
- ✅ Recipe scaling calculator
- ✅ Equipment and timing standards
- ✅ Plating instructions and notes

#### Production Schedule
- ✅ T-7 day complete timeline
- ✅ Ingredient ordering tracker (40 rows)
- ✅ Staff assignment sheet (20 rows)
- ✅ Equipment checklist (30 items)
- ✅ Vendor dropdowns (Shamrock, SYSCO, US Foods, etc.)

#### Event Execution Plan
- ✅ Hour-by-hour event timeline
- ✅ Pre-event setup checklist (12 tasks)
- ✅ Quality control checklist
- ✅ Staff positioning (8 positions)
- ✅ Emergency contacts (17 entries)
- ✅ Post-event debrief (30+ questions)

### Automation System (Python)

#### Automatic Invoice Import
- ✅ Scans `/INVOICE SPREADSHEETS/` for new files
- ✅ Extracts client, date, items, pricing
- ✅ Updates database (7 sheets)
- ✅ Tracks processed files (no duplicates)

#### Database Analysis (7 Sheets)
1. **Event Summary** - All events with totals
2. **All Line Items** - Every item from every event
3. **Menu Item Popularity** - Frequency rankings
4. **Top Revenue Items** - Highest earners
5. **Monthly Summary** - Revenue by month
6. **Pricing History** - Price tracking and variance
7. **Executive Summary** - Insights and recommendations

#### Monthly Reports
- ✅ Auto-generates text reports
- ✅ Revenue statistics
- ✅ Top items analysis
- ✅ Category breakdown
- ✅ Event-by-event details

---

## 📈 Key Insights from Historical Data

### Revenue Drivers

**Top 5 Revenue Items:**
1. OPEN BAR BASIC - $9,810 (16.5%)
2. Bar Spend Amount - $9,460 (15.9%)
3. Full open bar 3 hours - $7,200 (12.1%)
4. Bar Budget - $3,875 (6.5%)
5. Full open bar - $3,435 (5.8%)

**Most Popular Items (by frequency):**
1. Rope Caesar Salad Buffet (5 orders)
2. Green Chile Mac Buffet (5 orders)
3. Battered Avocado Taco Buffet (4 orders)
4. Rope Burger slider (4 orders)
5. Trio Dips (4 orders)

### Category Breakdown

| Category | Revenue | % of Subtotal |
|----------|---------|---------------|
| **Bar Services** | $35,180 | 59.2% |
| **Taco Buffets** | $8,850 | 14.9% |
| **Salads** | $3,150 | 5.3% |
| **Desserts** | $1,050 | 1.8% |

### Strategic Recommendations

💡 **Bar services are the #1 revenue driver** - Consider:
- Creating bar + food combo packages
- Expanding cocktail menu options
- Standardizing bar pricing tiers

💡 **Taco buffets have broad appeal** - Consider:
- Featuring prominently in marketing
- Adding seasonal taco variations
- Creating "Taco Bar" package deals

💡 **Average event size is $12,521** - Consider:
- Creating tiered packages ($10K, $15K, $20K)
- Minimum spend packages around this size
- Upsell opportunities for smaller events

---

## 🎨 Color Coding System

All templates use consistent color coding:

- 🟨 **Yellow (#FFF2CC):** Formula cells (do not edit)
- ⬜ **White (#FFFFFF):** Input cells (enter data here)
- 🟦 **Blue (#DDEBF7):** Dropdown cells (select from list)
- 🟩 **Green (#70AD47):** Section headers
- 🟥 **Red (#C00000):** Critical sections (event day)
- ⬜ **Gray (#E7E6E6):** Totals/calculated sections

---

## 🔧 Technical Requirements

### Required Software
- **Microsoft Excel** or compatible spreadsheet software
- **Python 3.8+** (for automation system)
- **openpyxl** Python library (auto-installed with automation)

### Installation (Automation Only)

```bash
# Install Python dependencies (if needed)
pip3 install openpyxl

# Make launcher executable
chmod +x lariat_automation.command
```

### System Tested On
- macOS (primary platform)
- Python 3.13
- Microsoft Excel for Mac
- openpyxl 3.1.2

---

## 📋 Best Practices

### File Naming Conventions

**Invoices:**
```
Invoice_[ClientName]_[YYYY-MM-DD]_v[#].xlsx

Examples:
✅ Invoice_Anderson_Wedding_2024-12-15_v1.xlsx
✅ Invoice_Smith_Holiday_Party_2025-01-10_v2.xlsx
```

**Kitchen Prep:**
```
KitchenPrep_[ClientName]_[YYYY-MM-DD].xlsx
```

**Production Schedule:**
```
ProductionSchedule_[ClientName]_[YYYY-MM-DD].xlsx
```

**Event Execution:**
```
EventExecution_[ClientName]_[YYYY-MM-DD].xlsx
```

### Version Control

- Increment version number for changes (v1, v2, v3)
- Always save originals in `/INVOICE SPREADSHEETS/`
- Backup database monthly

### Data Entry

- Always use dropdowns when available
- Enter dates in YYYY-MM-DD format
- Use consistent client name spelling
- Save invoices before importing

---

## 🆘 Troubleshooting

### Templates Not Calculating

**Issue:** Formulas showing as text or not calculating

**Solution:**
1. Check that formulas start with `=`
2. Verify Excel is set to auto-calculate (Formulas → Calculation Options → Automatic)
3. Try F9 to force recalculation
4. If persistent, see CLAUDE_CODE_GUIDELINES.md for formula fixes

### Automation Can't Find Invoices

**Issue:** "No new invoices found" message

**Solution:**
1. Verify files are in `/INVOICE SPREADSHEETS/` folder
2. Check file naming: must include client name and date (M_D or M_D_YY)
3. Check that files end with `.xlsx`
4. If reprocessing needed, delete `.processed_invoices.json`

### Database Not Updating

**Issue:** New data not appearing in database

**Solution:**
1. Verify `Lariat_BEO_Database_Analysis.xlsx` is not open in Excel
2. Check that automation completed without errors
3. Verify date format in Event Summary sheet (YYYY-MM-DD)
4. Check file permissions (should be writable)

---

## 📞 Support

### Getting Help

1. **Check documentation:**
   - AUTOMATION_GUIDE.md (automation issues)
   - QUICK_REFERENCE.txt (command syntax)
   - PROJECT_RULES.md (business rules)

2. **Check sample files:**
   - Event_Execution_Plan_Anderson_Wedding_SAMPLE.xlsx

3. **Verify setup:**
   - Templates have correct formulas
   - Python 3 installed and accessible
   - File permissions correct

---

## 🔮 Future Enhancements

### Potential Additions

**Phase 1 (Current):** ✅ Complete
- Templates for all workflow stages
- Historical data analysis
- Monthly reporting automation

**Phase 2 (Planned):**
- [ ] Web dashboard for real-time analytics
- [ ] Inventory tracking integration
- [ ] Automatic email invoicing
- [ ] Client portal for event planning

**Phase 3 (Future):**
- [ ] Mobile app for day-of execution
- [ ] Staff scheduling automation
- [ ] Vendor price comparison alerts
- [ ] Predictive analytics for menu trends

---

## 📊 Success Metrics

### How to Measure System Impact

**Efficiency Gains:**
- Time saved on data entry (manual → automated)
- Faster quote generation (template-based)
- Reduced errors in pricing and calculations

**Business Insights:**
- Revenue trends by month
- Menu item performance
- Pricing optimization opportunities
- Client repeat business tracking

**Operational Improvements:**
- Consistent event execution
- Better prep planning
- Staff accountability
- Quality control tracking

---

## 🎓 Training Recommendations

### For New Users

**Week 1: Templates**
- Learn Invoice Template (2 hours)
- Practice Kitchen Prep Sheet (1 hour)
- Review sample files

**Week 2: Workflow**
- Create test event using all 4 templates
- Follow complete workflow start to finish
- Practice with different menu combinations

**Week 3: Automation**
- Learn automation basics (QUICK_REFERENCE.txt)
- Import sample invoice
- Generate monthly report

**Week 4: Analysis**
- Explore database sheets
- Understand key metrics
- Generate insights from historical data

---

## 📝 Changelog

### Version 1.0 (2024-11-19)

**Initial Release:**
- ✅ 4 production templates (Invoice, Kitchen Prep, Production, Execution)
- ✅ Automated invoice import system
- ✅ 7-sheet analysis database
- ✅ Monthly report generator
- ✅ Historical data analysis (6 events)
- ✅ Complete documentation suite
- ✅ Command-line and GUI automation interfaces

**Files Created:**
- Invoice_Template_v3.0_FIXED.xlsx
- Kitchen_Prep_Sheet_Template_v1.0.xlsx
- Production_Schedule_Template_v1.0.xlsx
- Event_Execution_Plan_Template_v1.0.xlsx
- Lariat_BEO_Database_Analysis.xlsx
- lariat_invoice_automation.py
- lariat_automation.command
- AUTOMATION_GUIDE.md
- QUICK_REFERENCE.txt
- README.md (this file)

**Data Imported:**
- 6 historical events (July 2024 - October 2025)
- 73 line items
- $75,131.07 total revenue analyzed

---

## 🏆 Credits

**Built for:** The Lariat Restaurant
**Location:** Fort Collins, Colorado
**Owner:** Sean Burdges
**Version:** 1.0
**Date:** November 19, 2024

**Technologies:**
- Python 3.13
- openpyxl 3.1.2
- Microsoft Excel
- macOS automation

---

## 📜 License & Usage

This system is proprietary to The Lariat Restaurant. All templates, automation scripts, and documentation are for internal use only.

**Do not distribute without permission.**

---

**🎉 The complete Lariat Banquet Management System is ready for production use!**

For questions or support, refer to the documentation files or contact system administrator.

---

*Last Updated: November 19, 2024*
*System Version: 1.0*
*Status: Production Ready*
