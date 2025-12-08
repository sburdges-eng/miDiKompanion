#!/usr/bin/env python3
"""
Google Drive Spreadsheet Downloader
This script helps download spreadsheets from Google Drive
"""

import os
from pathlib import Path
import subprocess
import json

def setup_google_drive_access():
    """Instructions for setting up Google Drive access"""
    
    instructions = """
    ============================================
    GOOGLE DRIVE SPREADSHEET DOWNLOAD SETUP
    ============================================
    
    Option 1: Using Google Drive Desktop App (Easiest)
    ---------------------------------------------------
    1. Install Google Drive for Desktop:
       https://www.google.com/drive/download/
    
    2. Sign in with sburdges@gmail.com
    
    3. Your files will sync to:
       - Mac: ~/Google Drive/My Drive/
       - Windows: G:\My Drive\ (or similar)
    
    4. Run the main script to copy synced files
    
    Option 2: Manual Download from Web
    ---------------------------------------------------
    1. Go to https://drive.google.com
    2. Sign in with sburdges@gmail.com
    3. Search for file type:
       - In search bar, click the filter icon
       - Select "Type" → "Spreadsheets"
    4. Select all spreadsheets:
       - Click first file
       - Shift+Click last file
    5. Right-click → Download
    6. Files will download as .xlsx to Downloads folder
    7. Run the main script to organize them
    
    Option 3: Using Google Takeout (For Everything)
    ---------------------------------------------------
    1. Go to https://takeout.google.com
    2. Deselect all
    3. Select only "Drive"
    4. Click "All Drive data included"
    5. Choose "Specific file types" → Spreadsheets
    6. Create export → Download
    7. Extract the archive
    8. Run the main script on extracted folder
    
    ============================================
    """
    
    print(instructions)
    
    # Save instructions to file
    desktop = Path.home() / "Desktop"
    instructions_file = desktop / "XLprogram" / "GOOGLE_DRIVE_INSTRUCTIONS.txt"
    instructions_file.parent.mkdir(exist_ok=True)
    
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"Instructions saved to: {instructions_file}")

def create_powershell_script():
    """Create PowerShell script for Windows users"""
    
    ps_script = '''# PowerShell Script to Gather Spreadsheets
# Run this in PowerShell as Administrator

$desktopPath = [Environment]::GetFolderPath("Desktop")
$xlFolder = "$desktopPath\XLprogram"

# Create folder
New-Item -Path $xlFolder -ItemType Directory -Force

Write-Host "Searching for spreadsheet files..." -ForegroundColor Green

# Search locations
$searchPaths = @(
    [Environment]::GetFolderPath("MyDocuments"),
    [Environment]::GetFolderPath("Desktop"),
    "$env:USERPROFILE\Downloads",
    "$env:USERPROFILE\Google Drive",
    "$env:USERPROFILE\OneDrive"
)

$fileTypes = @("*.xlsx", "*.xls", "*.csv", "*.xlsm")
$foundFiles = @()

foreach ($path in $searchPaths) {
    if (Test-Path $path) {
        Write-Host "Searching in: $path"
        foreach ($type in $fileTypes) {
            $files = Get-ChildItem -Path $path -Filter $type -Recurse -ErrorAction SilentlyContinue
            $foundFiles += $files
        }
    }
}

Write-Host "Found $($foundFiles.Count) spreadsheet files" -ForegroundColor Yellow

# Copy files
$copied = 0
foreach ($file in $foundFiles) {
    $destPath = "$xlFolder\$($file.Name)"
    
    # Add number if file exists
    $counter = 1
    while (Test-Path $destPath) {
        $destPath = "$xlFolder\$($file.BaseName)_$counter$($file.Extension)"
        $counter++
    }
    
    Copy-Item -Path $file.FullName -Destination $destPath -Force
    Write-Host "  Copied: $($file.Name)"
    $copied++
}

Write-Host "`nComplete! Copied $copied files to $xlFolder" -ForegroundColor Green
Write-Host "Press any key to open the folder..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Open the folder
Start-Process $xlFolder
'''
    
    desktop = Path.home() / "Desktop"
    ps_file = desktop / "XLprogram" / "gather_spreadsheets.ps1"
    ps_file.parent.mkdir(exist_ok=True)
    
    with open(ps_file, 'w') as f:
        f.write(ps_script)
    
    print(f"PowerShell script created: {ps_file}")
    return ps_file

def create_batch_script():
    """Create batch script for easy execution on Windows"""
    
    batch_script = '''@echo off
echo ============================================
echo SPREADSHEET GATHERER FOR WINDOWS
echo ============================================
echo.

echo Creating XLprogram folder on Desktop...
mkdir "%USERPROFILE%\Desktop\XLprogram" 2>nul

echo.
echo Searching for Excel files...
echo.

set count=0

echo Searching in Documents...
for /R "%USERPROFILE%\Documents" %%f in (*.xlsx *.xls *.csv *.xlsm) do (
    copy "%%f" "%USERPROFILE%\Desktop\XLprogram\" >nul 2>&1
    set /a count+=1
    echo   Found: %%~nxf
)

echo Searching in Downloads...
for /R "%USERPROFILE%\Downloads" %%f in (*.xlsx *.xls *.csv *.xlsm) do (
    copy "%%f" "%USERPROFILE%\Desktop\XLprogram\" >nul 2>&1
    set /a count+=1
    echo   Found: %%~nxf
)

echo Searching in Desktop...
for /R "%USERPROFILE%\Desktop" %%f in (*.xlsx *.xls *.csv *.xlsm) do (
    if not "%%~dpf"=="%USERPROFILE%\Desktop\XLprogram\" (
        copy "%%f" "%USERPROFILE%\Desktop\XLprogram\" >nul 2>&1
        set /a count+=1
        echo   Found: %%~nxf
    )
)

echo.
echo ============================================
echo COMPLETE! Files copied to XLprogram folder
echo ============================================
echo.
echo Opening folder now...
start "" "%USERPROFILE%\Desktop\XLprogram"

pause
'''
    
    desktop = Path.home() / "Desktop"
    batch_file = desktop / "XLprogram" / "RUN_GATHERER.bat"
    batch_file.parent.mkdir(exist_ok=True)
    
    with open(batch_file, 'w') as f:
        f.write(batch_script)
    
    print(f"Batch file created: {batch_file}")
    return batch_file

if __name__ == "__main__":
    print("="*50)
    print("GOOGLE DRIVE & SPREADSHEET SETUP")
    print("="*50)
    
    # Show instructions
    setup_google_drive_access()
    
    # Create helper scripts
    print("\nCreating helper scripts...")
    
    if os.name == 'nt':  # Windows
        create_batch_script()
        create_powershell_script()
        print("\n✓ Windows scripts created!")
        print("  Run: RUN_GATHERER.bat (double-click it)")
    
    print("\n" + "="*50)
    print("Follow the instructions above to access your Google Drive files")
    print("="*50)
