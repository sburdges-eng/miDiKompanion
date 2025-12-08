#!/usr/bin/env python3
import os
import zipfile
from pathlib import Path

MAX_SIZE = 25 * 1024 * 1024
OUT = Path("/Users/seanburdges/Desktop/REPOSEAN_ZIPS")

EXTS = {'.py','.js','.ts','.jsx','.tsx','.swift','.rs','.cpp','.c','.h','.hpp',
        '.java','.html','.css','.scss','.json','.yaml','.yml','.sh','.sql',
        '.xml','.toml','.cfg','.ini','.gradle','.cmake','.mk','.spec',
        '.plist','.md','.markdown','.mid','.midi','.wav','.txt','.m','.mm',
        '.go','.kt','.rb','.php','.lua','.r','.svg','.entitlements'}

SKIP = {'.git','node_modules','__pycache__','.venv','venv','JUCE','JUCE 2','JUCE 4',
        'Python.framework','Python3.framework','PythonT.framework',
        'JavaScriptCore.framework','_WebKit_SwiftUI.framework','PlugInKitDaemon.framework',
        '.tmp.drivedownload','.tmp.driveupload','ALL_ZIPS_ORGANIZED','ORGANIZED_ZIPS',
        'ZIP_ORGANIZED_FINAL','git-core','man','Archive_2_copy_extracted','REPOSEAN_ZIPS'}

files = []
for base in ["/Users/seanburdges/Desktop", "/Users/seanburdges/Documents"]:
    for root, dirs, fnames in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.endswith('.framework')]
        for f in fnames:
            if f.endswith('.zip'): continue
            ext = Path(f).suffix.lower()
            if ext in EXTS:
                fp = os.path.join(root, f)
                try:
                    sz = os.path.getsize(fp)
                    if 0 < sz < MAX_SIZE:
                        files.append((fp, sz))
                except: pass

print(f"Found {len(files)} files")
files.sort()

znum, csz, cfiles = 1, 0, []
for fp, sz in files:
    if csz + sz > MAX_SIZE and cfiles:
        zp = OUT / f"reposean{znum}.zip"
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
            for f in cfiles:
                try: z.write(f, os.path.relpath(f, "/Users/seanburdges"))
                except: pass
        print(f"Created {zp.name} ({os.path.getsize(zp)/1024/1024:.1f}MB)")
        znum += 1
        csz, cfiles = 0, []
    cfiles.append(fp)
    csz += sz

if cfiles:
    zp = OUT / f"reposean{znum}.zip"
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in cfiles:
            try: z.write(f, os.path.relpath(f, "/Users/seanburdges"))
            except: pass
    print(f"Created {zp.name} ({os.path.getsize(zp)/1024/1024:.1f}MB)")

print(f"\nDone! {znum} zips in {OUT}")
