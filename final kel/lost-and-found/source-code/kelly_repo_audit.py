#!/usr/bin/env python3
"""
Kelly Project Repository Consolidation Audit
Analyzes 5 repos to identify unique features, duplicates, and consolidation strategy
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configuration
REPOS = [
    'penta-core',
    '1DAW1',
    'iDAW',
    'iDAWi',
    'DAiW-Music-Brain'
]

BASE_DIR = Path.home() / 'kelly-consolidation'

# File patterns to analyze
CODE_EXTENSIONS = {
    '.rs', '.ts', '.tsx', '.js', '.jsx', '.py', 
    '.json', '.toml', '.yaml', '.yml', '.md'
}

IGNORE_DIRS = {
    'node_modules', '.git', 'target', 'dist', 'build',
    '__pycache__', '.next', '.vscode', '.idea', 'coverage'
}

class RepoAnalyzer:
    def __init__(self, repo_name, base_path):
        self.name = repo_name
        self.path = base_path / repo_name
        self.files = {}
        self.directories = set()
        self.stats = {
            'total_files': 0,
            'code_files': 0,
            'total_size': 0,
            'file_types': defaultdict(int)
        }
        
    def scan(self):
        """Scan the repository and collect file information"""
        if not self.path.exists():
            print(f"⚠️  {self.name} not found at {self.path}")
            return False
            
        print(f"📁 Scanning {self.name}...")
        
        for root, dirs, files in os.walk(self.path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            rel_root = Path(root).relative_to(self.path)
            self.directories.add(str(rel_root))
            
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.path)
                
                try:
                    size = file_path.stat().st_size
                    ext = file_path.suffix.lower()
                    
                    self.files[str(rel_path)] = {
                        'size': size,
                        'ext': ext,
                        'is_code': ext in CODE_EXTENSIONS
                    }
                    
                    self.stats['total_files'] += 1
                    self.stats['total_size'] += size
                    self.stats['file_types'][ext] += 1
                    
                    if ext in CODE_EXTENSIONS:
                        self.stats['code_files'] += 1
                        
                except Exception as e:
                    print(f"  ⚠️  Error reading {rel_path}: {e}")
                    
        return True
    
    def get_structure_summary(self):
        """Get a summary of the directory structure"""
        top_level_dirs = set()
        for d in self.directories:
            if d != '.':
                parts = Path(d).parts
                if len(parts) > 0:
                    top_level_dirs.add(parts[0])
        return sorted(top_level_dirs)

def compare_repos(analyzers):
    """Compare all repos and identify similarities and differences"""
    print("\n" + "="*70)
    print("📊 COMPARISON ANALYSIS")
    print("="*70)
    
    # Find common directory structures
    all_dirs = defaultdict(list)
    for analyzer in analyzers:
        for dir_name in analyzer.get_structure_summary():
            all_dirs[dir_name].append(analyzer.name)
    
    print("\n🗂️  Common Top-Level Directories:")
    common_dirs = {k: v for k, v in all_dirs.items() if len(v) > 1}
    for dir_name, repos in sorted(common_dirs.items()):
        print(f"  • {dir_name}: {', '.join(repos)}")
    
    print("\n🎯 Unique Top-Level Directories:")
    unique_dirs = {k: v for k, v in all_dirs.items() if len(v) == 1}
    for dir_name, repos in sorted(unique_dirs.items()):
        print(f"  • {dir_name}: {repos[0]}")
    
    # Find duplicate files (same relative path)
    all_files = defaultdict(list)
    for analyzer in analyzers:
        for file_path in analyzer.files.keys():
            all_files[file_path].append(analyzer.name)
    
    duplicates = {k: v for k, v in all_files.items() if len(v) > 1}
    print(f"\n📋 Duplicate Files: {len(duplicates)} files found in multiple repos")
    
    # Show most common duplicates
    if duplicates:
        print("\n  Most common duplicates:")
        sorted_dupes = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
        for file_path, repos in sorted_dupes[:10]:
            if any(ext in file_path for ext in ['.ts', '.tsx', '.rs', '.py']):
                print(f"    • {file_path}")
                print(f"      Found in: {', '.join(repos)}")

def generate_consolidation_plan(analyzers):
    """Generate a step-by-step consolidation plan"""
    print("\n" + "="*70)
    print("📝 CONSOLIDATION ROADMAP")
    print("="*70)
    
    # Find the base repo (iDAW)
    base = next((a for a in analyzers if a.name == 'iDAW'), analyzers[0])
    others = [a for a in analyzers if a.name != base.name and a.name != '1DAW1']
    
    print(f"\n✅ Base Repository: {base.name} (most comprehensive)")
    print(f"   Files: {base.stats['code_files']} code files")
    print(f"   Size: {base.stats['total_size'] / 1024 / 1024:.2f} MB")
    
    print("\n📦 Repositories to Merge:")
    for analyzer in others:
        print(f"\n  {analyzer.name}:")
        print(f"    • Files: {analyzer.stats['code_files']} code files")
        print(f"    • Size: {analyzer.stats['total_size'] / 1024 / 1024:.2f} MB")
        
        # Find unique directories
        unique_to_this = set(analyzer.get_structure_summary()) - set(base.get_structure_summary())
        if unique_to_this:
            print(f"    • Unique directories: {', '.join(sorted(unique_to_this))}")
    
    print("\n" + "-"*70)
    print("🎯 RECOMMENDED STRATEGY:")
    print("-"*70)
    print("""
1. PREPARATION:
   • Create backup of all 5 repos
   • Clone iDAW to 1DAW1 as starting point
   
2. FEATURE EXTRACTION (in order):
   
   a) DAiW-Music-Brain:
      • Extract: Core music processing logic, Music Vault
      • Integrate into: src/backend/music-brain/
   
   b) penta-core:
      • Extract: Core Tauri/Rust foundation
      • Integrate into: src-tauri/ or src/core/
   
   c) iDAWi:
      • Extract: Unique frontend features (check for differences from iDAW)
      • Look for: Dreamstate features, Parrot implementation
      • Integrate into: src/frontend/dreamstate/
   
3. CLEANUP:
   • Remove duplicate node_modules, target directories
   • Consolidate package.json dependencies
   • Fix case-sensitivity issues (iDAWi vs idawi)
   
4. TESTING:
   • Test each integrated feature separately
   • Verify no broken imports
   • Run build to confirm everything compiles
   
5. DOCUMENTATION:
   • Create CONSOLIDATION_LOG.md tracking what came from where
   • Update README with unified architecture
   • Document feature locations
""")

def generate_feature_matrix(analyzers):
    """Generate a feature matrix showing what's in each repo"""
    print("\n" + "="*70)
    print("🎨 FEATURE DETECTION")
    print("="*70)
    
    # Feature indicators (file/directory names that suggest features)
    features = {
        'Side A/Side B UI': ['SideA', 'SideB', 'cassette'],
        'DAW Professional': ['daw', 'mixer', 'timeline', 'transport'],
        'Emotion Wheel': ['emotion', 'EmotionWheel', 'thesaurus'],
        'Dreamstate': ['dreamstate', 'dream'],
        'Parrot': ['parrot', 'Parrot'],
        'Music Brain': ['MusicBrain', 'music-brain', 'musicbrain'],
        'Music Vault': ['MusicVault', 'music-vault', 'vault'],
        'AI/Interrogation': ['interrogat', 'ai-engine', 'GhostWriter'],
        'Tauri Backend': ['src-tauri', 'tauri.conf'],
    }
    
    print("\nFeature Detection Matrix:")
    print(f"\n{'Feature':<25} | " + " | ".join(f"{r.name:<15}" for r in analyzers))
    print("-" * (25 + 19 * len(analyzers)))
    
    for feature_name, keywords in features.items():
        row = f"{feature_name:<25} | "
        for analyzer in analyzers:
            found = False
            for file_path in analyzer.files.keys():
                if any(kw.lower() in file_path.lower() for kw in keywords):
                    found = True
                    break
            row += f"{'✓':<15} | " if found else f"{'—':<15} | "
        print(row)

def main():
    print("="*70)
    print("🎵 KELLY PROJECT REPOSITORY CONSOLIDATION AUDIT")
    print("="*70)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Scan all repositories
    analyzers = []
    for repo_name in REPOS:
        analyzer = RepoAnalyzer(repo_name, BASE_DIR)
        if analyzer.scan():
            analyzers.append(analyzer)
    
    # Repository Statistics
    print("\n" + "="*70)
    print("📈 REPOSITORY STATISTICS")
    print("="*70)
    
    for analyzer in analyzers:
        print(f"\n{analyzer.name}:")
        print(f"  Total files: {analyzer.stats['total_files']}")
        print(f"  Code files: {analyzer.stats['code_files']}")
        print(f"  Total size: {analyzer.stats['total_size'] / 1024 / 1024:.2f} MB")
        print(f"  Top directories: {', '.join(analyzer.get_structure_summary()[:5])}")
    
    # Compare repositories
    compare_repos(analyzers)
    
    # Feature detection
    generate_feature_matrix(analyzers)
    
    # Generate consolidation plan
    generate_consolidation_plan(analyzers)
    
    print("\n" + "="*70)
    print("✅ AUDIT COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. Review the analysis above")
    print("2. Manually explore unique features in each repo")
    print("3. Follow the consolidation roadmap")
    print("4. Create detailed migration checklist")
    print("\n")

if __name__ == "__main__":
    main()
