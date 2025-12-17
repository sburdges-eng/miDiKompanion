#!/usr/bin/env python3
"""
Automated Migration Script: music_brain → miDEE, emotion → KELLY

This script automates the refactoring process:
1. Renames directories
2. Updates imports in Python files
3. Updates C++ includes and namespaces
4. Updates configuration files
5. Resets version numbers
"""

import os
import re
import shutil
import json
from pathlib import Path
from typing import List, Tuple


class MigrationTool:
    """Handles the migration from old names to miDEE/KELLY."""
    
    def __init__(self, repo_root: Path, dry_run: bool = False):
        self.repo_root = Path(repo_root)
        self.dry_run = dry_run
        self.changes_log = []
        
    def log_change(self, action: str, details: str):
        """Log a change for review."""
        msg = f"{action}: {details}"
        self.changes_log.append(msg)
        print(f"  {'[DRY RUN] ' if self.dry_run else ''}✓ {msg}")
    
    def rename_directory(self, old_path: str, new_path: str):
        """Rename a directory."""
        old_full = self.repo_root / old_path
        new_full = self.repo_root / new_path
        
        if not old_full.exists():
            print(f"  ⊘ Skip: {old_path} does not exist")
            return
        
        if new_full.exists():
            print(f"  ⚠ Skip: {new_path} already exists")
            return
            
        if not self.dry_run:
            # Ensure parent directory exists
            new_full.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_full), str(new_full))
        
        self.log_change("RENAME DIR", f"{old_path} → {new_path}")
    
    def update_python_file(self, filepath: Path):
        """Update imports and references in a Python file."""
        if not filepath.exists():
            return
        
        try:
            content = filepath.read_text()
            original = content
            
            # Update music_brain → midee
            content = re.sub(r'\bfrom music_brain\b', 'from midee', content)
            content = re.sub(r'\bimport music_brain\b', 'import midee', content)
            content = re.sub(r'\bmusic_brain\.', 'midee.', content)
            content = re.sub(r'"music_brain"', '"midee"', content)
            content = re.sub(r"'music_brain'", "'midee'", content)
            
            # Update emotion_thesaurus → kelly.thesaurus
            content = re.sub(r'\bfrom emotion_thesaurus\b', 'from kelly.thesaurus', content)
            content = re.sub(r'\bimport emotion_thesaurus\b', 'import kelly.thesaurus', content)
            content = re.sub(r'\bemotion_thesaurus\.', 'kelly.thesaurus.', content)
            
            # Update emotional_mapping → kelly.emotional_mapping
            content = re.sub(r'\bfrom emotional_mapping\b', 'from kelly.emotional_mapping', content)
            content = re.sub(r'\bimport emotional_mapping\b', 'import kelly.emotional_mapping', content)
            
            if content != original and not self.dry_run:
                filepath.write_text(content)
                self.log_change("UPDATE PY", str(filepath.relative_to(self.repo_root)))
            elif content != original:
                self.log_change("WOULD UPDATE PY", str(filepath.relative_to(self.repo_root)))
                
        except Exception as e:
            print(f"  ✗ Error updating {filepath}: {e}")
    
    def update_cpp_file(self, filepath: Path):
        """Update includes and namespaces in C++ files."""
        if not filepath.exists():
            return
            
        try:
            content = filepath.read_text()
            original = content
            
            # Update includes
            content = re.sub(r'#include\s+"music_brain/', '#include "midee/', content)
            content = re.sub(r'#include\s+<music_brain/', '#include <midee/', content)
            
            # Update namespaces
            content = re.sub(r'\bnamespace music_brain\b', 'namespace midee', content)
            content = re.sub(r'\bnamespace MusicBrain\b', 'namespace miDEE', content)
            content = re.sub(r'\bmusic_brain::', 'midee::', content)
            content = re.sub(r'\bMusicBrain::', 'miDEE::', content)
            
            # Update include guards
            content = re.sub(r'\bMUSIC_BRAIN_', 'MIDEE_', content)
            
            if content != original and not self.dry_run:
                filepath.write_text(content)
                self.log_change("UPDATE CPP", str(filepath.relative_to(self.repo_root)))
            elif content != original:
                self.log_change("WOULD UPDATE CPP", str(filepath.relative_to(self.repo_root)))
                
        except Exception as e:
            print(f"  ✗ Error updating {filepath}: {e}")
    
    def update_toml_file(self, filepath: Path):
        """Update pyproject.toml files."""
        if not filepath.exists():
            return
            
        try:
            content = filepath.read_text()
            original = content
            
            # Update package names
            content = re.sub(r'name\s*=\s*"music-brain"', 'name = "midee"', content)
            content = re.sub(r'name\s*=\s*"music_brain"', 'name = "midee"', content)
            
            if content != original and not self.dry_run:
                filepath.write_text(content)
                self.log_change("UPDATE TOML", str(filepath.relative_to(self.repo_root)))
            elif content != original:
                self.log_change("WOULD UPDATE TOML", str(filepath.relative_to(self.repo_root)))
                
        except Exception as e:
            print(f"  ✗ Error updating {filepath}: {e}")
    
    def update_json_file(self, filepath: Path):
        """Update package.json files."""
        if not filepath.exists():
            return
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            updated = False
            if 'name' in data and data['name'] in ['music-brain', 'music_brain']:
                data['name'] = 'midee'
                updated = True
            
            if updated and not self.dry_run:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                    f.write('\n')
                self.log_change("UPDATE JSON", str(filepath.relative_to(self.repo_root)))
            elif updated:
                self.log_change("WOULD UPDATE JSON", str(filepath.relative_to(self.repo_root)))
                
        except Exception as e:
            print(f"  ✗ Error updating {filepath}: {e}")
    
    def update_markdown_files(self, filepath: Path):
        """Update references in markdown files."""
        if not filepath.exists():
            return
            
        try:
            content = filepath.read_text()
            original = content
            
            # Update references
            content = re.sub(r'\bmusic_brain\b', 'midee', content)
            content = re.sub(r'\bMusic Brain\b', 'miDEE', content)
            content = re.sub(r'\bemotion_thesaurus\b', 'kelly.thesaurus', content)
            content = re.sub(r'\bEmotion Thesaurus\b', 'KELLY Thesaurus', content)
            
            if content != original and not self.dry_run:
                filepath.write_text(content)
                self.log_change("UPDATE MD", str(filepath.relative_to(self.repo_root)))
            elif content != original:
                self.log_change("WOULD UPDATE MD", str(filepath.relative_to(self.repo_root)))
                
        except Exception as e:
            print(f"  ✗ Error updating {filepath}: {e}")
    
    def migrate_directories(self):
        """Execute directory renaming."""
        print("\n" + "=" * 60)
        print("PHASE 1: Directory Renaming")
        print("=" * 60 + "\n")
        
        renames = [
            # Music → miDEE
            ("music_brain", "midee"),
            ("cpp_music_brain", "cpp_midee"),
            ("penta_core_music-brain", "penta_core_midee"),
            ("examples_music-brain", "examples_midee"),
            ("docs_music-brain", "docs_midee"),
            ("tests_music-brain", "tests_midee"),
        ]
        
        for old, new in renames:
            self.rename_directory(old, new)
        
        # Handle emotion → kelly specially
        print("\nCreating KELLY structure...")
        kelly_dir = self.repo_root / "kelly"
        if not self.dry_run:
            kelly_dir.mkdir(exist_ok=True)
            (kelly_dir / "thesaurus").mkdir(exist_ok=True)
            (kelly_dir / "rules").mkdir(exist_ok=True)
        self.log_change("CREATE DIR", "kelly/")
    
    def migrate_emotion_files(self):
        """Move emotion files to KELLY structure."""
        print("\n" + "=" * 60)
        print("PHASE 2: KELLY Structure Creation")
        print("=" * 60 + "\n")
        
        emotion_thesaurus = self.repo_root / "emotion_thesaurus"
        kelly_thesaurus = self.repo_root / "kelly" / "thesaurus"
        
        if emotion_thesaurus.exists():
            for file in emotion_thesaurus.iterdir():
                if file.is_file():
                    dest = kelly_thesaurus / file.name
                    if not self.dry_run:
                        shutil.copy2(str(file), str(dest))
                    self.log_change("COPY", f"{file.name} → kelly/thesaurus/")
        
        # Move top-level emotion files
        emotion_files = [
            ("emotional_mapping.py", "kelly/emotional_mapping.py"),
            ("auto_emotion_sampler.py", "kelly/emotion_sampler.py"),
        ]
        
        for old, new in emotion_files:
            old_path = self.repo_root / old
            new_path = self.repo_root / new
            if old_path.exists() and not self.dry_run:
                shutil.copy2(str(old_path), str(new_path))
            if old_path.exists():
                self.log_change("COPY", f"{old} → {new}")
    
    def update_all_files(self):
        """Update imports and references in all files."""
        print("\n" + "=" * 60)
        print("PHASE 3: Code Updates")
        print("=" * 60 + "\n")
        
        # Update Python files
        print("Updating Python files...")
        for py_file in self.repo_root.rglob('*.py'):
            if '.git' not in str(py_file) and 'node_modules' not in str(py_file):
                self.update_python_file(py_file)
        
        # Update C++ files
        print("\nUpdating C++ files...")
        for cpp_file in list(self.repo_root.rglob('*.cpp')) + list(self.repo_root.rglob('*.h')):
            if '.git' not in str(cpp_file):
                self.update_cpp_file(cpp_file)
        
        # Update TOML files
        print("\nUpdating TOML files...")
        for toml_file in self.repo_root.rglob('*.toml'):
            if '.git' not in str(toml_file):
                self.update_toml_file(toml_file)
        
        # Update JSON files
        print("\nUpdating JSON files...")
        for json_file in self.repo_root.rglob('*.json'):
            if '.git' not in str(json_file) and 'node_modules' not in str(json_file):
                self.update_json_file(json_file)
        
        # Update markdown files
        print("\nUpdating Markdown files...")
        for md_file in self.repo_root.rglob('*.md'):
            if '.git' not in str(md_file):
                self.update_markdown_files(md_file)
    
    def save_migration_log(self):
        """Save the migration log."""
        log_file = self.repo_root / "MIGRATION_LOG.txt"
        with open(log_file, 'w') as f:
            f.write("miDEE/KELLY Migration Log\n")
            f.write("=" * 60 + "\n\n")
            for change in self.changes_log:
                f.write(change + "\n")
        print(f"\n✓ Migration log saved to {log_file}")
    
    def run(self):
        """Execute the full migration."""
        print("\n" + "=" * 60)
        print("miDEE/KELLY Migration Tool")
        print("=" * 60)
        print(f"Mode: {'DRY RUN (no changes will be made)' if self.dry_run else 'LIVE (changes will be applied)'}")
        print("=" * 60 + "\n")
        
        self.migrate_directories()
        self.migrate_emotion_files()
        self.update_all_files()
        
        print("\n" + "=" * 60)
        print("Migration Summary")
        print("=" * 60)
        print(f"Total changes: {len(self.changes_log)}")
        
        if not self.dry_run:
            self.save_migration_log()
        
        print("\n✓ Migration complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate to miDEE/KELLY structure')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    
    args = parser.parse_args()
    
    tool = MigrationTool(Path(args.repo_root), dry_run=args.dry_run)
    tool.run()


if __name__ == '__main__':
    main()
