#!/usr/bin/env python3
"""
Semantic Version Manager for miDiKompanion

Manages version numbers across multiple files following semantic versioning:
- MAJOR.MINOR.PATCH format
- Bug fixes increment PATCH (X.Y.Z+1)
- Features increment MINOR (X.Y+1.0)
- Breaking changes/builds increment MAJOR (X+1.0.0)

Tracks file dates and content changes to determine version bumps.
"""

import re
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import subprocess


@dataclass
class Version:
    """Semantic version representation."""
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> "Version":
        """Parse version string like '1.2.3' or 'v1.2.3'."""
        # Remove 'v' prefix if present
        version_str = version_str.lstrip('v')
        # Handle malformed versions like '1.0.04'
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2])
        )

    def bump_major(self) -> "Version":
        """Increment major version, reset minor and patch."""
        return Version(self.major + 1, 0, 0)

    def bump_minor(self) -> "Version":
        """Increment minor version, reset patch."""
        return Version(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "Version":
        """Increment patch version."""
        return Version(self.major, self.minor, self.patch + 1)


class VersionManager:
    """Manages semantic versioning across multiple project files."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.version_file = self.repo_root / "VERSION"
        self.pyproject_toml = self.repo_root / "pyproject.toml"
        self.package_json = self.repo_root / "package.json"
        self.version_h = self.repo_root / "iDAW_Core" / "include" / "Version.h"

    def get_current_version(self) -> Version:
        """Read current version from VERSION file."""
        if not self.version_file.exists():
            return Version(0, 1, 0)
        
        version_str = self.version_file.read_text().strip()
        try:
            return Version.from_string(version_str)
        except ValueError:
            # Handle malformed versions
            print(f"Warning: Malformed version in VERSION file: {version_str}")
            return Version(1, 0, 0)

    def update_version_file(self, version: Version) -> None:
        """Update VERSION file."""
        self.version_file.write_text(f"{version}\n")
        print(f"✓ Updated VERSION: {version}")

    def update_pyproject_toml(self, version: Version) -> None:
        """Update version in pyproject.toml."""
        if not self.pyproject_toml.exists():
            print(f"⊘ Skipped pyproject.toml (not found)")
            return

        content = self.pyproject_toml.read_text()
        lines = content.split('\n')
        new_lines = []
        in_project = False
        version_updated = False
        
        for line in lines:
            # Track if we're in [project] section
            if line.strip().startswith('[project]'):
                in_project = True
                new_lines.append(line)
            elif line.strip().startswith('[') and in_project:
                # Entering a new section, exit project section
                in_project = False
                new_lines.append(line)
            elif in_project and not version_updated and re.match(r'^\s*version\s*=', line):
                # Update the version line in project section
                new_lines.append(f'version = "{version}"')
                version_updated = True
            else:
                new_lines.append(line)
        
        self.pyproject_toml.write_text('\n'.join(new_lines))
        print(f"✓ Updated pyproject.toml: {version}")

    def update_package_json(self, version: Version) -> None:
        """Update version in package.json."""
        if not self.package_json.exists():
            print(f"⊘ Skipped package.json (not found)")
            return

        with open(self.package_json, 'r') as f:
            data = json.load(f)
        
        data['version'] = str(version)
        
        with open(self.package_json, 'w') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        
        print(f"✓ Updated package.json: {version}")

    def update_version_h(self, version: Version) -> None:
        """Update version in C++ Version.h header."""
        if not self.version_h.exists():
            print(f"⊘ Skipped Version.h (not found)")
            return

        content = self.version_h.read_text()
        
        # Update version macros
        content = re.sub(
            r'#define IDAW_VERSION_MAJOR \d+',
            f'#define IDAW_VERSION_MAJOR {version.major}',
            content
        )
        content = re.sub(
            r'#define IDAW_VERSION_MINOR \d+',
            f'#define IDAW_VERSION_MINOR {version.minor}',
            content
        )
        content = re.sub(
            r'#define IDAW_VERSION_PATCH \d+',
            f'#define IDAW_VERSION_PATCH {version.patch}',
            content
        )
        content = re.sub(
            r'#define IDAW_VERSION_STRING "[^"]+"',
            f'#define IDAW_VERSION_STRING "{version}"',
            content
        )
        
        self.version_h.write_text(content)
        print(f"✓ Updated Version.h: {version}")

    def update_all(self, version: Version) -> None:
        """Update version in all project files."""
        self.update_version_file(version)
        self.update_pyproject_toml(version)
        self.update_package_json(version)
        self.update_version_h(version)

    def bump_version(self, bump_type: str, message: Optional[str] = None) -> Version:
        """
        Bump version based on type.
        
        Args:
            bump_type: One of 'major', 'minor', 'patch'
            message: Optional commit message describing the change
        
        Returns:
            New version
        """
        current = self.get_current_version()
        
        if bump_type == 'major':
            new_version = current.bump_major()
        elif bump_type == 'minor':
            new_version = current.bump_minor()
        elif bump_type == 'patch':
            new_version = current.bump_patch()
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
        
        print(f"\n📦 Version bump: {current} → {new_version} ({bump_type})")
        if message:
            print(f"   Message: {message}")
        print()
        
        self.update_all(new_version)
        
        return new_version

    def analyze_changes(self) -> str:
        """
        Analyze git changes to suggest version bump type.
        
        Returns:
            Suggested bump type: 'major', 'minor', or 'patch'
        """
        try:
            # Get git diff
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            changed_files = result.stdout.strip().split('\n')
            
            if not changed_files or changed_files == ['']:
                # Check uncommitted changes
                result = subprocess.run(
                    ['git', 'diff', '--name-only'],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    check=True
                )
                changed_files = result.stdout.strip().split('\n')
            
            # Get commit message if available
            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=%B'],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            commit_msg = result.stdout.strip().lower()
            
            # Analyze changes
            has_breaking = any(
                'breaking' in commit_msg or
                'major' in commit_msg or
                file.startswith('CMakeLists') or
                file.endswith('Version.h')
                for file in changed_files
            )
            
            has_feature = any(
                'feat' in commit_msg or
                'feature' in commit_msg or
                'add' in commit_msg or
                file.endswith('.cpp') or
                file.endswith('.h') or
                (file.endswith('.py') and not file.startswith('test_'))
                for file in changed_files
            )
            
            has_fix = any(
                'fix' in commit_msg or
                'bug' in commit_msg or
                file.startswith('test_')
                for file in changed_files
            )
            
            if has_breaking:
                return 'major'
            elif has_feature:
                return 'minor'
            elif has_fix:
                return 'patch'
            else:
                return 'patch'  # Default to patch
                
        except subprocess.CalledProcessError:
            # If git commands fail, default to patch
            return 'patch'

    def generate_changelog_entry(self, version: Version, message: str) -> str:
        """Generate a changelog entry."""
        date = datetime.now().strftime('%Y-%m-%d')
        return f"""
## [{version}] - {date}

{message}

"""


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Semantic Version Manager for miDiKompanion'
    )
    parser.add_argument(
        'action',
        choices=['bump-major', 'bump-minor', 'bump-patch', 'current', 'auto'],
        help='Action to perform'
    )
    parser.add_argument(
        '-m', '--message',
        help='Commit message describing the change'
    )
    parser.add_argument(
        '--repo-root',
        default='.',
        help='Repository root directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    manager = VersionManager(Path(args.repo_root))
    
    if args.action == 'current':
        version = manager.get_current_version()
        print(f"Current version: {version}")
    elif args.action == 'auto':
        bump_type = manager.analyze_changes()
        print(f"Suggested bump type: {bump_type}")
        new_version = manager.bump_version(bump_type, args.message)
        print(f"\n✓ Version updated to {new_version}")
    else:
        # Extract bump type from action
        bump_type = args.action.replace('bump-', '')
        new_version = manager.bump_version(bump_type, args.message)
        print(f"\n✓ Version updated to {new_version}")


if __name__ == '__main__':
    main()
