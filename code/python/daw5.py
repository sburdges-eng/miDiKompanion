# init_project.py
# Script to initialize the DAiW-Music-Brain project structure

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the directory structure for DAiW-Music-Brain."""
    base_dir = Path("music_brain")
    
    # Main package directories
    dirs = [
        "music_brain",
        "music_brain/data",
        "music_brain/groove",
        "music_brain/structure",
        "music_brain/session",
        "music_brain/audio",
        "music_brain/utils",
        "tools/audio_cataloger",
        "vault/Songwriting_Guides",
        "vault/Theory_Reference",
        "vault/Songs/when-i-found-you-sleeping",
        "examples/midi"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # Create __init__.py in package dirs
        if dir_path.startswith("music_brain"):
            init_file = Path(dir_path) / "__init__.py"
            if not init_file.exists():
                init_file.touch()
    
    # Create key files with basic content
    files_content = {
        "music_brain/__init__.py": "# Public API exports\n",
        "music_brain/cli.py": "# CLI entry point (daiw command)\n",
        "music_brain/data/emotional_mapping.py": "# Emotion → musical parameters\n",
        "music_brain/data/chord_progression_families.json": "[]\n",
        "music_brain/data/music_vernacular_database.md": "# Casual language → technical\n",
        "music_brain/session/intent_schema.py": "# CompleteSongIntent, rule-breaking enums\n",
        "music_brain/groove/templates.py": "# Genre groove definitions\n",
        "music_brain/structure/progression.py": "# Chord parsing and diagnosis\n",
        "pyproject.toml": "[tool.black]\nline-length = 100\n\n[tool.flake8]\nmax-line-length = 100\n",
        "README.md": "# DAiW-Music-Brain\nA Python toolkit for emotionally-driven music composition.\n"
    }
    
    for file_path, content in files_content.items():
        file = Path(file_path)
        if not file.exists():
            file.write_text(content)
    
    print("Project structure initialized.")

if __name__ == "__main__":
    create_directory_structure()