"""
Setup script for Kelly MIDI Companion Python package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="kelly-midi-companion",
    version="2.0.0",
    description="Python interface for Kelly MIDI Companion - emotion-driven MIDI generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kelly Project",
    url="https://github.com/yourusername/kelly-midi-companion",
    packages=find_packages(),
    package_dir={"": "."},
    py_modules=["kelly"],
    python_requires=">=3.8",
    install_requires=[
        "mido>=1.2.10",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "types-torch>=2.0.0",
            "types-numpy>=1.24.0",
        ],
        "training": [
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "tqdm>=4.65.0",
            "types-torch>=2.0.0",
            "types-numpy>=1.24.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Musicians",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
