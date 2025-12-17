"""
Groove extraction and application module.

Extract timing/velocity patterns from MIDI files and apply them to other tracks.
"""

from midee.groove.extractor import extract_groove, GrooveTemplate
from midee.groove.applicator import apply_groove
from midee.groove.templates import get_genre_template, GENRE_TEMPLATES

__all__ = [
    "extract_groove",
    "apply_groove",
    "GrooveTemplate",
    "get_genre_template",
    "GENRE_TEMPLATES",
]
