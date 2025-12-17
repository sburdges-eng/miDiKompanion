"""
DAW Integration - Logic Pro and other DAW integration utilities.

Provides bridges for working with different DAW file formats,
MIDI export/import, and DAW-specific features.
"""

from midee.daw.logic import (
    LogicProject,
    export_to_logic,
    import_from_logic,
)

__all__ = [
    "LogicProject",
    "export_to_logic",
    "import_from_logic",
]
