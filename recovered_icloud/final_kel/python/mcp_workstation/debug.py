"""
MCP Workstation - Debug Protocol

Debugging and logging utilities.
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DebugCategory(Enum):
    """Debug categories."""
    PROPOSAL = "proposal"
    PHASE = "phase"
    CPP = "cpp"
    AI = "ai"
    SYSTEM = "system"


@dataclass
class DebugEvent:
    """A debug event."""
    timestamp: datetime
    category: DebugCategory
    level: LogLevel
    message: str
    data: Optional[Dict[str, Any]] = None


class DebugProtocol:
    """Debug protocol manager."""

    def __init__(self):
        self.events: List[DebugEvent] = []
        self.logger = logging.getLogger("mcp_workstation")
        self.logger.setLevel(logging.DEBUG)

    def log(
        self,
        category: DebugCategory,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Log a debug event."""
        event = DebugEvent(
            timestamp=datetime.now(),
            category=category,
            level=level,
            message=message,
            data=data,
        )
        self.events.append(event)

        # Also log to Python logger
        log_method = {
            LogLevel.DEBUG: self.logger.debug,
            LogLevel.INFO: self.logger.info,
            LogLevel.WARNING: self.logger.warning,
            LogLevel.ERROR: self.logger.error,
        }.get(level, self.logger.info)

        log_method(f"[{category.value}] {message}", extra=data or {})

    def get_events(
        self,
        category: Optional[DebugCategory] = None,
        level: Optional[LogLevel] = None,
    ) -> List[DebugEvent]:
        """Get filtered events."""
        events = self.events
        if category:
            events = [e for e in events if e.category == category]
        if level:
            events = [e for e in events if e.level == level]
        return events


# Global debug instance
_debug: Optional[DebugProtocol] = None


def get_debug() -> DebugProtocol:
    """Get global debug instance."""
    global _debug
    if _debug is None:
        _debug = DebugProtocol()
    return _debug


def trace(func):
    """Decorator to trace function calls."""
    def wrapper(*args, **kwargs):
        debug = get_debug()
        debug.log(
            DebugCategory.SYSTEM,
            LogLevel.DEBUG,
            f"Calling {func.__name__}",
            {"args": str(args), "kwargs": str(kwargs)},
        )
        return func(*args, **kwargs)
    return wrapper


def log_debug(message: str, data: Optional[Dict[str, Any]] = None):
    """Log debug message."""
    get_debug().log(DebugCategory.SYSTEM, LogLevel.DEBUG, message, data)


def log_info(message: str, data: Optional[Dict[str, Any]] = None):
    """Log info message."""
    get_debug().log(DebugCategory.SYSTEM, LogLevel.INFO, message, data)


def log_warning(message: str, data: Optional[Dict[str, Any]] = None):
    """Log warning message."""
    get_debug().log(DebugCategory.SYSTEM, LogLevel.WARNING, message, data)


def log_error(message: str, data: Optional[Dict[str, Any]] = None):
    """Log error message."""
    get_debug().log(DebugCategory.SYSTEM, LogLevel.ERROR, message, data)
