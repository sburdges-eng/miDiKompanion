"""
MCP Workstation - Data Models

Core data structures for multi-AI collaboration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class AIAgent(Enum):
    """AI agent types."""
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    GEMINI = "gemini"
    GITHUB_COPILOT = "github_copilot"


class ProposalStatus(Enum):
    """Proposal status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class ProposalCategory(Enum):
    """Proposal categories."""
    ARCHITECTURE = "architecture"
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    PERFORMANCE = "performance"


class PhaseStatus(Enum):
    """Phase status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Proposal:
    """A proposal from an AI agent."""
    id: str
    agent: AIAgent
    title: str
    description: str
    category: ProposalCategory
    status: ProposalStatus = ProposalStatus.PENDING
    votes: Dict[AIAgent, int] = field(default_factory=dict)  # -1, 0, or 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    implementation_notes: Optional[str] = None


@dataclass
class PhaseTask:
    """A task within a phase."""
    id: str
    title: str
    description: str
    assigned_to: Optional[AIAgent] = None
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class Phase:
    """A development phase."""
    id: str
    name: str
    description: str
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    tasks: List[PhaseTask] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class WorkstationState:
    """Complete workstation state."""
    registered_agents: List[AIAgent] = field(default_factory=list)
    proposals: List[Proposal] = field(default_factory=list)
    phases: List[Phase] = field(default_factory=list)
    current_phase_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
