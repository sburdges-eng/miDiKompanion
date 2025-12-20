"""
MCP Workstation - Phase Management

Manages development phases for iDAW.
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

from .models import Phase, PhaseStatus, PhaseTask, AIAgent


@dataclass
class PhaseDefinition:
    """Phase definition."""
    id: str
    name: str
    description: str
    tasks: List[str]  # Task descriptions


# iDAW Development Phases
IDAW_PHASES = [
    PhaseDefinition(
        id="phase_1",
        name="Foundation",
        description="Core infrastructure and basic features",
        tasks=[
            "Set up project structure",
            "Implement basic audio engine",
            "Create UI framework",
        ],
    ),
    PhaseDefinition(
        id="phase_2",
        name="Music Intelligence",
        description="AI-powered music generation",
        tasks=[
            "Integrate emotion mapping",
            "Implement intent processing",
            "Add rule-breaking engine",
        ],
    ),
    PhaseDefinition(
        id="phase_3",
        name="C++ Migration",
        description="Port Python to C++ for real-time",
        tasks=[
            "Port audio engine",
            "Port MIDI generation",
            "Optimize for RT-safety",
        ],
    ),
    PhaseDefinition(
        id="phase_4",
        name="Integration",
        description="Full system integration",
        tasks=[
            "Connect all components",
            "End-to-end testing",
            "Performance optimization",
        ],
    ),
]


class PhaseManager:
    """Manages development phases."""

    def __init__(self, phases: List[Phase]):
        self.phases = phases
        if not self.phases:
            self._initialize_phases()

    def _initialize_phases(self):
        """Initialize phases from definitions."""
        for phase_def in IDAW_PHASES:
            phase = Phase(
                id=phase_def.id,
                name=phase_def.name,
                description=phase_def.description,
                status=PhaseStatus.NOT_STARTED,
            )
            # Add tasks
            for i, task_desc in enumerate(phase_def.tasks):
                task = PhaseTask(
                    id=f"{phase_def.id}_task_{i}",
                    title=task_desc,
                    description=task_desc,
                )
                phase.tasks.append(task)
            self.phases.append(phase)

    def advance(self):
        """Advance to next phase."""
        # Find current phase
        current = None
        for phase in self.phases:
            if phase.status == PhaseStatus.IN_PROGRESS:
                current = phase
                break

        if current:
            # Mark current as completed
            current.status = PhaseStatus.COMPLETED
            current.completed_at = datetime.now()

        # Start next phase
        for phase in self.phases:
            if phase.status == PhaseStatus.NOT_STARTED:
                phase.status = PhaseStatus.IN_PROGRESS
                phase.started_at = datetime.now()
                break


def format_phase_progress(phases: List[Phase]) -> str:
    """Format phase progress."""
    lines = ["Development Phases:", "=" * 50]
    for phase in phases:
        status_icon = {
            PhaseStatus.NOT_STARTED: "○",
            PhaseStatus.IN_PROGRESS: "◐",
            PhaseStatus.COMPLETED: "●",
            PhaseStatus.BLOCKED: "⚠",
        }.get(phase.status, "?")

        lines.append(f"\n{status_icon} {phase.name}")
        lines.append(f"   {phase.description}")
        lines.append(f"   Status: {phase.status.value}")
        lines.append(f"   Tasks: {len(phase.tasks)}")
    return "\n".join(lines)


def get_next_actions(phases: List[Phase]) -> List[str]:
    """Get next actions from current phase."""
    for phase in phases:
        if phase.status == PhaseStatus.IN_PROGRESS:
            return [task.title for task in phase.tasks if task.status == PhaseStatus.NOT_STARTED]
    return []
