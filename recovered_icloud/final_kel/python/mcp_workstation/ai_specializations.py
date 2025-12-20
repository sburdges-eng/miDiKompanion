"""
MCP Workstation - AI Specializations

Defines AI capabilities and task assignment logic.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

from .models import AIAgent, ProposalCategory


class TaskType(Enum):
    """Task types."""
    ARCHITECTURE = "architecture"
    CODING = "coding"
    DEBUGGING = "debugging"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    RESEARCH = "research"
    REFACTORING = "refactoring"


@dataclass
class AICapabilities:
    """AI agent capabilities."""
    agent: AIAgent
    strengths: List[TaskType]
    weaknesses: List[TaskType]
    description: str


# AI Capabilities Matrix
AI_CAPABILITIES: Dict[AIAgent, AICapabilities] = {
    AIAgent.CLAUDE: AICapabilities(
        agent=AIAgent.CLAUDE,
        strengths=[
            TaskType.ARCHITECTURE,
            TaskType.DEBUGGING,
            TaskType.REFACTORING,
        ],
        weaknesses=[TaskType.RESEARCH],
        description="Strong in code architecture, real-time safety, complex debugging",
    ),
    AIAgent.CHATGPT: AICapabilities(
        agent=AIAgent.CHATGPT,
        strengths=[
            TaskType.RESEARCH,
            TaskType.DOCUMENTATION,
            TaskType.TESTING,
        ],
        weaknesses=[TaskType.ARCHITECTURE],
        description="Strong in theory analysis, explanations, documentation",
    ),
    AIAgent.GEMINI: AICapabilities(
        agent=AIAgent.GEMINI,
        strengths=[
            TaskType.RESEARCH,
            TaskType.ARCHITECTURE,
        ],
        weaknesses=[TaskType.DEBUGGING],
        description="Strong in cross-language patterns, multi-modal analysis",
    ),
    AIAgent.GITHUB_COPILOT: AICapabilities(
        agent=AIAgent.GITHUB_COPILOT,
        strengths=[
            TaskType.CODING,
            TaskType.REFACTORING,
        ],
        weaknesses=[TaskType.ARCHITECTURE, TaskType.DOCUMENTATION],
        description="Strong in code completion, boilerplate generation",
    ),
}


def get_capabilities(agent: AIAgent) -> AICapabilities:
    """Get capabilities for an agent."""
    return AI_CAPABILITIES[agent]


def get_best_agent_for_task(task_type: TaskType) -> Optional[AIAgent]:
    """Get best agent for a task type."""
    best_agent = None
    best_score = -1

    for agent, caps in AI_CAPABILITIES.items():
        if task_type in caps.strengths:
            score = len(caps.strengths)
            if score > best_score:
                best_score = score
                best_agent = agent

    return best_agent


def get_best_agents_for_task(task_type: TaskType, limit: int = 3) -> List[AIAgent]:
    """Get best agents for a task type."""
    agents_with_scores = []

    for agent, caps in AI_CAPABILITIES.items():
        if task_type in caps.strengths:
            score = len(caps.strengths)
            agents_with_scores.append((agent, score))

    agents_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [agent for agent, _ in agents_with_scores[:limit]]


def get_agents_for_category(category: ProposalCategory) -> List[AIAgent]:
    """Get suitable agents for a proposal category."""
    category_to_task = {
        ProposalCategory.ARCHITECTURE: TaskType.ARCHITECTURE,
        ProposalCategory.FEATURE: TaskType.CODING,
        ProposalCategory.BUGFIX: TaskType.DEBUGGING,
        ProposalCategory.REFACTOR: TaskType.REFACTORING,
        ProposalCategory.DOCUMENTATION: TaskType.DOCUMENTATION,
        ProposalCategory.TESTING: TaskType.TESTING,
        ProposalCategory.PERFORMANCE: TaskType.REFACTORING,
    }

    task_type = category_to_task.get(category, TaskType.CODING)
    return get_best_agents_for_task(task_type)


def suggest_task_assignment(task_type: TaskType) -> Dict[AIAgent, str]:
    """Suggest task assignment with rationale."""
    assignments = {}

    for agent, caps in AI_CAPABILITIES.items():
        if task_type in caps.strengths:
            assignments[agent] = f"Strong in {task_type.value}"
        elif task_type in caps.weaknesses:
            assignments[agent] = f"Weak in {task_type.value}"
        else:
            assignments[agent] = "Neutral"

    return assignments


def get_collaboration_strategy(task_type: TaskType) -> str:
    """Get collaboration strategy for a task."""
    primary = get_best_agent_for_task(task_type)
    if primary:
        return f"Primary: {primary.value}, with review from others"
    return "Distribute across all agents"


def print_ai_summary():
    """Print AI capabilities summary."""
    print("AI Agent Capabilities")
    print("=" * 50)
    for agent, caps in AI_CAPABILITIES.items():
        print(f"\n{agent.value.upper()}")
        print(f"  Description: {caps.description}")
        print(f"  Strengths: {[t.value for t in caps.strengths]}")
        print(f"  Weaknesses: {[t.value for t in caps.weaknesses]}")


def get_task_assignment_summary() -> str:
    """Get task assignment summary."""
    lines = ["Task Assignment Guide", "=" * 50]

    for task_type in TaskType:
        best = get_best_agent_for_task(task_type)
        lines.append(f"\n{task_type.value}: {best.value if best else 'Any'}")

    return "\n".join(lines)
