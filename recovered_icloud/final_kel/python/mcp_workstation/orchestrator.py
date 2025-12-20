"""
MCP Workstation - Orchestrator

Central coordinator for multi-AI collaboration.
"""

import json
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from .models import (
    WorkstationState,
    AIAgent,
    Proposal,
    ProposalStatus,
    ProposalCategory,
    Phase,
    PhaseStatus,
)
from .proposals import ProposalManager
from .phases import PhaseManager, IDAW_PHASES
from .cpp_planner import CppTransitionPlanner


class Workstation:
    """Central workstation orchestrator."""

    def __init__(self, state_file: Optional[str] = None):
        """Initialize workstation."""
        if state_file is None:
            state_dir = Path.home() / ".mcp_workstation"
            state_dir.mkdir(exist_ok=True)
            state_file = str(state_dir / "state.json")

        self.state_file = state_file
        self.state = self._load_state()
        self.proposal_manager = ProposalManager(self.state.proposals)
        self.phase_manager = PhaseManager(self.state.phases)
        self.cpp_planner = CppTransitionPlanner()

    def _load_state(self) -> WorkstationState:
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Convert dict back to WorkstationState
                    state = WorkstationState()
                    state.registered_agents = [AIAgent(a) for a in data.get('registered_agents', [])]
                    # Load proposals, phases, etc.
                    return state
            except Exception:
                pass
        return WorkstationState()

    def _save_state(self):
        """Save state to file."""
        data = {
            'registered_agents': [a.value for a in self.state.registered_agents],
            'proposals': [self._proposal_to_dict(p) for p in self.state.proposals],
            'phases': [self._phase_to_dict(p) for p in self.state.phases],
            'current_phase_id': self.state.current_phase_id,
            'created_at': self.state.created_at.isoformat(),
            'updated_at': datetime.now().isoformat(),
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
        self.state.updated_at = datetime.now()

    def _proposal_to_dict(self, p: Proposal) -> dict:
        """Convert proposal to dict."""
        return {
            'id': p.id,
            'agent': p.agent.value,
            'title': p.title,
            'description': p.description,
            'category': p.category.value,
            'status': p.status.value,
            'votes': {k.value: v for k, v in p.votes.items()},
            'created_at': p.created_at.isoformat(),
            'updated_at': p.updated_at.isoformat(),
            'implementation_notes': p.implementation_notes,
        }

    def _phase_to_dict(self, p: Phase) -> dict:
        """Convert phase to dict."""
        return {
            'id': p.id,
            'name': p.name,
            'description': p.description,
            'status': p.status.value,
            'tasks': [],  # Simplified
            'started_at': p.started_at.isoformat() if p.started_at else None,
            'completed_at': p.completed_at.isoformat() if p.completed_at else None,
        }

    def register_agent(self, agent: AIAgent):
        """Register an AI agent."""
        if agent not in self.state.registered_agents:
            self.state.registered_agents.append(agent)
            self._save_state()

    def submit_proposal(
        self,
        agent: AIAgent,
        title: str,
        description: str,
        category: ProposalCategory,
    ) -> Proposal:
        """Submit a proposal."""
        proposal = self.proposal_manager.create_proposal(
            agent, title, description, category
        )
        self.state.proposals.append(proposal)
        self._save_state()
        return proposal

    def vote_on_proposal(self, agent: AIAgent, proposal_id: str, vote: int):
        """Vote on a proposal (-1, 0, or 1)."""
        self.proposal_manager.vote(agent, proposal_id, vote)
        self._save_state()

    def get_dashboard(self) -> str:
        """Get dashboard summary."""
        lines = [
            "MCP Workstation Dashboard",
            "=" * 50,
            f"Registered Agents: {len(self.state.registered_agents)}",
            f"Active Proposals: {len([p for p in self.state.proposals if p.status == ProposalStatus.PENDING])}",
            f"Total Proposals: {len(self.state.proposals)}",
            f"Phases: {len(self.state.phases)}",
            "",
        ]
        return "\n".join(lines)

    def get_cpp_progress(self) -> str:
        """Get C++ transition progress."""
        return self.cpp_planner.format_plan()

    def advance_phase(self):
        """Advance to next phase."""
        self.phase_manager.advance()
        self._save_state()


# Singleton instance
_workstation: Optional[Workstation] = None


def get_workstation() -> Workstation:
    """Get singleton workstation instance."""
    global _workstation
    if _workstation is None:
        _workstation = Workstation()
    return _workstation


def shutdown_workstation():
    """Shutdown workstation."""
    global _workstation
    if _workstation is not None:
        _workstation._save_state()
        _workstation = None
