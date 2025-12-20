"""
MCP Workstation - Proposal Management

Handles proposal creation, voting, and formatting.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from .models import (
    AIAgent,
    Proposal,
    ProposalStatus,
    ProposalCategory,
)


class ProposalVote:
    """A vote on a proposal."""
    def __init__(self, agent: AIAgent, vote: int):
        self.agent = agent
        self.vote = vote  # -1, 0, or 1


class ProposalManager:
    """Manages proposals."""

    def __init__(self, proposals: List[Proposal]):
        self.proposals = proposals

    def create_proposal(
        self,
        agent: AIAgent,
        title: str,
        description: str,
        category: ProposalCategory,
    ) -> Proposal:
        """Create a new proposal."""
        proposal = Proposal(
            id=str(uuid.uuid4()),
            agent=agent,
            title=title,
            description=description,
            category=category,
            status=ProposalStatus.PENDING,
        )
        return proposal

    def vote(self, agent: AIAgent, proposal_id: str, vote: int):
        """Vote on a proposal."""
        proposal = self._find_proposal(proposal_id)
        if proposal:
            proposal.votes[agent] = vote
            proposal.updated_at = datetime.now()

            # Auto-approve if majority positive
            positive_votes = sum(1 for v in proposal.votes.values() if v > 0)
            total_votes = len(proposal.votes)
            if total_votes >= 2 and positive_votes > total_votes / 2:
                proposal.status = ProposalStatus.APPROVED

    def _find_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Find proposal by ID."""
        for p in self.proposals:
            if p.id == proposal_id:
                return p
        return None


def format_proposal(proposal: Proposal) -> str:
    """Format a proposal for display."""
    lines = [
        f"Proposal: {proposal.title}",
        f"ID: {proposal.id}",
        f"Agent: {proposal.agent.value}",
        f"Category: {proposal.category.value}",
        f"Status: {proposal.status.value}",
        f"Description: {proposal.description}",
        f"Votes: {len(proposal.votes)}",
    ]
    return "\n".join(lines)


def format_proposal_list(proposals: List[Proposal]) -> str:
    """Format a list of proposals."""
    if not proposals:
        return "No proposals."

    lines = ["Proposals:", "=" * 50]
    for p in proposals:
        lines.append(f"\n{format_proposal(p)}")
    return "\n".join(lines)


def get_proposal_template() -> str:
    """Get proposal template."""
    return """
Proposal Template:
------------------
Title: [Brief title]
Description: [Detailed description]
Category: [architecture|feature|bugfix|refactor|documentation|testing|performance]
"""
