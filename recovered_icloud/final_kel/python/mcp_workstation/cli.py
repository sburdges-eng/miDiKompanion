"""
MCP Workstation - CLI

Command-line interface for workstation.
"""

import argparse
import sys
from typing import Optional

from .orchestrator import get_workstation, shutdown_workstation
from .models import AIAgent, ProposalCategory
from .proposals import format_proposal_list
from .phases import format_phase_progress
from .ai_specializations import print_ai_summary, get_task_assignment_summary


def cmd_status(args):
    """Status command."""
    ws = get_workstation()
    print(ws.get_dashboard())


def cmd_register(args):
    """Register agent command."""
    ws = get_workstation()
    try:
        agent = AIAgent(args.agent.lower())
        ws.register_agent(agent)
        print(f"Registered {agent.value}")
    except ValueError:
        print(f"Invalid agent: {args.agent}")
        sys.exit(1)


def cmd_propose(args):
    """Propose command."""
    ws = get_workstation()
    try:
        agent = AIAgent(args.agent.lower())
        category = ProposalCategory(args.category.lower())
        proposal = ws.submit_proposal(
            agent,
            args.title,
            args.description,
            category,
        )
        print(f"Created proposal: {proposal.id}")
        print(f"Title: {proposal.title}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_vote(args):
    """Vote command."""
    ws = get_workstation()
    try:
        agent = AIAgent(args.agent.lower())
        vote = int(args.vote)
        if vote not in [-1, 0, 1]:
            print("Vote must be -1, 0, or 1")
            sys.exit(1)
        ws.vote_on_proposal(agent, args.proposal_id, vote)
        print(f"Voted {vote} on proposal {args.proposal_id}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_phases(args):
    """Phases command."""
    ws = get_workstation()
    print(format_phase_progress(ws.state.phases))


def cmd_cpp(args):
    """C++ plan command."""
    ws = get_workstation()
    print(ws.get_cpp_progress())


def cmd_ai(args):
    """AI specializations command."""
    print_ai_summary()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MCP Workstation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Status
    subparsers.add_parser("status", help="Show workstation status")

    # Register
    register_parser = subparsers.add_parser("register", help="Register an AI agent")
    register_parser.add_argument("agent", help="Agent name (claude, chatgpt, gemini, github_copilot)")

    # Propose
    propose_parser = subparsers.add_parser("propose", help="Submit a proposal")
    propose_parser.add_argument("agent", help="Agent name")
    propose_parser.add_argument("title", help="Proposal title")
    propose_parser.add_argument("description", help="Proposal description")
    propose_parser.add_argument("category", help="Category (architecture, feature, bugfix, etc.)")

    # Vote
    vote_parser = subparsers.add_parser("vote", help="Vote on a proposal")
    vote_parser.add_argument("agent", help="Agent name")
    vote_parser.add_argument("proposal_id", help="Proposal ID")
    vote_parser.add_argument("vote", help="Vote (-1, 0, or 1)")

    # Phases
    subparsers.add_parser("phases", help="Show phase progress")

    # C++
    subparsers.add_parser("cpp", help="Show C++ transition plan")

    # AI
    subparsers.add_parser("ai", help="Show AI specializations")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "status":
            cmd_status(args)
        elif args.command == "register":
            cmd_register(args)
        elif args.command == "propose":
            cmd_propose(args)
        elif args.command == "vote":
            cmd_vote(args)
        elif args.command == "phases":
            cmd_phases(args)
        elif args.command == "cpp":
            cmd_cpp(args)
        elif args.command == "ai":
            cmd_ai(args)
    finally:
        shutdown_workstation()


if __name__ == "__main__":
    main()
