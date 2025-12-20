"""
MCP Workstation - MCP Server Implementation

MCP server for workstation tools.
"""

from typing import Any, Dict, List, Optional
import json

from .orchestrator import get_workstation
from .models import AIAgent, ProposalCategory
from .ai_specializations import get_best_agent_for_task, TaskType


def get_mcp_tools() -> List[Dict[str, Any]]:
    """Get MCP tools for workstation."""
    return [
        {
            "name": "workstation_status",
            "description": "Get workstation status and dashboard",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "register_agent",
            "description": "Register an AI agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": ["claude", "chatgpt", "gemini", "github_copilot"],
                    },
                },
                "required": ["agent"],
            },
        },
        {
            "name": "submit_proposal",
            "description": "Submit a proposal",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["agent", "title", "description", "category"],
            },
        },
        {
            "name": "vote_proposal",
            "description": "Vote on a proposal",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string"},
                    "proposal_id": {"type": "string"},
                    "vote": {"type": "integer", "minimum": -1, "maximum": 1},
                },
                "required": ["agent", "proposal_id", "vote"],
            },
        },
        {
            "name": "get_phases",
            "description": "Get development phases",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "get_cpp_plan",
            "description": "Get C++ transition plan",
            "inputSchema": {
                "type": "object",
                "properties": {},
            },
        },
    ]


def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP tool call."""
    ws = get_workstation()

    if tool_name == "workstation_status":
        return {"result": ws.get_dashboard()}

    elif tool_name == "register_agent":
        agent = AIAgent(arguments["agent"])
        ws.register_agent(agent)
        return {"result": f"Registered {agent.value}"}

    elif tool_name == "submit_proposal":
        agent = AIAgent(arguments["agent"])
        category = ProposalCategory(arguments["category"])
        proposal = ws.submit_proposal(
            agent,
            arguments["title"],
            arguments["description"],
            category,
        )
        return {"result": f"Created proposal {proposal.id}"}

    elif tool_name == "vote_proposal":
        agent = AIAgent(arguments["agent"])
        ws.vote_on_proposal(agent, arguments["proposal_id"], arguments["vote"])
        return {"result": "Vote recorded"}

    elif tool_name == "get_phases":
        from .phases import format_phase_progress
        return {"result": format_phase_progress(ws.state.phases)}

    elif tool_name == "get_cpp_plan":
        return {"result": ws.get_cpp_progress()}

    else:
        return {"error": f"Unknown tool: {tool_name}"}


def run_server():
    """Run MCP server (stdio mode)."""
    import sys
    import json

    # Simple stdio MCP server
    while True:
        line = sys.stdin.readline()
        if not line:
            break

        try:
            request = json.loads(line)
            method = request.get("method")

            if method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {"tools": get_mcp_tools()},
                }
            elif method == "tools/call":
                tool_name = request["params"]["name"]
                arguments = request["params"].get("arguments", {})
                result = handle_tool_call(tool_name, arguments)
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": result,
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {"code": -32601, "message": "Method not found"},
                }

            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if 'request' in locals() else None,
                "error": {"code": -32603, "message": str(e)},
            }
            print(json.dumps(error_response))
            sys.stdout.flush()
