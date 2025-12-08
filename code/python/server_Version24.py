import os
import logging
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import FastMCP, tool

# === 1. ENVIRONMENT SETUP ===

# Load .env variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("penta-core")

# === 2. CLIENT INITIALIZATION ===

import openai
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
xai_client = openai.OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")
import anthropic
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
import httpx

GOOGLE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
GITHUB_API_URL = "https://api.github.com"

# === 3. MCP SERVER (FastMCP) INITIALIZATION ===

mcp_server = FastMCP(
    name="Penta-Core",
    description="A Model Context Protocol (MCP) server aggregating OpenAI (GPT-4o), Anthropic (Claude 3.5 Sonnet), Google Gemini 1.5 Pro, xAI Grok, and GitHub into a single Swarm toolset for IDE integration."
)

# === 4. SWARM TOOL DEFINITIONS ===

@tool("consult_architect")
def consult_architect(prompt: str) -> str:
    """Tool A: High-level architecture planning via OpenAI (GPT-4o)"""
    system_prompt = (
        "You are a Systems Architect. Focus on high-level logic, class structure, and design patterns."
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI Architect error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI Architect tool failed: {str(e)}")

@tool("consult_developer")
def consult_developer(prompt: str) -> str:
    """Tool B: Senior engineering/code review via Anthropic Claude 3.5 Sonnet"""
    system_prompt = (
        "You are a Senior Engineer. Focus on clean code, safety, and refactoring. Output production-ready code."
    )
    try:
        completion = anthropic_client.messages.create(
            model="claude-3.5-sonnet-20240606",
            max_tokens=1024,
            temperature=0.6,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.content[0].text
    except Exception as e:
        logger.error(f"Anthropic Developer error: {e}")
        raise HTTPException(status_code=500, detail=f"Anthropic Developer tool failed: {str(e)}")

@tool("consult_researcher")
def consult_researcher(prompt: str, context: str) -> str:
    """Tool C: Deep documentation/research via Google Gemini 1.5 Pro"""
    system_prompt = (
        "You are a Lead Researcher. Analyze documentation and find edge cases with your massive context window."
    )
    headers = {"Content-Type": "application/json"}
    params = {"key": GOOGLE_API_KEY}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"SYSTEM: {system_prompt}\n\nPrompt: {prompt}\n\nContext: {context}"}
                ]
            }
        ]
    }
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(GOOGLE_API_URL, headers=headers, params=params, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"Google Researcher error: {e}")
        raise HTTPException(status_code=500, detail=f"Google Researcher tool failed: {str(e)}")

@tool("consult_maverick")
def consult_maverick(prompt: str) -> str:
    """Tool D: xAI Grok Beta review via xAI API"""
    system_prompt = (
        "You are a Maverick Engineer. Criticize the plan, find non-obvious flaws, and suggest lateral/unconventional solutions. Be direct."
    )
    try:
        response = xai_client.chat.completions.create(
            model="grok-1.5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"xAI Maverick error: {e}")
        raise HTTPException(status_code=500, detail=f"xAI Maverick tool failed: {str(e)}")

@tool("fetch_repo_context")
def fetch_repo_context(owner: str, repo: str, path: str) -> Any:
    """Tool E: Fetch file contents or directory tree from a GitHub repo."""
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"GitHub fetch_repo_context error: {e}")
        raise HTTPException(status_code=500, detail=f"GitHub repo context retrieval failed: {str(e)}")

# === 5. REGISTER TOOLS ===

mcp_server.register_tools(
    consult_architect,
    consult_developer,
    consult_researcher,
    consult_maverick,
    fetch_repo_context,
)

# === 6. FASTAPI INTEGRATION & LAUNCH ===

# FastAPI app for doc serving, custom endpoints, and MCP ASGI integration
app = FastAPI(
    title="Penta-Core MCP Server",
    description="Aggregates OpenAI, Anthropic, Google, xAI, GitHub into Swarm for IDEs.",
)

# Mount the MCP server as a router under /mcp/api
app.mount("/mcp/api", mcp_server.as_fastapi())

# Optionally, add a healthcheck endpoint, and root welcome
@app.get("/")
def root():
    return {
        "service": "Penta-Core MCP Server",
        "status": "running",
        "docs": "/docs",
        "mcp_api": "/mcp/api"
    }

@app.get("/healthz")
def healthz():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)