#!/usr/bin/env python3
"""
Command Generator for Parallel GPT Codex Instances

Generates MCP tool call commands for 4 parallel GPT Codex instances
to process URLs in parallel.
"""

import json
import sys
from typing import List, Dict, Any


def generate_commands(urls: List[str], instance_id: int, command_type: str = "parse") -> List[Dict[str, Any]]:
    """
    Generate MCP tool calls for a specific instance.
    
    Args:
        urls: List of URLs to process
        instance_id: Instance ID (1-4)
        command_type: "parse" or "download"
    
    Returns:
        List of MCP tool call dictionaries
    """
    # Assign URLs to this instance (0-indexed: instance_id - 1)
    assigned_urls = [urls[i] for i in range(instance_id - 1, len(urls), 4)]
    
    if not assigned_urls:
        return []
    
    commands = []
    
    if command_type == "parse":
        # Use batch parsing for efficiency
        commands.append({
            "tool": "web_parse_batch",
            "arguments": {
                "urls": assigned_urls,
                "save": True
            }
        })
    elif command_type == "download":
        # Use batch download for efficiency
        commands.append({
            "tool": "web_download_batch",
            "arguments": {
                "urls": assigned_urls
            }
        })
    else:
        # Generate individual commands (less efficient but more granular)
        for url in assigned_urls:
            if command_type == "parse":
                commands.append({
                    "tool": "web_parse_url",
                    "arguments": {
                        "url": url,
                        "save": True
                    }
                })
            elif command_type == "download":
                commands.append({
                    "tool": "web_download_file",
                    "arguments": {
                        "url": url
                    }
                })
    
    return commands


def generate_all_instances(urls: List[str], command_type: str = "parse") -> Dict[int, List[Dict[str, Any]]]:
    """
    Generate commands for all 4 instances.
    
    Returns:
        Dictionary mapping instance_id to list of commands
    """
    all_commands = {}
    for instance_id in range(1, 5):
        all_commands[instance_id] = generate_commands(urls, instance_id, command_type)
    return all_commands


def print_commands(commands: Dict[int, List[Dict[str, Any]]], format: str = "json"):
    """Print commands in specified format."""
    if format == "json":
        print(json.dumps(commands, indent=2))
    elif format == "markdown":
        for instance_id, cmds in commands.items():
            print(f"\n## Instance {instance_id} Commands\n")
            for i, cmd in enumerate(cmds, 1):
                print(f"### Command {i}\n")
                print("```json")
                print(json.dumps(cmd, indent=2))
                print("```\n")
    elif format == "summary":
        for instance_id, cmds in commands.items():
            url_count = sum(
                len(cmd["arguments"].get("urls", [cmd["arguments"].get("url", "")]))
                for cmd in cmds
            )
            print(f"Instance {instance_id}: {len(cmds)} command(s), {url_count} URL(s)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate MCP commands for parallel GPT Codex instances"
    )
    parser.add_argument(
        "urls_file",
        type=str,
        help="File containing URLs (one per line) or '-' for stdin"
    )
    parser.add_argument(
        "--instance",
        type=int,
        choices=[1, 2, 3, 4],
        help="Generate commands for specific instance (1-4). If not specified, generates for all instances."
    )
    parser.add_argument(
        "--type",
        choices=["parse", "download"],
        default="parse",
        help="Command type: parse or download (default: parse)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "summary"],
        default="json",
        help="Output format (default: json)"
    )
    
    args = parser.parse_args()
    
    # Read URLs
    if args.urls_file == "-":
        urls = [line.strip() for line in sys.stdin if line.strip()]
    else:
        with open(args.urls_file, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        print("Error: No URLs found", file=sys.stderr)
        sys.exit(1)
    
    # Generate commands
    if args.instance:
        commands = {args.instance: generate_commands(urls, args.instance, args.type)}
    else:
        commands = generate_all_instances(urls, args.type)
    
    # Print results
    print_commands(commands, args.format)


if __name__ == "__main__":
    main()

