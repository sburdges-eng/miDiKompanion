#!/usr/bin/env python3
"""
CLI for MCP Web Parser

Quick command-line interface for testing web parsing functionality.
"""

import argparse
import json
from pathlib import Path
from typing import List

from .server import (
    ParallelParser,
    DownloadManager,
    MetadataManager,
    PARSED_DATA_DIR,
    DOWNLOAD_DIR,
)


def parse_urls(urls: List[str], save: bool = True):
    """Parse URLs from command line."""
    parser = ParallelParser()
    parsed_pages = parser.parse_urls(urls)
    
    metadata = MetadataManager()
    
    print(f"Parsing {len(urls)} URLs...")
    for parsed in parsed_pages:
        if save:
            output_file = parsed.save(PARSED_DATA_DIR)
            metadata.add_parsed_page(parsed)
            print(f"✓ Parsed: {parsed.url} → {output_file}")
        else:
            print(f"✓ Parsed: {parsed.url} ({len(parsed.content)} chars)")
    
    print(f"\nSuccessfully parsed {len(parsed_pages)}/{len(urls)} URLs")


def download_files(urls: List[str]):
    """Download files from URLs."""
    downloader = DownloadManager()
    metadata = MetadataManager()
    
    print(f"Downloading {len(urls)} files...")
    paths = downloader.download_parallel(urls)
    
    for path in paths:
        metadata.add_download(urls[paths.index(path)], path)
        print(f"✓ Downloaded: {path}")
    
    print(f"\nSuccessfully downloaded {len(paths)}/{len(urls)} files")


def show_statistics():
    """Show parsing/download statistics."""
    metadata = MetadataManager()
    stats = metadata.metadata["statistics"]
    
    print("Web Parser Statistics")
    print("=" * 50)
    print(f"Total Parsed: {stats['total_parsed']}")
    print(f"Total Downloaded: {stats['total_downloaded']}")
    print(f"Last Updated: {stats.get('last_updated', 'Never')}")
    print(f"\nData Directory: {PARSED_DATA_DIR.parent}")
    print(f"Parsed Directory: {PARSED_DATA_DIR}")
    print(f"Download Directory: {DOWNLOAD_DIR}")


def list_parsed(limit: int = 100):
    """List parsed pages."""
    metadata = MetadataManager()
    pages = metadata.metadata["parsed_pages"][-limit:]
    
    print(f"Parsed Pages (showing {len(pages)} of {len(metadata.metadata['parsed_pages'])}):")
    print("=" * 50)
    for page in pages:
        print(f"  {page['title']}")
        print(f"    URL: {page['url']}")
        print(f"    Hash: {page['url_hash']}")
        print(f"    Timestamp: {page['timestamp']}")
        print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MCP Web Parser CLI - Parallel web parsing and download tool"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse URLs")
    parse_parser.add_argument("urls", nargs="+", help="URLs to parse")
    parse_parser.add_argument("--no-save", action="store_true", help="Don't save to disk")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download files")
    download_parser.add_argument("urls", nargs="+", help="URLs to download")
    
    # Statistics command
    subparsers.add_parser("stats", help="Show statistics")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List parsed pages")
    list_parser.add_argument("--limit", type=int, default=100, help="Maximum results")
    
    args = parser.parse_args()
    
    if args.command == "parse":
        parse_urls(args.urls, save=not args.no_save)
    elif args.command == "download":
        download_files(args.urls)
    elif args.command == "stats":
        show_statistics()
    elif args.command == "list":
        list_parsed(args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

