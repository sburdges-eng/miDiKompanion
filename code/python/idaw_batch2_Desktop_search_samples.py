#!/usr/bin/env python3
"""
Sample Search Tool
Query the sample catalog database
"""

import json
import argparse
from pathlib import Path

SAMPLES_DIR = Path.home() / "Music" / "Samples"
CATALOG_FILE = SAMPLES_DIR / "sample_catalog.json"


def load_catalog():
    """Load sample catalog"""
    if not CATALOG_FILE.exists():
        print(f"❌ Catalog not found: {CATALOG_FILE}")
        print("   Run sample_cataloger.py first")
        return None

    with open(CATALOG_FILE) as f:
        return json.load(f)


def search(catalog, **filters):
    """Search samples with filters"""
    results = catalog["samples"]

    # Apply filters
    if filters.get("type"):
        results = [s for s in results if s["type"].lower() == filters["type"].lower()]

    if filters.get("bpm"):
        results = [s for s in results if s["bpm"] == filters["bpm"]]

    if filters.get("key"):
        results = [s for s in results if s["key"] and filters["key"].lower() in s["key"].lower()]

    if filters.get("category"):
        results = [s for s in results if filters["category"].lower() in s["category"].lower()]

    if filters.get("description"):
        results = [s for s in results if filters["description"].lower() in s["description"].lower()]

    return results


def main():
    """CLI search interface"""
    parser = argparse.ArgumentParser(description="Search sample catalog")
    parser.add_argument("--type", help="Sample type (e.g., Kick, Snare)")
    parser.add_argument("--bpm", help="BPM (e.g., 120)")
    parser.add_argument("--key", help="Musical key (e.g., Dmin, Cmaj)")
    parser.add_argument("--category", help="Category (e.g., Drums, Bass)")
    parser.add_argument("--description", help="Description keyword")
    parser.add_argument("--count", type=int, default=20, help="Max results to show")

    args = parser.parse_args()

    # Load catalog
    catalog = load_catalog()
    if not catalog:
        return

    # Build filter dict
    filters = {k: v for k, v in vars(args).items() if v and k != "count"}

    if not filters:
        print("❌ Please specify at least one filter")
        print("\nExamples:")
        print("  python3 search_samples.py --type Kick")
        print("  python3 search_samples.py --bpm 120 --key Dmin")
        print("  python3 search_samples.py --category Drums")
        return

    # Search
    print(f"🔍 Searching {catalog['total_samples']} samples...\n")

    results = search(catalog, **filters)

    if not results:
        print("❌ No results found")
        return

    print(f"✅ Found {len(results)} results")
    print(f"   Showing first {min(len(results), args.count)}:\n")

    for i, sample in enumerate(results[:args.count], 1):
        print(f"{i}. {sample['filename']}")
        print(f"   Category: {sample['category']}")
        print(f"   BPM: {sample['bpm']} | Key: {sample['key']}")
        print(f"   Path: {sample['relative_path']}")
        print()


if __name__ == "__main__":
    main()
