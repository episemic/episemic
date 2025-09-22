#!/usr/bin/env python3
"""
Script to generate API documentation using pydoc.

Usage:
    python generate_docs.py
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {cmd}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Generate all documentation."""
    print("🧠 Generating Episemic Core Documentation")
    print("=" * 50)

    # Ensure we're in the right directory
    if not Path("episemic_core").exists():
        print("❌ Error: episemic_core directory not found")
        print("Please run this script from the project root directory")
        sys.exit(1)

    # Create docs directory
    docs_dir = Path("docs/api")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # List of modules to document
    modules = [
        # User-facing API
        "episemic",
        "episemic.simple",

        # Internal API
        "episemic.api",
        "episemic.config",
        "episemic.models",

        # Core components
        "episemic.hippocampus.hippocampus",
        "episemic.cortex.cortex",
        "episemic.consolidation.consolidation",
        "episemic.retrieval.retrieval",

        # CLI
        "episemic.cli.main",
    ]

    # Generate documentation for each module
    success_count = 0
    for module in modules:
        cmd = f"python -m pydoc -w {module}"
        if run_command(cmd):
            success_count += 1

    # Move generated HTML files to docs directory
    print("\n📁 Moving files to docs/api/...")
    html_files = list(Path(".").glob("*.html"))

    if html_files:
        for html_file in html_files:
            target = docs_dir / html_file.name
            html_file.rename(target)
            print(f"   Moved {html_file} -> {target}")
    else:
        print("   No HTML files found to move")

    print(f"\n🎉 Documentation generation complete!")
    print(f"   Generated: {success_count}/{len(modules)} modules")
    print(f"   Location: docs/api/")
    print(f"   Index: docs/index.html")

    # Check if index.html exists
    index_file = Path("docs/index.html")
    if index_file.exists():
        print(f"\n🌐 Open docs/index.html in your browser to view the documentation")
    else:
        print(f"\n📝 Note: Create docs/index.html for a documentation landing page")


if __name__ == "__main__":
    main()