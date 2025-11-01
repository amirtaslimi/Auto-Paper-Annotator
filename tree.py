#!/usr/bin/env python3
"""
ASCII Tree Generator
--------------------
Usage:
    python ascii_tree.py [path] [--dirs-only] [--max-depth N]
Example:
    python ascii_tree.py . --max-depth 3
"""

import os
import argparse

def build_tree(root_path: str, prefix: str = "", dirs_only=False, max_depth=None, level=0):
    """Recursively print ASCII tree structure."""
    if max_depth is not None and level >= max_depth:
        return

    entries = sorted(os.listdir(root_path))
    if dirs_only:
        entries = [e for e in entries if os.path.isdir(os.path.join(root_path, e))]

    entries_count = len(entries)
    for i, entry in enumerate(entries):
        path = os.path.join(root_path, entry)
        connector = "└── " if i == entries_count - 1 else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "    " if i == entries_count - 1 else "│   "
            build_tree(path, prefix + extension, dirs_only, max_depth, level + 1)


def main():
    parser = argparse.ArgumentParser(description="Generate ASCII tree of directory structure.")
    parser.add_argument("path", nargs="?", default=".", help="Root path of directory tree.")
    parser.add_argument("--dirs-only", action="store_true", help="Show directories only.")
    parser.add_argument("--max-depth", type=int, help="Limit depth of recursion.")
    args = parser.parse_args()

    print(args.path)
    build_tree(args.path, dirs_only=args.dirs_only, max_depth=args.max_depth)

if __name__ == "__main__":
    main()
