#!/usr/bin/env python3
"""Find deep transitive wildcard references (5+ levels)"""

import re
from pathlib import Path
from collections import defaultdict

# Auto-detect paths
SCRIPT_DIR = Path(__file__).parent
IMPACT_PACK_DIR = SCRIPT_DIR.parent
WILDCARDS_DIR = IMPACT_PACK_DIR / "wildcards"
CUSTOM_WILDCARDS_DIR = IMPACT_PACK_DIR / "custom_wildcards"

# Build wildcard reference graph
wildcard_refs = defaultdict(set)  # wildcard -> set of wildcards it references
wildcard_files = {}  # wildcard_name -> file_path

def normalize_name(name):
    """Normalize wildcard name"""
    return name.lower().replace('/', '_').replace('\\', '_')

def find_wildcard_file(name):
    """Find wildcard file by name"""
    # Try different variations
    variations = [
        name,
        name.replace('/', '_'),
        name.replace('\\', '_'),
    ]

    for var in variations:
        # Check in wildcards/
        for ext in ['.txt', '.yaml', '.yml']:
            path = WILDCARDS_DIR / f"{var}{ext}"
            if path.exists():
                return str(path)

        # Check in custom_wildcards/
        for ext in ['.txt', '.yaml', '.yml']:
            path = CUSTOM_WILDCARDS_DIR / f"{var}{ext}"
            if path.exists():
                return str(path)

    return None

def scan_wildcards():
    """Scan all wildcard files and build reference graph"""
    print("Scanning wildcard files...")

    # Find all wildcard files
    for base_dir in [WILDCARDS_DIR, CUSTOM_WILDCARDS_DIR]:
        for ext in ['*.txt', '*.yaml', '*.yml']:
            for file_path in base_dir.rglob(ext):
                # Get wildcard name from file path
                rel_path = file_path.relative_to(base_dir)
                name = str(rel_path.with_suffix('')).replace('/', '_').replace('\\', '_')
                wildcard_files[normalize_name(name)] = str(file_path)

                # Find references in file
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    refs = re.findall(r'__([^_]+(?:/[^_]+)*)__', content)

                    for ref in refs:
                        ref_normalized = normalize_name(ref)
                        if ref_normalized and ref_normalized != '':
                            wildcard_refs[normalize_name(name)].add(ref_normalized)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Found {len(wildcard_files)} wildcard files")
    print(f"Found {sum(len(refs) for refs in wildcard_refs.values())} references")
    print()

def find_max_depth(start_wildcard, visited=None, path=None):
    """Find maximum depth of transitive references starting from a wildcard"""
    if visited is None:
        visited = set()
    if path is None:
        path = []

    if start_wildcard in visited:
        return 0, path  # Cycle detected

    visited.add(start_wildcard)
    path.append(start_wildcard)

    refs = wildcard_refs.get(start_wildcard, set())

    if not refs:
        return 1, path  # Leaf node

    max_depth = 0
    max_path = path.copy()

    for ref in refs:
        if ref in wildcard_files:  # Only follow if target exists
            depth, sub_path = find_max_depth(ref, visited.copy(), path.copy())
            if depth > max_depth:
                max_depth = depth
                max_path = sub_path

    return max_depth + 1, max_path

def main():
    scan_wildcards()

    # Find wildcards with references
    wildcards_with_refs = [(name, refs) for name, refs in wildcard_refs.items() if refs]

    print(f"Analyzing {len(wildcards_with_refs)} wildcards with references...")
    print()

    # Calculate depth for each wildcard
    depths = []
    for name, refs in wildcards_with_refs:
        depth, path = find_max_depth(name)
        if depth >= 2:  # At least one level of transitive reference
            depths.append((depth, name, path))

    # Sort by depth (deepest first)
    depths.sort(reverse=True)

    print("=" * 80)
    print("WILDCARD REFERENCE DEPTH ANALYSIS")
    print("=" * 80)
    print()

    # Show top 20 deepest
    print("Top 20 Deepest Transitive References:")
    print()
    for i, (depth, name, path) in enumerate(depths[:20], 1):
        print(f"{i}. Depth {depth}: __{name}__")
        print(f"   Path: {' → '.join(f'__{p}__' for p in path)}")
        if name in wildcard_files:
            print(f"   File: {wildcard_files[name]}")
        print()

    # Find 5+ depth wildcards
    deep_wildcards = [(depth, name, path) for depth, name, path in depths if depth >= 5]

    print()
    print("=" * 80)
    print(f"WILDCARDS WITH 5+ DEPTH ({len(deep_wildcards)} found)")
    print("=" * 80)
    print()

    if deep_wildcards:
        for depth, name, path in deep_wildcards:
            print(f"🎯 __{name}__ (Depth: {depth})")
            print(f"   Chain: {' → '.join(f'__{p}__' for p in path)}")
            if name in wildcard_files:
                print(f"   File: {wildcard_files[name]}")
            print()

        print()
        print("=" * 80)
        print("RECOMMENDED TEST CASE")
        print("=" * 80)
        print()
        depth, name, path = deep_wildcards[0]
        print(f"Use __{name}__ for testing deep transitive loading")
        print(f"Depth: {depth} levels")
        print(f"Chain: {' → '.join(f'__{p}__' for p in path)}")
        print()
        print(f"Test input: \"__{name}__\"")
        print(f"Expected: Will resolve through {depth} levels to actual content")
    else:
        print("No wildcards with 5+ depth found.")
        print()
        if depths:
            depth, name, path = depths[0]
            print(f"Maximum depth found: {depth}")
            print(f"Wildcard: __{name}__")
            print(f"Chain: {' → '.join(f'__{p}__' for p in path)}")

if __name__ == "__main__":
    main()
