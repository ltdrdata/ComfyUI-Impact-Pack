#!/usr/bin/env python3
"""
Final comprehensive wildcard test - validates consistency between full cache and on-demand modes
Tests include:
1. Simple wildcard substitution
2. Nested wildcards (transitive loading)
3. Multiple wildcards in single prompt
4. Dynamic prompts combined with wildcards
5. YAML-based wildcards
"""

import subprocess
import time
import sys
from pathlib import Path

# Auto-detect paths
SCRIPT_DIR = Path(__file__).parent
IMPACT_PACK_DIR = SCRIPT_DIR.parent
COMFYUI_DIR = IMPACT_PACK_DIR.parent.parent
CONFIG_FILE = IMPACT_PACK_DIR / "impact-pack.ini"

def run_test(test_name, cache_limit, test_cases):
    """Run tests with specific cache limit"""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"Cache Limit: {cache_limit} MB")
    print(f"{'='*60}\n")

    # Update config
    config_content = f"""[default]
dependency_version = 24
mmdet_skip = True
sam_editor_cpu = False
sam_editor_model = sam_vit_h_4b8939.pth
custom_wildcards = {IMPACT_PACK_DIR}/custom_wildcards
disable_gpu_opencv = True
wildcard_cache_limit_mb = {cache_limit}
"""

    with open(CONFIG_FILE, 'w') as f:
        f.write(config_content)

    # Start ComfyUI
    print("Starting ComfyUI...")
    proc = subprocess.Popen(
        ['bash', 'run.sh', '--listen', '127.0.0.1', '--port', '8191'],
        cwd=str(COMFYUI_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Wait for server to start
    time.sleep(20)

    # Check logs
    import requests
    try:
        response = requests.get('http://127.0.0.1:8191/')
        print("✓ Server started successfully\n")
    except Exception:
        print("✗ Server failed to start")
        proc.terminate()
        return {}

    # Run test cases
    results = {}
    for i, (description, text, seed) in enumerate(test_cases, 1):
        print(f"Test {i}: {description}")
        print(f"  Input:  {text}")

        try:
            response = requests.post(
                'http://127.0.0.1:8191/impact/wildcards',
                json={'text': text, 'seed': seed},
                timeout=5
            )
            result = response.json()
            output = result.get('text', '')
            print(f"  Output: {output}")
            results[f"test{i}"] = output
        except Exception as e:
            print(f"  Error: {e}")
            results[f"test{i}"] = f"ERROR: {e}"

        print()

    # Stop server
    proc.terminate()
    time.sleep(2)

    return results

def main():
    print("\n" + "="*60)
    print("WILDCARD COMPREHENSIVE CONSISTENCY TEST")
    print("="*60)

    # Test cases: (description, wildcard text, seed)
    test_cases = [
        # Test 1: Simple wildcard
        ("Simple wildcard", "__samples/flower__", 42),

        # Test 2: Multiple wildcards
        ("Multiple wildcards", "a __samples/flower__ and a __samples/jewel__", 123),

        # Test 3: Dynamic prompt
        ("Dynamic prompt", "{red|blue|green} flower", 456),

        # Test 4: Combined wildcard + dynamic
        ("Combined", "{beautiful|elegant} __samples/flower__ with {gold|silver} __samples/jewel__", 789),

        # Test 5: Nested selection (multi-select)
        ("Multi-select", "{2$$, $$__samples/flower__|rose|tulip|daisy}", 111),

        # Test 6: Transitive YAML wildcard (custom_wildcards/test.yaml)
        # __colors__ → __cold__|__warm__ → blue|red|orange|yellow
        ("Transitive YAML wildcard", "__colors__", 222),

        # Test 7: Transitive with text
        ("Transitive with context", "a {beautiful|vibrant} __colors__ flower", 333),
    ]

    # Test with full cache mode
    results_full = run_test("Full Cache Mode", 50, test_cases)

    time.sleep(5)

    # Test with on-demand mode
    results_on_demand = run_test("On-Demand Mode", 1, test_cases)

    # Compare results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60 + "\n")

    all_match = True
    for key in results_full.keys():
        full_result = results_full.get(key, "MISSING")
        on_demand_result = results_on_demand.get(key, "MISSING")

        match = full_result == on_demand_result
        all_match = all_match and match

        status = "✓ MATCH" if match else "✗ DIFFER"
        print(f"{key}: {status}")
        if not match:
            print(f"  Full cache:  {full_result}")
            print(f"  On-demand:   {on_demand_result}")
        print()

    # Final verdict
    print("="*60)
    if all_match:
        print("✅ ALL TESTS PASSED - Results are identical")
        print("="*60)
        return 0
    else:
        print("❌ TESTS FAILED - Results differ between modes")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
