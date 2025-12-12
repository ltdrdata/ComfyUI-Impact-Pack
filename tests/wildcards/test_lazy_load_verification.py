#!/usr/bin/env python3
"""
Verify that wildcard lists are identical before and after on-demand loading.

This test ensures that LazyWildcardLoader maintains consistency:
1. Full cache mode: all data loaded immediately
2. On-demand mode (before access): LazyWildcardLoader proxies
3. On-demand mode (after access): data loaded on demand

All three scenarios should produce identical wildcard lists and values.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.impact import config
from modules.impact.wildcards import wildcard_load, wildcard_dict, is_on_demand_mode, process


def get_wildcard_list():
    """Get list of all wildcard keys"""
    return sorted(list(wildcard_dict.keys()))


def get_wildcard_sample_values(wildcards_to_test=None):
    """Get sample values from specific wildcards"""
    if wildcards_to_test is None:
        wildcards_to_test = [
            'samples/flower',
            'samples/jewel',
            'adnd',  # Depth 3 transitive
            'all',   # Depth 3 transitive
            'colors',  # YAML transitive
        ]

    values = {}
    for key in wildcards_to_test:
        if key in wildcard_dict:
            data = wildcard_dict[key]
            # Convert to list if it's a LazyWildcardLoader
            if hasattr(data, 'get_data'):
                data = data.get_data()
            values[key] = list(data) if data else []
        else:
            values[key] = None

    return values


def test_full_cache_mode():
    """Test with full cache mode (limit = 100 MB)"""
    print("=" * 80)
    print("TEST 1: Full Cache Mode")
    print("=" * 80)
    print()

    # Set high cache limit to force full cache mode
    config.get_config()['wildcard_cache_limit_mb'] = 100

    # Reload wildcards
    wildcard_load()

    # Check mode
    mode = is_on_demand_mode()
    print(f"Mode: {'On-Demand' if mode else 'Full Cache'}")
    assert not mode, "Should be in Full Cache mode"

    # Get wildcard list
    wc_list = get_wildcard_list()
    print(f"Total wildcards: {len(wc_list)}")
    print(f"Sample wildcards: {wc_list[:10]}")
    print()

    # Get sample values
    values = get_wildcard_sample_values()
    print("Sample values:")
    for key, val in values.items():
        if val is not None:
            print(f"  {key}: {len(val)} items - {val[:3] if len(val) >= 3 else val}")
        else:
            print(f"  {key}: NOT FOUND")
    print()

    return {
        'mode': 'full_cache',
        'wildcard_list': wc_list,
        'values': values,
    }


def test_on_demand_mode_before_access():
    """Test with on-demand mode before accessing data"""
    print("=" * 80)
    print("TEST 2: On-Demand Mode (Before Access)")
    print("=" * 80)
    print()

    # Set low cache limit to force on-demand mode
    config.get_config()['wildcard_cache_limit_mb'] = 1

    # Reload wildcards
    wildcard_load()

    # Check mode
    mode = is_on_demand_mode()
    print(f"Mode: {'On-Demand' if mode else 'Full Cache'}")
    assert mode, "Should be in On-Demand mode"

    # Get wildcard list (should work even without loading data)
    wc_list = get_wildcard_list()
    print(f"Total wildcards: {len(wc_list)}")
    print(f"Sample wildcards: {wc_list[:10]}")
    print()

    # Check that wildcards are LazyWildcardLoader instances
    lazy_count = sum(1 for k in wc_list if hasattr(wildcard_dict[k], 'get_data'))
    print(f"LazyWildcardLoader instances: {lazy_count}/{len(wc_list)}")
    print()

    return {
        'mode': 'on_demand_before',
        'wildcard_list': wc_list,
        'lazy_count': lazy_count,
    }


def test_on_demand_mode_after_access():
    """Test with on-demand mode after accessing data"""
    print("=" * 80)
    print("TEST 3: On-Demand Mode (After Access)")
    print("=" * 80)
    print()

    # Mode should still be on-demand from previous test
    mode = is_on_demand_mode()
    print(f"Mode: {'On-Demand' if mode else 'Full Cache'}")
    assert mode, "Should still be in On-Demand mode"

    # Get sample values (this will trigger lazy loading)
    values = get_wildcard_sample_values()
    print("Sample values (after access):")
    for key, val in values.items():
        if val is not None:
            print(f"  {key}: {len(val)} items - {val[:3] if len(val) >= 3 else val}")
        else:
            print(f"  {key}: NOT FOUND")
    print()

    # Test deep transitive wildcards
    print("Testing deep transitive wildcards:")
    test_cases = [
        ("__adnd__", 42),  # Depth 3: adnd → dragon → dragon_spirit
        ("__all__", 123),  # Depth 3: all → giant → giant_soldier
    ]

    for wildcard_text, seed in test_cases:
        result = process(wildcard_text, seed)
        print(f"  {wildcard_text} (seed={seed}): {result}")
    print()

    return {
        'mode': 'on_demand_after',
        'wildcard_list': get_wildcard_list(),
        'values': values,
    }


def compare_results(result1, result2, result3):
    """Compare results from all three tests"""
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print()

    # Compare wildcard lists
    list1 = result1['wildcard_list']
    list2 = result2['wildcard_list']
    list3 = result3['wildcard_list']

    print("1. Wildcard List Comparison")
    print(f"   Full Cache:        {len(list1)} wildcards")
    print(f"   On-Demand (before): {len(list2)} wildcards")
    print(f"   On-Demand (after):  {len(list3)} wildcards")

    if list1 == list2 == list3:
        print("   ✅ All lists are IDENTICAL")
    else:
        print("   ❌ Lists DIFFER")
        if list1 != list2:
            print(f"      Full Cache vs On-Demand (before): {len(set(list1) - set(list2))} differences")
        if list1 != list3:
            print(f"      Full Cache vs On-Demand (after): {len(set(list1) - set(list3))} differences")
        if list2 != list3:
            print(f"      On-Demand (before) vs On-Demand (after): {len(set(list2) - set(list3))} differences")
    print()

    # Compare sample values
    values1 = result1['values']
    values3 = result3['values']

    print("2. Sample Values Comparison")
    all_match = True
    for key in values1.keys():
        v1 = values1[key]
        v3 = values3[key]

        if v1 == v3:
            status = "✅ MATCH"
        else:
            status = "❌ DIFFER"
            all_match = False

        print(f"   {key}: {status}")
        if v1 != v3:
            print(f"      Full Cache: {len(v1) if v1 else 0} items")
            print(f"      On-Demand:  {len(v3) if v3 else 0} items")
    print()

    if all_match:
        print("✅ ALL VALUES MATCH - On-demand loading is CONSISTENT")
    else:
        print("❌ VALUES DIFFER - On-demand loading has ISSUES")
    print()

    return list1 == list2 == list3 and all_match


def main():
    print()
    print("=" * 80)
    print("WILDCARD LAZY LOAD VERIFICATION TEST")
    print("=" * 80)
    print()
    print("This test verifies that on-demand loading produces identical results")
    print("to full cache mode.")
    print()

    # Run tests
    result1 = test_full_cache_mode()
    result2 = test_on_demand_mode_before_access()
    result3 = test_on_demand_mode_after_access()

    # Compare results
    success = compare_results(result1, result2, result3)

    # Final result
    print("=" * 80)
    if success:
        print("🎉 TEST PASSED - Lazy loading is working correctly!")
    else:
        print("❌ TEST FAILED - Lazy loading has consistency issues!")
    print("=" * 80)
    print()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
