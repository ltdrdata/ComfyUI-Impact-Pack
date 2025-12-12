#!/usr/bin/env python3
"""
Progressive On-Demand Wildcard Loading Unit Tests

Tests that wildcard loading happens progressively as wildcards are accessed.
"""
import sys
import os
import tempfile

# Add parent directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
impact_pack_dir = os.path.dirname(test_dir)
sys.path.insert(0, impact_pack_dir)

from modules.impact import wildcards


def test_early_termination():
    """Test that calculate_directory_size stops early when limit exceeded"""
    print("=" * 60)
    print("TEST 1: Early Termination Size Calculation")
    print("=" * 60)

    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files totaling 100 bytes
        for i in range(10):
            with open(os.path.join(tmpdir, f"test{i}.txt"), 'w') as f:
                f.write("x" * 10)  # 10 bytes each

        # Test without limit (should scan all)
        total_size = wildcards.calculate_directory_size(tmpdir)
        print(f"✓ Total size without limit: {total_size} bytes")
        assert total_size == 100, f"Expected 100 bytes, got {total_size}"

        # Test with limit (should stop early)
        limited_size = wildcards.calculate_directory_size(tmpdir, limit=50)
        print(f"✓ Size with 50 byte limit: {limited_size} bytes")
        assert limited_size >= 50, f"Expected >= 50 bytes, got {limited_size}"
        assert limited_size <= total_size, "Limited should not exceed total"

        print(f"✓ Early termination working (stopped at {limited_size} bytes)")
        print("\n✅ Early termination test PASSED\n")


def test_metadata_scan():
    """Test that scan_wildcard_metadata only scans file paths, not data"""
    print("=" * 60)
    print("TEST 2: Metadata-Only Scan")
    print("=" * 60)

    # Create temporary wildcard directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file1 = os.path.join(tmpdir, "test1.txt")
        test_file2 = os.path.join(tmpdir, "test2.txt")
        test_yaml = os.path.join(tmpdir, "test3.yaml")

        with open(test_file1, 'w') as f:
            f.write("option1a\noption1b\noption1c\n")

        with open(test_file2, 'w') as f:
            f.write("option2a\noption2b\n")

        with open(test_yaml, 'w') as f:
            f.write("key1:\n  - value1\n  - value2\n")

        # Clear globals
        wildcards.available_wildcards = {}
        wildcards.loaded_wildcards = {}

        # Scan metadata only
        print(f"✓ Scanning directory: {tmpdir}")
        discovered = wildcards.scan_wildcard_metadata(tmpdir)

        print(f"✓ Discovered {discovered} wildcards")
        assert discovered == 3, f"Expected 3 wildcards, got {discovered}"

        print(f"✓ Available wildcards: {list(wildcards.available_wildcards.keys())}")
        assert len(wildcards.available_wildcards) == 3

        # Verify that data is NOT loaded
        assert len(wildcards.loaded_wildcards) == 0, "Data should not be loaded yet"
        print("✓ No data loaded (metadata only)")

        # Verify file paths are stored
        for key in wildcards.available_wildcards.keys():
            file_path = wildcards.available_wildcards[key]
            assert os.path.exists(file_path), f"File path should exist: {file_path}"
            print(f"  - {key} -> {file_path}")

        print("\n✅ Metadata scan test PASSED\n")


def test_progressive_loading():
    """Test that wildcards are loaded progressively on access"""
    print("=" * 60)
    print("TEST 3: Progressive On-Demand Loading")
    print("=" * 60)

    # Create temporary wildcard directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file1 = os.path.join(tmpdir, "wildcard1.txt")
        test_file2 = os.path.join(tmpdir, "wildcard2.txt")
        test_file3 = os.path.join(tmpdir, "wildcard3.txt")

        with open(test_file1, 'w') as f:
            f.write("option1a\noption1b\n")

        with open(test_file2, 'w') as f:
            f.write("option2a\noption2b\n")

        with open(test_file3, 'w') as f:
            f.write("option3a\noption3b\n")

        # Clear globals
        wildcards.available_wildcards = {}
        wildcards.loaded_wildcards = {}
        wildcards._on_demand_mode = True

        # Scan metadata
        discovered = wildcards.scan_wildcard_metadata(tmpdir)
        print(f"✓ Discovered {discovered} wildcards")
        print(f"✓ Available: {len(wildcards.available_wildcards)}")
        print(f"✓ Loaded: {len(wildcards.loaded_wildcards)}")

        # Initial state: 3 available, 0 loaded
        assert len(wildcards.available_wildcards) == 3
        assert len(wildcards.loaded_wildcards) == 0

        # Access first wildcard
        print("\nAccessing wildcard1...")
        data1 = wildcards.get_wildcard_value("wildcard1")
        assert data1 is not None, "Should load wildcard1"
        assert len(data1) == 2, f"Expected 2 options, got {len(data1)}"
        print(f"✓ Loaded wildcard1: {data1}")
        print(f"✓ Loaded count: {len(wildcards.loaded_wildcards)}")
        assert len(wildcards.loaded_wildcards) == 1, "Should have 1 loaded wildcard"

        # Access second wildcard
        print("\nAccessing wildcard2...")
        data2 = wildcards.get_wildcard_value("wildcard2")
        assert data2 is not None, "Should load wildcard2"
        print(f"✓ Loaded wildcard2: {data2}")
        print(f"✓ Loaded count: {len(wildcards.loaded_wildcards)}")
        assert len(wildcards.loaded_wildcards) == 2, "Should have 2 loaded wildcards"

        # Re-access first wildcard (should use cache)
        print("\nRe-accessing wildcard1 (cached)...")
        data1_again = wildcards.get_wildcard_value("wildcard1")
        assert data1_again == data1, "Cached data should match"
        print("✓ Cache hit, data matches")
        print(f"✓ Loaded count: {len(wildcards.loaded_wildcards)}")
        assert len(wildcards.loaded_wildcards) == 2, "Count should not increase on cache hit"

        # Access third wildcard
        print("\nAccessing wildcard3...")
        data3 = wildcards.get_wildcard_value("wildcard3")
        assert data3 is not None, "Should load wildcard3"
        print(f"✓ Loaded wildcard3: {data3}")
        print(f"✓ Loaded count: {len(wildcards.loaded_wildcards)}")
        assert len(wildcards.loaded_wildcards) == 3, "Should have 3 loaded wildcards"

        # Verify all loaded
        assert set(wildcards.loaded_wildcards.keys()) == {"wildcard1", "wildcard2", "wildcard3"}
        print("✓ All wildcards loaded progressively")

        print("\n✅ Progressive loading test PASSED\n")


def test_wildcard_list_functions():
    """Test get_wildcard_list() and get_loaded_wildcard_list()"""
    print("=" * 60)
    print("TEST 4: Wildcard List Functions")
    print("=" * 60)

    # Create temporary wildcard directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        for i in range(5):
            with open(os.path.join(tmpdir, f"test{i}.txt"), 'w') as f:
                f.write(f"option{i}a\noption{i}b\n")

        # Clear globals
        wildcards.available_wildcards = {}
        wildcards.loaded_wildcards = {}
        wildcards._on_demand_mode = True

        # Scan metadata
        wildcards.scan_wildcard_metadata(tmpdir)

        # Test get_wildcard_list (should return all available)
        all_wildcards = wildcards.get_wildcard_list()
        print(f"✓ get_wildcard_list(): {len(all_wildcards)} wildcards")
        assert len(all_wildcards) == 5, "Should return all available wildcards"

        # Test get_loaded_wildcard_list (should return 0 initially)
        loaded_wildcards_list = wildcards.get_loaded_wildcard_list()
        print(f"✓ get_loaded_wildcard_list(): {len(loaded_wildcards_list)} wildcards (initial)")
        assert len(loaded_wildcards_list) == 0, "Should return no loaded wildcards initially"

        # Load some wildcards
        wildcards.get_wildcard_value("test0")
        wildcards.get_wildcard_value("test1")

        # Test get_loaded_wildcard_list (should return 2 now)
        loaded_wildcards_list = wildcards.get_loaded_wildcard_list()
        print(f"✓ get_loaded_wildcard_list(): {len(loaded_wildcards_list)} wildcards (after loading 2)")
        assert len(loaded_wildcards_list) == 2, "Should return 2 loaded wildcards"

        # Verify loaded list is subset of available list
        assert set(loaded_wildcards_list).issubset(set(all_wildcards)), "Loaded should be subset of available"
        print("✓ Loaded list is subset of available list")

        print("\n✅ Wildcard list functions test PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("PROGRESSIVE ON-DEMAND LOADING TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_early_termination()
        test_metadata_scan()
        test_progressive_loading()
        test_wildcard_list_functions()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
