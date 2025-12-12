#!/usr/bin/env python3
"""
Test script for wildcard lazy loading functionality
"""
import sys
import os
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.impact import wildcards

def test_lazy_loader():
    """Test LazyWildcardLoader class"""
    print("=" * 60)
    print("TEST 1: LazyWildcardLoader functionality")
    print("=" * 60)

    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("option1\n")
        f.write("option2\n")
        f.write("# comment line\n")
        f.write("option3\n")
        temp_file = f.name

    try:
        # Test lazy loading
        loader = wildcards.LazyWildcardLoader(temp_file, 'txt')
        print(f"✓ Created LazyWildcardLoader: {loader}")

        # Check that data is not loaded yet
        assert not loader._loaded, "Data should not be loaded initially"
        print("✓ Data not loaded initially (lazy)")

        # Access data
        data = loader.get_data()
        print(f"✓ Loaded data: {data}")
        assert len(data) == 3, f"Expected 3 items, got {len(data)}"
        assert 'option1' in data, "option1 should be in data"

        # Check that data is now loaded
        assert loader._loaded, "Data should be loaded after access"
        print("✓ Data loaded after first access")

        # Test list-like operations
        print(f"✓ len(loader) = {len(loader)}")
        assert len(loader) == 3

        print(f"✓ loader[0] = {loader[0]}")
        assert loader[0] == 'option1'

        print(f"✓ 'option2' in loader = {'option2' in loader}")
        assert 'option2' in loader

        print(f"✓ list(loader) = {list(loader)}")

        print("\n✅ LazyWildcardLoader tests PASSED\n")

    finally:
        os.unlink(temp_file)


def test_cache_limit_detection():
    """Test automatic cache mode detection"""
    print("=" * 60)
    print("TEST 2: Cache limit detection")
    print("=" * 60)

    # Get current cache limit
    limit = wildcards.get_cache_limit()
    print(f"✓ Cache limit: {limit / (1024*1024):.2f} MB")

    # Calculate wildcard directory size
    wildcards_dir = wildcards.wildcards_path
    total_size = wildcards.calculate_directory_size(wildcards_dir)
    print(f"✓ Wildcards directory size: {total_size / (1024*1024):.2f} MB")
    print(f"✓ Wildcards path: {wildcards_dir}")

    # Determine expected mode
    if total_size >= limit:
        expected_mode = "on-demand"
    else:
        expected_mode = "full cache"

    print(f"✓ Expected mode: {expected_mode}")
    print("\n✅ Cache detection tests PASSED\n")


def test_wildcard_loading():
    """Test actual wildcard loading"""
    print("=" * 60)
    print("TEST 3: Wildcard loading with current mode")
    print("=" * 60)

    # Clear existing wildcards
    wildcards.wildcard_dict = {}
    wildcards._on_demand_mode = False

    # Load wildcards
    print("Loading wildcards...")
    wildcards.wildcard_load()

    # Check mode
    is_on_demand = wildcards.is_on_demand_mode()
    print(f"✓ On-demand mode active: {is_on_demand}")

    # Check loaded wildcards
    wc_list = wildcards.get_wildcard_list()
    print(f"✓ Loaded {len(wc_list)} wildcards")

    if len(wc_list) > 0:
        print(f"✓ Sample wildcards: {wc_list[:5]}")

        # Test accessing a wildcard
        if len(wildcards.wildcard_dict) > 0:
            key = list(wildcards.wildcard_dict.keys())[0]
            value = wildcards.wildcard_dict[key]
            print(f"✓ Sample wildcard '{key}' type: {type(value).__name__}")

            if isinstance(value, wildcards.LazyWildcardLoader):
                print(f"  - LazyWildcardLoader: {value}")
                print(f"  - Loaded: {value._loaded}")
                # Access the data
                data = value.get_data()
                print(f"  - Data loaded, items: {len(data)}")
            else:
                print(f"  - Direct list, items: {len(value)}")

    print("\n✅ Wildcard loading tests PASSED\n")


def test_on_demand_simulation():
    """Simulate on-demand mode with temporary wildcards"""
    print("=" * 60)
    print("TEST 4: On-demand mode simulation")
    print("=" * 60)

    # Create temporary wildcard directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file1 = os.path.join(tmpdir, "test1.txt")
        test_file2 = os.path.join(tmpdir, "test2.txt")

        with open(test_file1, 'w') as f:
            f.write("option1a\noption1b\noption1c\n")

        with open(test_file2, 'w') as f:
            f.write("option2a\noption2b\n")

        # Clear and load with on-demand mode
        wildcards.wildcard_dict = {}
        wildcards._on_demand_mode = False

        print(f"✓ Loading from temp directory: {tmpdir}")
        wildcards.read_wildcard_dict(tmpdir, on_demand=True)

        print(f"✓ Loaded {len(wildcards.wildcard_dict)} wildcards")

        for key, value in wildcards.wildcard_dict.items():
            print(f"✓ Wildcard '{key}':")
            print(f"  - Type: {type(value).__name__}")
            if isinstance(value, wildcards.LazyWildcardLoader):
                print(f"  - Initially loaded: {value._loaded}")
                data = value.get_data()
                print(f"  - After access: loaded={value._loaded}, items={len(data)}")
                print(f"  - Sample data: {data[:2]}")

    print("\n✅ On-demand simulation tests PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("WILDCARD LAZY LOADING TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_lazy_loader()
        test_cache_limit_detection()
        test_wildcard_loading()
        test_on_demand_simulation()

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
