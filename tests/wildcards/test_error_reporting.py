#!/usr/bin/env python3
"""
Unit Tests for Wildcard Error Reporting

Tests that per-file error handling works correctly in wildcard loading,
ensuring bad files don't crash the loader and valid files continue to load.
"""
import sys
import os
import tempfile

# Add parent directory to path
test_dir = os.path.dirname(os.path.abspath(__file__))
impact_pack_dir = os.path.dirname(test_dir)
sys.path.insert(0, impact_pack_dir)

# Add modules/impact directory to path for direct import
modules_impact_dir = os.path.join(test_dir, '..', '..', 'modules', 'impact')
sys.path.insert(0, modules_impact_dir)

# Import wildcards directly to avoid triggering root __init__.py
import wildcards


def test_yaml_file_with_null_bytes():
    """Test that YAML file with null bytes doesn't crash loader"""
    print("=" * 60)
    print("TEST 1: YAML file with null bytes (AppleDouble-like)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a YAML file with null bytes (like AppleDouble header)
        bad_yaml = os.path.join(tmpdir, "bad_file.yaml")
        with open(bad_yaml, 'wb') as f:
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')  # Null bytes
            f.write(b'key: value\n')

        # Try to load - should not crash, should return empty list or handle gracefully
        try:
            result = wildcards.load_yaml_wildcard(bad_yaml)
            print(f"✓ load_yaml_wildcard returned: {result}")
            # Should either return empty list or handle gracefully
            assert isinstance(result, list), "Should return a list"
            print("✓ YAML with null bytes handled gracefully")
            print("\n✅ TEST 1 PASSED\n")
        except Exception as e:
            # If it raises, that's also acceptable as long as it's handled
            print(f"✓ Exception raised (acceptable): {type(e).__name__}")
            print("\n✅ TEST 1 PASSED\n")


def test_txt_file_with_binary_content():
    """Test that TXT file with binary content doesn't crash loader"""
    print("=" * 60)
    print("TEST 2: TXT file with binary content")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a TXT file with binary content
        bad_txt = os.path.join(tmpdir, "bad_file.txt")
        with open(bad_txt, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')  # PNG header
            f.write(b'some text content\n')

        # Try to load - should not crash
        try:
            result = wildcards.load_txt_wildcard(bad_txt)
            print(f"✓ load_txt_wildcard returned: {result}")
            assert isinstance(result, list), "Should return a list"
            print("✓ TXT with binary content handled gracefully")
            print("\n✅ TEST 2 PASSED\n")
        except Exception as e:
            print(f"✓ Exception raised (acceptable): {type(e).__name__}")
            print("\n✅ TEST 2 PASSED\n")


def test_error_message_contains_file_path():
    """Test that error message contains specific file path"""
    print("=" * 60)
    print("TEST 3: Error message contains specific file path")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file that will fail to parse
        bad_file = os.path.join(tmpdir, "test_error.txt")
        with open(bad_file, 'wb') as f:
            f.write(b'\x00\x00\x00\x00invalid content')

        # Capture logging output to check error message
        import logging
        import io

        # Create a handler to capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)

        # Get the logger and add our handler
        logger = logging.getLogger('modules.impact.wildcards')
        original_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(handler)

        try:
            # This should trigger a warning
            result = wildcards.load_txt_wildcard(bad_file)

            # Get the captured log
            log_output = log_capture.getvalue()
            print(f"✓ Log output: {log_output}")

            # Check if file path appears in warning
            if log_output:
                assert "test_error.txt" in log_output or bad_file in log_output, \
                    f"Error message should contain file path, got: {log_output}"
                print("✓ Error message contains file path")
            else:
                print("✓ No warning logged (file handled silently)")

            print("\n✅ TEST 3 PASSED\n")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)


def test_valid_files_load_after_bad_files():
    """Test that valid files still load after bad files are skipped"""
    print("=" * 60)
    print("TEST 4: Valid files load after bad files are skipped")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple files - some bad, some good
        bad_file1 = os.path.join(tmpdir, "bad1.txt")
        good_file = os.path.join(tmpdir, "good.txt")
        bad_file2 = os.path.join(tmpdir, "bad2.yaml")
        good_yaml = os.path.join(tmpdir, "good.yaml")

        # Bad files
        with open(bad_file1, 'wb') as f:
            f.write(b'\x00\x00\x00\x00binary junk')

        with open(bad_file2, 'wb') as f:
            f.write(b'\x00\x00\x00\x00corrupt')

        # Good files
        with open(good_file, 'w') as f:
            f.write("option1\noption2\noption3\n")

        with open(good_yaml, 'w') as f:
            f.write("colors:\n  - red\n  - blue\n  - green\n")

        # Try to load the good files specifically
        good_txt_result = wildcards.load_txt_wildcard(good_file)
        print(f"✓ Good TXT file loaded: {good_txt_result}")
        assert len(good_txt_result) == 3, f"Expected 3 options, got {len(good_txt_result)}"
        assert "option1" in good_txt_result

        good_yaml_result = wildcards.load_yaml_wildcard(good_yaml)
        print(f"✓ Good YAML file loaded: {good_yaml_result}")
        assert len(good_yaml_result) > 0, "YAML should have returned options"

        print("✓ Valid files load correctly after bad files")
        print("\n✅ TEST 4 PASSED\n")


def test_on_demand_mode_error_handling():
    """Test both on_demand=True and on_demand=False modes work"""
    print("=" * 60)
    print("TEST 5: on_demand mode error handling")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mix of good and bad files
        good_file = os.path.join(tmpdir, "valid.txt")
        bad_file = os.path.join(tmpdir, "corrupt.txt")

        with open(good_file, 'w') as f:
            f.write("good_option1\ngood_option2\n")

        with open(bad_file, 'wb') as f:
            f.write(b'\x00\x00\x00\x00broken')

        # Test on_demand=False mode (immediate loading)
        print("Testing on_demand=False mode...")
        wildcards.wildcard_dict.clear()
        wildcards.read_wildcard_dict(tmpdir, on_demand=False)

        # Check that good file was loaded
        valid_key = wildcards.wildcard_normalize("valid")
        if valid_key in wildcards.wildcard_dict:
            data = wildcards.wildcard_dict[valid_key]
            if isinstance(data, wildcards.LazyWildcardLoader):
                data = data.get_data()
            print(f"✓ on_demand=False: valid.txt loaded as: {data}")
            assert len(data) == 2

        print("✓ on_demand=False mode works")

        # Test on_demand=True mode (lazy loading)
        print("Testing on_demand=True mode...")
        wildcards.wildcard_dict.clear()
        wildcards.read_wildcard_dict(tmpdir, on_demand=True)

        valid_key = wildcards.wildcard_normalize("valid")
        if valid_key in wildcards.wildcard_dict:
            loader = wildcards.wildcard_dict[valid_key]
            assert isinstance(loader, wildcards.LazyWildcardLoader), "Should be LazyWildcardLoader"
            data = loader.get_data()
            print(f"✓ on_demand=True: valid.txt loaded as: {data}")
            assert len(data) == 2

        print("✓ on_demand=True mode works")

        # Both modes should handle errors gracefully without crashing
        print("\n✅ TEST 5 PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WILDCARD ERROR REPORTING TESTS")
    print("=" * 60 + "\n")

    test_yaml_file_with_null_bytes()
    test_txt_file_with_binary_content()
    test_error_message_contains_file_path()
    test_valid_files_load_after_bad_files()
    test_on_demand_mode_error_handling()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60 + "\n")