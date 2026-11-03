# Source-pinned focused tests

These files match the published code commit in ../commit.json. From this directory, Node 24 with --experimental-vm-modules --test tests/wildcard_seed_control.test.mjs runs the 31 seed tests; from tests/, Python pytest test_wildcard_cache_artifacts.py runs the 3 artifact tests. The full project Node suite has 60 tests. Complete commands/results and their boundaries are in ../summary.json. No installed models or runtime copies are included.
