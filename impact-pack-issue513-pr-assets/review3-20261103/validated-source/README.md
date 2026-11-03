# Source-pinned focused tests

These files match the code commit in ../commit.json. From this directory, Node 24 with `node --experimental-vm-modules --test tests/*.test.mjs` runs all 62 Node tests. From tests/, `python -m pytest test_wildcard_cache_artifacts.py -v` runs the three POST artifact checks with pytest-timeout, Pillow and Playwright installed; it does not start Chrome or ComfyUI. This bounded source bundle is not a full ComfyUI installation. Exact commands and limits are in ../summary.json and ../review-notes.md.
