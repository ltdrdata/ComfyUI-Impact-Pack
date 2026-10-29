"""Stub dependencies unused by wildcard parsing; pytest.ini handles import paths."""
import sys
import types

for name in ("folder_paths", "nodes", "impact.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))
