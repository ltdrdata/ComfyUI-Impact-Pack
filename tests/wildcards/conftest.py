"""Stub dependencies unused by wildcard parsing; pytest.ini handles import paths."""
import sys
import types
from unittest.mock import Mock


for name in ("folder_paths", "nodes", "impact.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))

graph_utils = types.ModuleType("comfy_execution.graph_utils")
graph_utils.is_link = Mock(side_effect=AssertionError("Run cache tests with real ComfyUI using integration_wildcard_cache.py"))
sys.modules.setdefault("comfy_execution.graph_utils", graph_utils)
