"""Stub dependencies unused by wildcard parsing; pytest.ini handles import paths."""
import sys
import types
from unittest.mock import Mock

import pytest

for name in ("folder_paths", "nodes", "impact.utils"):
    sys.modules.setdefault(name, types.ModuleType(name))

graph_utils = types.ModuleType("comfy_execution.graph_utils")
graph_utils.is_link = Mock(side_effect=AssertionError("Run cache tests with real ComfyUI using integration_wildcard_cache.py"))
sys.modules.setdefault("comfy_execution.graph_utils", graph_utils)


from impact import wildcards


@pytest.fixture
def wildcard_loader(tmp_path, monkeypatch):
    for name in ("wildcard_dict", "available_wildcards", "loaded_wildcards"):
        monkeypatch.setattr(wildcards, name, {})
    monkeypatch.setattr(wildcards, "_on_demand_mode", False)
    monkeypatch.setattr(wildcards, "wildcards_path", str(tmp_path))
    monkeypatch.setattr(wildcards.config, "cached_config", {
        "custom_wildcards": "", "wildcard_cache_limit_mb": 50,
    })
    return tmp_path
