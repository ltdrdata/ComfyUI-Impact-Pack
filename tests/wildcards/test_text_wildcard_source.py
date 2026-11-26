import asyncio
import importlib.util
import json
from pathlib import Path
import random
import sys
import types

from aiohttp import web
import pytest

from impact import config, wildcards


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def source(tmp_path, monkeypatch):
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=web.RouteTableDef()))
    monkeypatch.setitem(sys.modules, "server", server)
    spec = importlib.util.spec_from_file_location("review_text_wildcard_source", ROOT / "modules/impact/wildcard_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    custom = tmp_path / "custom_wildcards"
    custom.mkdir()
    monkeypatch.setattr(config, "cached_config", {"custom_wildcards": str(custom)})
    monkeypatch.setattr(wildcards, "wildcard_dict", {"color": ["blue"]})
    monkeypatch.setattr(wildcards, "_on_demand_mode", False)
    return module, custom


def items(module, category):
    response = asyncio.run(module.get_text_items_handler(types.SimpleNamespace(query={"category": category})))
    return json.loads(response.text)


def build(module, **kwargs):
    values = dict(category="scenes", text_wildcard="first", mode="fixed", seed=0)
    return module.TextWildcardSource().build(**(values | kwargs))


def test_every_line_is_selectable_without_lora_syntax(source):
    module, custom = source
    (custom / "scenes.txt").write_text("# comment\nfirst\n\nsecond\nthird, <lora:A:1>\n")
    assert items(module, "scenes") == ["first", "second", "third, <lora:A:1>"]
    assert module.TextWildcardSource.INPUT_TYPES()["required"]["text_wildcard"][0] == items(module, "scenes")


def test_long_and_repeated_lines_get_short_unique_labels(source):
    module, custom = source
    paragraph = "A neon-lit alley at night, " * 5
    (custom / "scenes.txt").write_text(f"{paragraph}\n{paragraph}\n")
    labels = items(module, "scenes")
    assert len(labels) == 2 and labels[0] != labels[1]
    assert len(labels[0]) == 60 and labels[0].endswith("\u2026")
    assert labels[1] == labels[0] + " (1)"
    assert build(module, text_wildcard=labels[1]) == ([paragraph.strip()], [labels[1]])


@pytest.mark.parametrize("mode,seed,expected", [
    ("fixed", 7, ["second"]),
    ("sequential", 0, ["first"]), ("sequential", 1, ["second"]),
    ("sequential", 2, ["third"]), ("sequential", 3, ["first"]),
    ("random", 3, [random.Random(3).choice(["first", "second", "third"])]),
    ("batch", 0, ["first", "second", "third"]),
])
def test_modes(source, mode, seed, expected):
    module, custom = source
    (custom / "scenes.txt").write_text("first\nsecond\nthird\n")
    state = random.getstate()
    assert build(module, mode=mode, seed=seed, text_wildcard="second") == (expected, expected)
    assert random.getstate() == state


def test_wildcard_syntax_inside_a_line_is_processed(source):
    module, custom = source
    (custom / "scenes.txt").write_text("a __color__ door\n")
    assert build(module, text_wildcard="a __color__ door")[0] == ["a blue door"]


def test_missing_fixed_selection_is_not_silently_replaced(source):
    module, custom = source
    (custom / "scenes.txt").write_text("first\n")
    with pytest.raises(ValueError, match="missing"):
        build(module, text_wildcard="missing")


def test_empty_category_outputs_empty_text(source):
    module, custom = source
    (custom / "scenes.txt").write_text("# only comments\n")
    assert build(module) == ([""], ["no_scenes"])


def test_category_cannot_read_outside_configured_directory(source, tmp_path):
    module, _ = source
    (tmp_path / "outside.txt").write_text("secret\n")
    assert items(module, "../outside") == []
    assert build(module, category="../outside") == ([""], ["no_../outside"])
