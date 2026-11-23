import logging

import pytest

from impact import wildcards


@pytest.mark.parametrize("on_demand", [False, True], ids=["full-cache", "on-demand"])
def test_loading_modes_and_reload(wildcard_loader, on_demand):
    (wildcard_loader / "colors.txt").write_text("red\ngreen\nblue\n")
    (wildcard_loader / "alias.txt").write_text("__colors__\n")
    (wildcard_loader / "palette.yaml").write_text("palette:\n  colors: [red, green, blue]\n")
    wildcards.config.get_config()["wildcard_cache_limit_mb"] = 0 if on_demand else 50
    wildcards.wildcard_load()
    assert wildcards.is_on_demand_mode() is on_demand
    all_keys = {"__colors__", "__alias__", "__palette/colors__"}
    before = {"__palette/colors__"} if on_demand else all_keys
    assert set(wildcards.get_wildcard_list()) == set(wildcards.get_loaded_wildcard_list()) == before

    for seed, expected in ((2, "red"), (5, "blue")):
        assert wildcards.process("__palette/colors__", seed) == expected
    assert set(wildcards.get_loaded_wildcard_list()) == before

    for seed, expected in ((2, "red"), (5, "blue")):
        assert wildcards.process("__alias__", seed) == expected
    assert set(wildcards.get_wildcard_list()) == set(wildcards.get_loaded_wildcard_list()) == all_keys

    (wildcard_loader / "colors.txt").write_text("yellow\n")
    assert wildcards.process("__colors__", 2) == "red"
    wildcards.wildcard_load()
    assert wildcards.is_on_demand_mode() is on_demand
    assert set(wildcards.get_loaded_wildcard_list()) == before
    assert wildcards.process("__alias__", 2) == "yellow"
    assert set(wildcards.get_wildcard_list()) == all_keys


NESTED = """\
color: [red]
animal:
  cat: [tabby]
outfit:
  top:
    casual: [tee]
    formal: [shirt]
place:
  europe:
    france:
      city: [paris]
mood:
  calm:
    deep: still
"""


@pytest.fixture(params=[50, 0], ids=["full cache", "on demand"])
def nested_yaml(request, wildcard_loader):
    (wildcard_loader / "place.yaml").write_text(NESTED)
    (wildcard_loader / "tree.txt").write_text("oak\n")
    wildcards.config.cached_config["wildcard_cache_limit_mb"] = request.param
    wildcards.wildcard_load()
    assert wildcards.is_on_demand_mode() == (request.param == 0)


@pytest.mark.parametrize("prompt, expected", [
    ("__color__", "red"),
    ("__animal/cat__", "tabby"),
    ("__outfit/top/casual__", "tee"),
    ("__place/europe/france/city__", "paris"),
    ("__mood/calm/deep__", "still"),
    ("__place__", "paris"),
    ("__*/casual__", "tee"),
    ("__outfit/top/c*__", "tee"),
    ("{1$$__*/formal__}", "shirt"),
    ("__tree__", "oak"),
])
def test_yaml_wildcard_resolves_in_both_loading_modes(nested_yaml, caplog, prompt, expected):
    assert wildcards.process(prompt, seed=0) == expected
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_nested_yaml_wildcards_are_listed_in_both_loading_modes(nested_yaml):
    assert {"__color__", "__animal/cat__", "__outfit/top/casual__", "__place/europe/france/city__"} <= set(wildcards.get_wildcard_list())
