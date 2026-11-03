"""Real loader modes, YAML preload, nested TXT access and fresh reloads."""
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
    if on_demand:
        all_keys.add("__palette__")
    before = {"__palette__", "__palette/colors__"} if on_demand else all_keys
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
