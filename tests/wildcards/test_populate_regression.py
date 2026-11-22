from __future__ import annotations

import logging
import os
import sys
import tempfile

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

from impact import wildcards as wc  # noqa: E402
from populate_helpers import (  # noqa: E402
    assert_determinism,
    assert_distribution,
    assert_membership,
    install_wildcards,
    populate,
)

SEEDS = [0, 1, 42, 100, 999, 12345, 2_147_483_647]


@pytest.fixture
def wildcard_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(wc, "wildcards_path", str(tmp_path))
    monkeypatch.setattr(wc, "wildcard_dict", {})
    monkeypatch.setattr(wc, "available_wildcards", {})
    monkeypatch.setattr(wc, "loaded_wildcards", {})
    monkeypatch.setattr(wc, "_on_demand_mode", True)
    monkeypatch.setattr(wc.config, "get_config", dict)
    return tmp_path


@pytest.mark.parametrize("name", ["my colors.txt", "colors.TXT.txt"])
@pytest.mark.parametrize("short_name,prefix", [(False, ""), (False, "collection/"), (True, "collection/")])
def test_on_demand_expands_indexed_normalized_paths(wildcard_directory, name, prefix, short_name):
    path = wildcard_directory / prefix / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("red\n", encoding="utf-8")
    wc.scan_wildcard_metadata(str(wildcard_directory))
    key = wc.wildcard_normalize(os.path.splitext(name if short_name else prefix + name)[0])

    assert wc.process(f"__{key}__", 42) == "red"


@pytest.mark.parametrize("key", ["missing", "collection/missing"])
def test_on_demand_missing_indexed_file_does_not_recurse(wildcard_directory, key):
    wc.available_wildcards[key] = str(wildcard_directory / "removed.txt")

    assert wc.get_wildcard_value(key) is None


@pytest.mark.parametrize("on_demand", [False, True])
def test_loading_preserves_direct_file_precedence(wildcard_directory, monkeypatch, on_demand):
    custom = wildcard_directory / "custom"
    custom.mkdir()
    (wildcard_directory / "shared.txt").write_text("builtin\n", encoding="utf-8")
    (custom / "shared.txt").write_text("custom\n", encoding="utf-8")
    monkeypatch.setattr(wc.config, "get_config", lambda: {"custom_wildcards": str(custom)})
    monkeypatch.setattr(wc, "get_cache_limit", lambda: 1 if on_demand else 1024 * 1024)

    wc.wildcard_load()

    assert wc.process("__shared__", 42) == ("builtin" if on_demand else "custom")


def test_on_demand_depth_match_keeps_all_available_options(wildcard_directory):
    for directory, option in (("first", "red"), ("second/nested", "blue")):
        path = wildcard_directory / directory / "my colors.txt"
        path.parent.mkdir(parents=True)
        path.write_text(option + "\n", encoding="utf-8")
    wc.scan_wildcard_metadata(str(wildcard_directory))
    wc.available_wildcards["removed/my-colors"] = str(wildcard_directory / "removed.txt")

    assert sorted(wc.get_wildcard_value("my-colors")) == ["blue", "red"]


@pytest.mark.parametrize("on_demand", [False, True])
def test_wildcard_scans_ignore_git_metadata(wildcard_directory, monkeypatch, caplog, on_demand):
    for name in (".git", ".github"):
        metadata = wildcard_directory / "collection" / name / "workflows"
        metadata.mkdir(parents=True)
        (metadata / "publish.yml").write_text("on: [push]\n", encoding="utf-8")
        (metadata / "internal.txt").write_text("not a wildcard\n", encoding="utf-8")
    visible = wildcard_directory / ".styles"
    visible.mkdir()
    (visible / "colors.txt").write_text("red\n", encoding="utf-8")
    (wildcard_directory / "colors.yaml").write_text("palette: [blue]\n", encoding="utf-8")
    expected_size = sum(p.stat().st_size for p in
                        (visible / "colors.txt", wildcard_directory / "colors.yaml"))
    monkeypatch.setattr(wc, "get_cache_limit", lambda: 1 if on_demand else 1024 * 1024)

    wc.wildcard_load()

    assert wc.calculate_directory_size(str(wildcard_directory)) == expected_size
    assert wc.process("__.styles/colors__ __palette__", 42) == "red blue"
    keys = set(wc.available_wildcards) | set(wc.wildcard_dict) | set(wc.loaded_wildcards)
    assert not any("/.git/" in k or "/.github/" in k for k in keys)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


_FLOWERS = ["rose", "orchid", "iris", "carnation", "lily"]


def test_c1_1_membership_exact_and_pool_filter():
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    ) as f:
        f.write("rose\norchid\n# not selected\niris\ncarnation\n\nlily\n")
        path = f.name
    try:
        loaded = wc.load_txt_wildcard(path)
        assert set(loaded) == set(_FLOWERS), (
            f"load_txt_wildcard filter mismatch: {loaded!r}"
        )
        assert "# not selected" not in loaded
        assert "" not in loaded

        install_wildcards({"flowers": loaded})
        expected = {f"A {x} B" for x in _FLOWERS}
        for seed in SEEDS:
            out = assert_determinism("A __flowers__ B", seed)
            assert_membership(out, expected)
    finally:
        os.unlink(path)


def test_c1_2_prefix_suffix_byte_preservation():
    install_wildcards({"flowers": _FLOWERS})
    text = "🌸 __flowers__ !@#\n"
    for seed in SEEDS:
        out = populate(text, seed)
        assert out.startswith("🌸 "), f"prefix lost: {out!r}"
        assert out.endswith(" !@#\n"), f"suffix lost: {out!r}"
        middle = out[len("🌸 "): -len(" !@#\n")]
        assert middle in set(_FLOWERS), f"middle {middle!r} not a flower"


def test_c2_1_exact_membership():
    install_wildcards({})
    expected = {"X red Y", "X blue Y", "X green Y"}
    for seed in SEEDS:
        out = assert_determinism("X {red|blue|green} Y", seed)
        assert_membership(out, expected)


def test_c2_2_whitespace_in_options_preserved():
    install_wildcards({})
    expected = {" red apple ", " blue sky "}
    for seed in SEEDS:
        out = assert_determinism("{ red apple | blue sky }", seed)
        assert_membership(out, expected)


def test_c2_3_empty_option_current_behavior():
    install_wildcards({})
    allowed = {"a", "", "b"}
    for seed in SEEDS:
        out = assert_determinism("{a||b}", seed)
        assert out in allowed, (
            f"baseline violation: {out!r} not in {allowed!r} — spec gap may have shifted"
        )


def test_c3_2_two_level_nesting():
    install_wildcards({})
    expected = {"a", "b", "x", "y", "d", "c"}
    for seed in SEEDS:
        out = assert_determinism("{a|{b|{x|y}|d}|c}", seed)
        assert_membership(out, expected)


def test_c3_3_nested_distribution_current_behavior():
    install_wildcards({})

    def get(i):
        return populate("{a|{d|e|f}|c}", i)

    weights = {"a": 3, "c": 3, "d": 1, "e": 1, "f": 1}
    assert_distribution(get, weights, n=1000, sigma=3.0)


def test_c4_2_tutorial_l29_example():
    install_wildcards({
        "safruit": ["apple", "cherry"],
        "flower": ["rose", "iris"],
        "vehicle": ["car", "bike"],
    })
    text = (
        "1{girl is holding {blue pencil|red __safruit__|colorful __flower__}"
        "|boy is riding __vehicle__}"
    )
    expected = {
        "1girl is holding blue pencil",
        "1girl is holding red apple",
        "1girl is holding red cherry",
        "1girl is holding colorful rose",
        "1girl is holding colorful iris",
        "1boy is riding car",
        "1boy is riding bike",
    }
    for seed in SEEDS:
        out = assert_determinism(text, seed)
        assert_membership(out, expected)


def test_c5_2_weighted_distribution_normalized():
    install_wildcards({})

    def get(i):
        return populate("{5::red|4::green|7::blue|black}", i)

    assert_distribution(
        get,
        weights={"red": 5, "green": 4, "blue": 7, "black": 1},
        n=1000,
        sigma=3.0,
    )


def test_c5_3_default_weight_one():
    install_wildcards({})

    def get(i):
        return populate("{5::red|green|blue}", i)

    assert_distribution(
        get,
        weights={"red": 5, "green": 1, "blue": 1},
        n=1000,
        sigma=3.0,
    )


def test_c5_4_wrong_syntax_rejected_as_equal_weights():
    install_wildcards({})

    def get(i):
        return populate("{common::10|rare::1}", i)

    assert_distribution(
        get,
        weights={"common::10": 1, "rare::1": 1},
        n=1000,
        sigma=3.0,
    )


def test_c5_5_decimal_weight_current_behavior():
    install_wildcards({})
    allowed = {"a b", "b a"}
    for seed in SEEDS:
        out = assert_determinism("{0.5::a|0.3::b}", seed)
        assert out in allowed, (
            f"baseline violation: {out!r} — decimal-weight parsing drift"
        )


def test_c5_6_simple_decimal_acceptance():
    install_wildcards({})
    for seed in SEEDS:
        out = assert_determinism("{0.5::a|b}", seed)
        assert_membership(out, {"a", "b"})


def test_c6_1_case_insensitive_equivalence():
    install_wildcards({"jewel": ["diamond", "ruby", "sapphire"]})
    for seed in SEEDS:
        lower = populate("__jewel__", seed)
        upper = populate("__JEWEL__", seed)
        title = populate("__Jewel__", seed)
        mixed = populate("__jEwEl__", seed)
        assert lower == upper == title == mixed, (
            f"case-insensitive failed seed={seed}: "
            f"lower={lower!r} upper={upper!r} title={title!r} mixed={mixed!r}"
        )


def test_c6_2_filename_case_normalize():
    install_wildcards({})
    wc.read_wildcard("Jewel", ["diamond", "ruby", "sapphire"])
    assert "Jewel" not in wc.wildcard_dict
    assert "jewel" in wc.wildcard_dict
    for seed in SEEDS:
        out = assert_determinism("__jewel__", seed)
        assert_membership(out, {"diamond", "ruby", "sapphire"})


def test_c7_1_aggregation_union_membership():
    install_wildcards({
        "garden/flower": ["lily"],
        "ocean/flower": ["coral"],
        "meadow/flower": ["daisy", "poppy"],
    })
    expected = {"lily", "coral", "daisy", "poppy"}
    for seed in SEEDS:
        out = assert_determinism("__*/flower__", seed)
        assert_membership(out, expected)


def test_c7_2_aggregation_variance():
    install_wildcards({
        "garden/flower": ["lily"],
        "ocean/flower": ["coral"],
        "meadow/flower": ["daisy", "poppy"],
    })
    seen = {populate("__*/flower__", s) for s in range(20)}
    assert len(seen) >= 3, f"aggregation variance degenerate: {seen!r}"


def test_c7_3_aggregation_no_match_current_behavior():
    install_wildcards({"garden/flower": ["lily"]})
    for seed in SEEDS:
        out = assert_determinism("__*/nonexistent__", seed)
        assert out == "__*/nonexistent__", (
            f"baseline drift: no-match pattern now returns {out!r}"
        )


def test_c8_1_exact_n_tokens():
    install_wildcards({})
    pool = {"a", "b", "c", "d"}
    for seed in SEEDS:
        out = assert_determinism("{2$$a|b|c|d}", seed)
        tokens = out.split(" ")
        assert len(tokens) == 2, f"expected 2 tokens, got {len(tokens)}: {out!r}"
        for t in tokens:
            assert t in pool, f"token {t!r} not in {pool!r}"
        assert tokens[0] != tokens[1], (
            f"baseline drift: tokens now repeat for n≤pool — {tokens!r}"
        )


def test_c8_2_n_greater_than_pool_size():
    install_wildcards({})
    pool = {"a", "b", "c"}
    for seed in SEEDS:
        out = assert_determinism("{5$$a|b|c}", seed)
        tokens = out.split(" ")
        assert len(tokens) <= 3, f"expected ≤3 tokens, got {len(tokens)}: {out!r}"
        for t in tokens:
            assert t in pool, f"token {t!r} not in {pool!r}"


def test_c9_1_range_boundary():
    install_wildcards({})
    counts_seen = set()
    for seed in range(100):
        out = populate("{1-3$$a|b|c}", seed)
        tokens = [t for t in out.split(" ") if t != ""]
        n = len(tokens)
        assert 1 <= n <= 3, f"token count {n} outside [1,3] for seed={seed}: {out!r}"
        counts_seen.add(n)
    assert counts_seen == {1, 2, 3}, (
        f"range boundary not fully exercised: observed {counts_seen!r}"
    )


def test_c9_2_open_lower_range():
    install_wildcards({})
    for seed in range(30):
        out = populate("{-3$$a|b|c|d}", seed)
        tokens = [t for t in out.split(" ") if t != ""]
        n = len(tokens)
        assert 1 <= n <= 3, (
            f"open-lower range: count {n} outside [1,3] for seed={seed}: {out!r}"
        )


def test_c10_2_prd_custom_separator_example():
    install_wildcards({})
    pool = {"red", "blue", "green"}
    for seed in SEEDS:
        out = assert_determinism("{2$$ and $$red|blue|green}", seed)
        parts = out.split(" and ")
        assert len(parts) == 2, f"expected 2 parts, got {len(parts)}: {out!r}"
        for p in parts:
            assert p in pool, f"part {p!r} not in {pool!r}"
        assert parts[0] != parts[1], f"2-perm had repeats: {out!r}"


def test_c10_3_range_plus_custom_separator():
    install_wildcards({})
    pool = {"apple", "orange", "banana"}
    for seed in range(30):
        out = populate("{1-2$$ or $$apple|orange|banana}", seed)
        parts = [p for p in out.split(" or ") if p]
        assert 1 <= len(parts) <= 2, (
            f"range+sep: {len(parts)} parts for seed={seed}: {out!r}"
        )
        for p in parts:
            assert p in pool, f"part {p!r} not in pool"


def test_c11_1_tutorial_l43_quantify_inside_multi_select():
    install_wildcards({"wildcards": ["a", "b", "c", "d", "e"]})
    pool = {"a", "b", "c", "d", "e"}
    seen_pairs = set()
    for seed in range(50):
        out = populate("{2$$, $$5#__wildcards__}", seed)
        tokens = out.split(", ")
        assert len(tokens) == 2, (
            f"quantify+multi: expected 2 tokens, got {len(tokens)}: {out!r}"
        )
        for t in tokens:
            assert t in pool, f"token {t!r} not in pool"
        seen_pairs.add(tuple(tokens))
    assert len(seen_pairs) >= 5, (
        f"quantify variance degenerate: {len(seen_pairs)} distinct pairs"
    )


def test_c12_1_tutorial_exact_input_output():
    install_wildcards({})
    text = (
        "first {a|b|c} second # not a comment,\n"
        "# this is a comment\n"
        "trailing text\n"
        "newline text"
    )
    expected = {
        "first a second # not a comment, trailing text\nnewline text",
        "first b second # not a comment, trailing text\nnewline text",
        "first c second # not a comment, trailing text\nnewline text",
    }
    for seed in SEEDS:
        out = assert_determinism(text, seed)
        assert out in expected, (
            f"comment-out mismatch seed={seed}:\n  got:      {out!r}\n  expected: {expected!r}"
        )


def test_c13_1_subfolder_path_resolution():
    install_wildcards({"obj/person": ["alice", "bob", "carol"]})
    expected = {"alice", "bob", "carol"}
    for seed in SEEDS:
        out = assert_determinism("__obj/person__", seed)
        assert_membership(out, expected)


def test_c13_2_missing_subfolder_current_behavior():
    install_wildcards({"obj/person": ["alice", "bob", "carol"]})
    for seed in SEEDS:
        out = assert_determinism("__obj/nonexistent__", seed)
        assert out == "__obj/nonexistent__", (
            f"baseline drift: missing subfolder now returns {out!r}"
        )


def test_c13_3_depth_agnostic_fallback():
    install_wildcards({"obj/person": ["alice", "bob", "carol"]})
    expected = {"alice", "bob", "carol"}
    for seed in SEEDS:
        out = assert_determinism("__person__", seed)
        assert_membership(out, expected)


def test_c14_1_transitive_resolution():
    install_wildcards({
        "animal": ["__cat__", "__dog__", "bird"],
        "cat": ["persian", "siamese"],
        "dog": ["labrador", "poodle"],
    })
    expected = {"persian", "siamese", "labrador", "poodle", "bird"}
    for seed in SEEDS:
        out = assert_determinism("__animal__", seed)
        assert_membership(out, expected)
        assert "__cat__" not in out and "__dog__" not in out, (
            f"transitive not fully resolved: {out!r}"
        )


def test_c14_2_transitive_variance():
    install_wildcards({
        "animal": ["__cat__", "__dog__", "bird"],
        "cat": ["persian", "siamese"],
        "dog": ["labrador", "poodle"],
    })
    seen = {populate("__animal__", s) for s in range(20)}
    assert len(seen) >= 3, f"transitive variance degenerate: {seen!r}"


@pytest.mark.timeout(5)
def test_c15_2_iteration_cap_unresolved():
    install_wildcards({"a": ["__b__"], "b": ["__a__"]})
    for seed in SEEDS:
        out = populate("__a__", seed)
        assert out in {"__a__", "__b__"}, (
            f"cap behaviour drift: out={out!r} for seed={seed}"
        )


@pytest.mark.timeout(5)
def test_c15_3_warning_emission(caplog):
    install_wildcards({"a": ["__b__"], "b": ["__a__"]})
    caplog.set_level(logging.WARNING)
    populate("__a__", 0)
    assert "Wildcard expansion limit reached" in caplog.text, (
        "Circular references must emit an expansion-limit warning"
    )
    caplog.clear()
    install_wildcards({"a": ["__b__"], "b": ["resolved"]})
    assert populate("__a__", 0) == "resolved"
    assert not caplog.records, "Successful expansion must not warn"
