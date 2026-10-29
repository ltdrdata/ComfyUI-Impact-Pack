"""Regression tests for wildcard expansion, sampling, nesting, and iteration limits."""
from __future__ import annotations

import logging
import os
import sys
import tempfile

import pytest

# Use pytest's bootstrap before importing the wildcard module in script mode.
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


# ============================================================================
# Category 1: Basic wildcard __x__
# Tutorial L28 / PRD §2.1, §2.5
# ============================================================================

_FLOWERS = ["rose", "orchid", "iris", "carnation", "lily"]


def test_c1_1_membership_exact_and_pool_filter():
    """AC-1.1: `A __flowers__ B` → `A {x} B`, x ∈ 5-member pool.

    Covers PRD §2.5 pool-filter rule by loading from a real file with an
    embedded comment and a blank line — those must be excluded from the
    selection pool.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    ) as f:
        f.write("rose\norchid\n# not selected\niris\ncarnation\n\nlily\n")
        path = f.name
    try:
        loaded = wc.load_txt_wildcard(path)
        # Filter behaviour (PRD §2.5): comment + blank both excluded.
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
    """AC-1.2: I2 byte-for-byte preservation of Unicode prefix + punctuation suffix."""
    install_wildcards({"flowers": _FLOWERS})
    text = "🌸 __flowers__ !@#\n"
    for seed in SEEDS:
        out = populate(text, seed)
        assert out.startswith("🌸 "), f"prefix lost: {out!r}"
        assert out.endswith(" !@#\n"), f"suffix lost: {out!r}"
        middle = out[len("🌸 "): -len(" !@#\n")]
        assert middle in set(_FLOWERS), f"middle {middle!r} not a flower"


def test_c1_3_variance_over_seeds():
    """AC-1.3: 7 seeds yield ≥3 distinct selections (no degenerate seed collapse)."""
    install_wildcards({"flowers": _FLOWERS})
    seen = {populate("__flowers__", s) for s in SEEDS}
    assert len(seen) >= 3, f"variance degenerate: {seen!r} across {len(SEEDS)} seeds"


def test_c1_4_prd_5_1_getting_started():
    """AC-1.4: PRD §5.1 "Getting Started" — `a beautiful __flower__` ∈ 5-member set."""
    install_wildcards({"flower": _FLOWERS})
    expected = {f"a beautiful {x}" for x in _FLOWERS}
    for seed in SEEDS:
        out = assert_determinism("a beautiful __flower__", seed)
        assert_membership(out, expected)


# ============================================================================
# Category 2: Simple dynamic {a|b|c}
# Tutorial L25-27 / PRD §2.2
# ============================================================================


def test_c2_1_exact_membership():
    """AC-2.1: `X {red|blue|green} Y` → exact membership in 3-element set."""
    install_wildcards({})
    expected = {"X red Y", "X blue Y", "X green Y"}
    for seed in SEEDS:
        out = assert_determinism("X {red|blue|green} Y", seed)
        assert_membership(out, expected)


def test_c2_2_whitespace_in_options_preserved():
    """AC-2.2: whitespace inside options is byte-preserved."""
    install_wildcards({})
    # process() wraps the output — input is purely `{...}` so output ==
    # one of the options verbatim (with leading/trailing spaces).
    expected = {" red apple ", " blue sky "}
    for seed in SEEDS:
        out = assert_determinism("{ red apple | blue sky }", seed)
        assert_membership(out, expected)


def test_c2_3_empty_option_current_behavior():
    """AC-2.3: `{a||b}` — spec-unspecified. Baseline: current impl behavior.

    PRD/tutorial do not document the empty-option branch. Current
    observable: outputs form a subset of {'a', '', 'b'}. This test locks
    the allowed set as baseline without requiring a particular distribution.
    """
    install_wildcards({})
    allowed = {"a", "", "b"}
    for seed in SEEDS:
        out = assert_determinism("{a||b}", seed)
        assert out in allowed, (
            f"baseline violation: {out!r} not in {allowed!r} — spec gap may have shifted"
        )


# ============================================================================
# Category 3: Nested dynamic {a|{d|e|f}|c}
# Tutorial L26 / PRD §2.2 "Unlimited nesting"
# ============================================================================


def test_c3_1_resolvable_membership_one_level():
    """AC-3.1: `{a|{d|e|f}|c}` → ∈ {a, d, e, f, c}, ≥3 distinct across 7 seeds."""
    install_wildcards({})
    expected = {"a", "d", "e", "f", "c"}
    seen = set()
    for seed in SEEDS:
        out = assert_determinism("{a|{d|e|f}|c}", seed)
        assert_membership(out, expected)
        seen.add(out)
    assert len(seen) >= 3, f"only {len(seen)} distinct across 7 seeds: {seen!r}"


def test_c3_2_two_level_nesting():
    """AC-3.2: `{a|{b|{x|y}|d}|c}` → ∈ {a, b, x, y, d, c}."""
    install_wildcards({})
    expected = {"a", "b", "x", "y", "d", "c"}
    for seed in SEEDS:
        out = assert_determinism("{a|{b|{x|y}|d}|c}", seed)
        assert_membership(out, expected)


def test_c3_3_nested_distribution_current_behavior():
    """AC-3.3: baseline — nested `{a|{d|e|f}|c}` distribution.

    Tutorial/PRD do not specify whether nested resolution yields
    outer-uniform-1/3 (parent-picks-nested-then-recurses) or flat-uniform-1/5.
    Baseline: current impl observable via N=1000. A subsequent change to
    the random algorithm will shift the distribution past ±3σ and trip
    this guard. The exact weight values here are empirical (not normative).
    """
    install_wildcards({})

    def get(i):
        return populate("{a|{d|e|f}|c}", i)

    # baseline: current impl behavior — outer-uniform 1/3 × nested-uniform 1/3
    # yields a/c at ~333 each, d/e/f at ~111 each.
    weights = {"a": 3, "c": 3, "d": 1, "e": 1, "f": 1}
    assert_distribution(get, weights, n=1000, sigma=3.0)


# ============================================================================
# Category 4: Compound wildcard + dynamic
# Tutorial L27, L29 / PRD §2.2
# ============================================================================


def test_c4_1_tutorial_l27_example():
    """AC-4.1: tutorial L27 `{blue apple|red {cherry|berry}|green melon}`."""
    install_wildcards({})
    expected = {"blue apple", "red cherry", "red berry", "green melon"}
    for seed in SEEDS:
        out = assert_determinism(
            "{blue apple|red {cherry|berry}|green melon}", seed
        )
        assert_membership(out, expected)


def test_c4_2_tutorial_l29_example():
    """AC-4.2: tutorial L29 compound wildcard + dynamic with 3 wildcards."""
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


# ============================================================================
# Category 5: Weighted {w1::opt1|...}
# Tutorial L30-31 / PRD §2.2 "Weighted Selection"
# ============================================================================


def test_c5_1_weighted_membership():
    """AC-5.1: `{5::red|4::green|7::blue|black}` → ∈ {red, green, blue, black}."""
    install_wildcards({})
    expected = {"red", "green", "blue", "black"}
    for seed in SEEDS:
        out = assert_determinism("{5::red|4::green|7::blue|black}", seed)
        assert_membership(out, expected)


def test_c5_2_weighted_distribution_normalized():
    """AC-5.2: weights `5:4:7:1` → normalised 5/17 : 4/17 : 7/17 : 1/17 (±3σ)."""
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
    """AC-5.3: unweighted option defaults to weight 1 — `{5::red|green|blue}`."""
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
    """AC-5.4: PRD L72 — `{common::10|rare::1}` → equal weights (prefix `::`, not suffix)."""
    install_wildcards({})

    def get(i):
        return populate("{common::10|rare::1}", i)

    # PRD normative: suffix-weight syntax is NOT a recognised weight form.
    # Both options fall back to default weight 1 → 50/50.
    assert_distribution(
        get,
        weights={"common::10": 1, "rare::1": 1},
        n=1000,
        sigma=3.0,
    )


def test_c5_5_decimal_weight_current_behavior():
    """AC-5.5: baseline — `{0.5::a|0.3::b}` current observable.

    PRD §2.2 Limitations flags this as a known parsing-conflict surface.
    Observed impl behavior: sum of decimal weights (0.5+0.3=0.8) falls
    under the `total_prob <= 1` guard at replace_options, which triggers
    `shuffle_all_and_return_pool` instead of weighted single-pick. Output
    is therefore the *entire* pool joined by the default separator (space).
    Locking ∈ {"a b", "b a"} catches future algorithm changes that would
    emit any other form (single-element, different separator, etc.).
    """
    install_wildcards({})
    allowed = {"a b", "b a"}  # baseline: current impl behavior
    for seed in SEEDS:
        out = assert_determinism("{0.5::a|0.3::b}", seed)
        assert out in allowed, (
            f"baseline violation: {out!r} — decimal-weight parsing drift"
        )


def test_c5_6_simple_decimal_acceptance():
    """AC-5.6: PRD L77 — `{0.5::a|b}` accepts simple decimals."""
    install_wildcards({})
    for seed in SEEDS:
        out = assert_determinism("{0.5::a|b}", seed)
        assert_membership(out, {"a", "b"})


# ============================================================================
# Category 6: Case-insensitive wildcard names
# Tutorial L32 / PRD §2.1
# ============================================================================


def test_c6_1_case_insensitive_equivalence():
    """AC-6.1: all case variants resolve identically with same seed."""
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
    """AC-6.2: loaded key is lowercased regardless of file-system casing.

    read_wildcard() normalises the provided key via wildcard_normalize
    (lowercase + path-sep + space→dash). Simulate a file named `Jewel.txt`
    being ingested into the dict under the name `Jewel`, then look it up
    as `__jewel__`.
    """
    install_wildcards({})
    # Simulate the ingestion path: read_wildcard normalises the key.
    wc.read_wildcard("Jewel", ["diamond", "ruby", "sapphire"])
    # The store must hold the normalised key — no uppercase 'Jewel' left.
    assert "Jewel" not in wc.wildcard_dict
    assert "jewel" in wc.wildcard_dict
    for seed in SEEDS:
        out = assert_determinism("__jewel__", seed)
        assert_membership(out, {"diamond", "ruby", "sapphire"})


# ============================================================================
# Category 7: Aggregation pattern */name
# Tutorial L33-34 / PRD §2.1 "aggregation pattern (V4.15.1+)"
# ============================================================================


def test_c7_1_aggregation_union_membership():
    """AC-7.1: `__*/flower__` unions garden/ocean/meadow flower sets."""
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
    """AC-7.2: ≥3 distinct outputs across 20 seeds (union variance)."""
    install_wildcards({
        "garden/flower": ["lily"],
        "ocean/flower": ["coral"],
        "meadow/flower": ["daisy", "poppy"],
    })
    seen = {populate("__*/flower__", s) for s in range(20)}
    assert len(seen) >= 3, f"aggregation variance degenerate: {seen!r}"


def test_c7_3_aggregation_no_match_current_behavior():
    """AC-7.3: baseline — `__*/nonexistent__` leaves the reference intact."""
    install_wildcards({"garden/flower": ["lily"]})
    for seed in SEEDS:
        out = assert_determinism("__*/nonexistent__", seed)
        # baseline: current impl behavior — unchanged when pattern matches
        # zero known keys. Does not emit error.
        assert out == "__*/nonexistent__", (
            f"baseline drift: no-match pattern now returns {out!r}"
        )


# ============================================================================
# Category 8: Multi-select {n$$sel|sel}
# Tutorial L35-38 / PRD §2.2 "Multi-Select"
# ============================================================================


def test_c8_1_exact_n_tokens():
    """AC-8.1: `{2$$a|b|c|d}` → exactly 2 space-separated tokens, each ∈ pool.

    baseline: current impl behavior — the default separator when the
    multi-select form omits `$$sep$$` is a single space (`select_sep = ' '`
    at replace_options), NOT the tutorial-implied `, `. Spec gap tracked.
    """
    install_wildcards({})
    pool = {"a", "b", "c", "d"}
    for seed in SEEDS:
        out = assert_determinism("{2$$a|b|c|d}", seed)
        tokens = out.split(" ")
        assert len(tokens) == 2, f"expected 2 tokens, got {len(tokens)}: {out!r}"
        for t in tokens:
            assert t in pool, f"token {t!r} not in {pool!r}"
        # np.random.choice(..., replace=False) makes tokens distinct for n ≤ pool.
        assert tokens[0] != tokens[1], (
            f"baseline drift: tokens now repeat for n≤pool — {tokens!r}"
        )


def test_c8_2_n_greater_than_pool_size():
    """AC-8.2: `{5$$a|b|c}` — excess ignored, result capped at pool size (3).

    baseline: current impl behavior — tokens joined by default space
    separator (see c8_1 note).
    """
    install_wildcards({})
    pool = {"a", "b", "c"}
    for seed in SEEDS:
        out = assert_determinism("{5$$a|b|c}", seed)
        tokens = out.split(" ")
        assert len(tokens) <= 3, f"expected ≤3 tokens, got {len(tokens)}: {out!r}"
        for t in tokens:
            assert t in pool, f"token {t!r} not in {pool!r}"


def test_c8_3_multi_select_variance():
    """AC-8.3: 100 seeds on `{2$$a|b|c|d}` → ≥4 distinct pair combinations (of 6)."""
    install_wildcards({})
    seen = set()
    for seed in range(100):
        out = populate("{2$$a|b|c|d}", seed)
        seen.add(tuple(sorted(out.split(" "))))
    assert len(seen) >= 4, (
        f"multi-select variance degenerate: {len(seen)} distinct pairs: {seen!r}"
    )


# ============================================================================
# Category 9: Range multi-select
# Tutorial L36-37 / PRD §2.2
# ============================================================================


def test_c9_1_range_boundary():
    """AC-9.1: `{1-3$$a|b|c}` — token count in [1,3] and every count observed.

    baseline: tokens joined by default space separator (see c8_1 note).
    Uses 100 seeds so every count in [1,3] is reliably observed given
    np.random's ~equal draw across the 3-value range.
    """
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
    """AC-9.2: `{-3$$a|b|c|d}` — implicit low=1, count ∈ [1,3], never 0.

    baseline: tokens joined by default space separator (see c8_1 note).
    """
    install_wildcards({})
    for seed in range(30):
        out = populate("{-3$$a|b|c|d}", seed)
        tokens = [t for t in out.split(" ") if t != ""]
        n = len(tokens)
        assert 1 <= n <= 3, (
            f"open-lower range: count {n} outside [1,3] for seed={seed}: {out!r}"
        )


# ============================================================================
# Category 10: Custom separator {n$$ sep $$sel|...}
# Tutorial L39-41 / PRD §2.2
# ============================================================================


def test_c10_1_default_vs_custom_separator():
    """AC-10.1: default vs custom separator semantics.

    baseline: current impl behavior — default multi-select separator is a
    single space (`select_sep = ' '` at replace_options when the
    `count$$sep$$opts` triple is absent). The tutorial L39 suggests `, `
    as the default; that's a **spec gap** noted here. Custom separator
    path (`{n$$ sep $$opts}`) correctly substitutes the explicit literal
    with leading/trailing spaces preserved.
    """
    install_wildcards({})
    # default (space) — tokens separated by one space, no `, ` leakage
    for seed in SEEDS:
        out = populate("{2$$a|b|c}", seed)
        tokens = out.split(" ")
        assert len(tokens) == 2, f"default separator broke token split: {out!r}"
        # baseline: current impl behavior — no comma-separator form.
        assert ", " not in out, f"unexpected `, ` leaked: {out!r}"
    # custom ` and ` literal (incl. leading/trailing space)
    for seed in SEEDS:
        out = populate("{2$$ and $$a|b|c}", seed)
        assert " and " in out, f"custom separator missing: {out!r}"
        assert ", " not in out, f"default `, ` leaked with custom: {out!r}"


def test_c10_2_prd_custom_separator_example():
    """AC-10.2: PRD L87 — 2-permutation of 3 options joined by ` and `."""
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
    """AC-10.3: `{1-2$$ or $$apple|orange|banana}` — count ∈ [1,2] with ` or `."""
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


# ============================================================================
# Category 11: Quantifying wildcard N#__x__
# Tutorial L42-45 / PRD §2.1 "Quantifiers"
# ============================================================================


def test_c11_1_tutorial_l43_quantify_inside_multi_select():
    """AC-11.1: `{2$$, $$5#__wildcards__}` — X, Y each ∈ pool, ≥5 distinct (X,Y)."""
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


def test_c11_2_quantify_alone_current_behavior():
    """AC-11.2: baseline — `3#__x__` alone (no multi-select wrapper)."""
    install_wildcards({"x": ["p", "q", "r"]})
    pool = {"p", "q", "r"}
    for seed in SEEDS:
        out = assert_determinism("3#__x__", seed)
        # baseline: current impl behavior — the 3# quantifier expands to
        # `__x__|__x__|__x__` as an internal pipe group. Without a
        # surrounding `{n$$...}` wrapper, the pipes become literal.
        # Observe: output should contain at least one valid value from
        # the pool, or the literal expanded form — both acceptable baselines.
        assert any(v in out for v in pool) or "|" in out, (
            f"baseline drift: quantify-alone out={out!r}"
        )


# ============================================================================
# Category 12: Comment-out
# Tutorial L46-60 / PRD §2.1 "Comments"
# ============================================================================


def test_c12_1_tutorial_exact_input_output():
    """AC-12.1: tutorial exact input → one of three possible outputs."""
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


def test_c12_2_inline_hash_preserved():
    """AC-12.2: `#` mid-line is literal, not a comment marker."""
    install_wildcards({})
    text = "foo bar # baz\nother"
    # No leading-# line → output preserves inline '#' verbatim.
    for seed in SEEDS:
        out = assert_determinism(text, seed)
        assert "# baz" in out, f"inline # lost: {out!r}"


def test_c12_3_single_space_join():
    """AC-12.3: comment line is replaced by single space joining neighbours."""
    install_wildcards({})
    text = "head\n# comment\ntail"
    for seed in SEEDS:
        out = assert_determinism(text, seed)
        # PRD §2.1: join with single space. Other possibilities (newline,
        # empty string) must NOT be produced.
        assert out == "head tail", f"single-space-join drift: {out!r}"


# ============================================================================
# Category 13: Subfolder wildcards __obj/person__
# Tutorial L194-195 / PRD §2.1
# ============================================================================


def test_c13_1_subfolder_path_resolution():
    """AC-13.1: `__obj/person__` resolves to the pool in that subpath."""
    install_wildcards({"obj/person": ["alice", "bob", "carol"]})
    expected = {"alice", "bob", "carol"}
    for seed in SEEDS:
        out = assert_determinism("__obj/person__", seed)
        assert_membership(out, expected)


def test_c13_2_missing_subfolder_current_behavior():
    """AC-13.2: baseline — `__obj/nonexistent__` leaves the reference unchanged."""
    install_wildcards({"obj/person": ["alice", "bob", "carol"]})
    for seed in SEEDS:
        out = assert_determinism("__obj/nonexistent__", seed)
        # baseline: current impl behavior — no match → unchanged (no error)
        assert out == "__obj/nonexistent__", (
            f"baseline drift: missing subfolder now returns {out!r}"
        )


def test_c13_3_depth_agnostic_fallback():
    """AC-13.3: PRD §2.6 depth-agnostic — `__person__` resolves via `obj/person`."""
    install_wildcards({"obj/person": ["alice", "bob", "carol"]})
    expected = {"alice", "bob", "carol"}
    for seed in SEEDS:
        out = assert_determinism("__person__", seed)
        assert_membership(out, expected)


# ============================================================================
# Category 14: Transitive wildcards
# PRD §2.1 "Transitive wildcards"
# ============================================================================


def test_c14_1_transitive_resolution():
    """AC-14.1: `__animal__` fully resolves via `__cat__` / `__dog__` nesting."""
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
    """AC-14.2: 20 seeds yield ≥3 distinct final outputs."""
    install_wildcards({
        "animal": ["__cat__", "__dog__", "bird"],
        "cat": ["persian", "siamese"],
        "dog": ["labrador", "poodle"],
    })
    seen = {populate("__animal__", s) for s in range(20)}
    assert len(seen) >= 3, f"transitive variance degenerate: {seen!r}"


# ============================================================================
# Category 15: 100-iteration safety limit (Infinite Loop Protection)
# PRD FR-1 / DESIGN §7.3
# ============================================================================


@pytest.mark.timeout(5)
def test_c15_1_termination_under_5s():
    """AC-15.1: circular `__a__` ↔ `__b__` terminates in <5s."""
    install_wildcards({"a": ["__b__"], "b": ["__a__"]})
    for seed in SEEDS:
        populate("__a__", seed)


@pytest.mark.timeout(5)
def test_c15_2_iteration_cap_unresolved():
    """AC-15.2: after 100-iter cap, result is unresolved `__a__` or `__b__`."""
    install_wildcards({"a": ["__b__"], "b": ["__a__"]})
    for seed in SEEDS:
        out = populate("__a__", seed)
        # baseline: current impl behavior — iter cap yields one of the two
        # unresolved forms (depending on where the cap hit).
        assert out in {"__a__", "__b__"}, (
            f"cap behaviour drift: out={out!r} for seed={seed}"
        )


@pytest.mark.timeout(5)
def test_c15_3_warning_emission(caplog):
    """AC-15.3: warn on exhaustion, but not on successful expansion."""
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
