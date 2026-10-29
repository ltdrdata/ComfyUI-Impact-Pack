"""Helpers for the populate regression suite (test_populate_regression.py).

`wildcards.process(text, seed)` is the system-under-test. Helpers here wrap
the call with deterministic setup, provide the common assertions
(determinism / set-membership / binomial distribution within ±sigma), and
offer a single entry point to install fixture data into the module-global
`wildcard_dict` used by process().
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable

from impact.wildcards import process as populate, wildcard_dict


def assert_determinism(text: str, seed: int) -> str:
    """The same text and seed must produce the same output."""
    first = populate(text, seed)
    second = populate(text, seed)
    assert first == second, (
        f"Non-deterministic output for {text!r}, seed={seed}: {first!r} != {second!r}"
    )
    return first


def assert_membership(output: str, expected: Iterable[str]) -> None:
    expected_set = set(expected)
    assert output in expected_set, (
        f"output {output!r} not in expected set (size={len(expected_set)})\n"
        f"  expected: {sorted(expected_set)!r}"
    )


def assert_distribution(
    get_output: Callable[[int], str],
    weights: dict[str, float],
    n: int = 1000,
    sigma: float = 3.0,
) -> Counter:
    """Empirical binomial distribution check with ±sigma tolerance.

    `get_output(i)` must return one of the keys in `weights`. The i-th
    invocation receives a distinct seed so 1000 samples explore the PRNG
    space rather than hammering a single seed.
    """
    counts: Counter = Counter()
    misses = []
    for i in range(n):
        out = get_output(i)
        if out in weights:
            counts[out] += 1
        else:
            misses.append(out)
    assert not misses, (
        f"{len(misses)} outputs fell outside the expected key set.\n"
        f"  first 5 misses: {misses[:5]!r}\n"
        f"  expected keys: {sorted(weights.keys())!r}"
    )

    total_weight = sum(weights.values())
    failures = []
    for key, w in weights.items():
        p = w / total_weight
        expected = n * p
        sd = math.sqrt(n * p * (1.0 - p))
        lower = expected - sigma * sd
        upper = expected + sigma * sd
        actual = counts.get(key, 0)
        if not (lower <= actual <= upper):
            failures.append(
                f"  {key!r}: expected {expected:.1f} ±{sigma*sd:.1f} "
                f"→ [{lower:.1f}, {upper:.1f}], got {actual}"
            )
    assert not failures, (
        f"Distribution outside ±{sigma}σ for {len(failures)} key(s):\n"
        + "\n".join(failures)
        + f"\n  full counts: {dict(counts)}"
    )
    return counts


def install_wildcards(mapping: dict[str, list[str]]) -> None:
    """Clear wildcard_dict and install the provided mapping.

    Tests use this instead of touching the filesystem so each test has an
    isolated fixture. Keys are stored as-given — wildcards.process()
    normalises lookup keys via `wildcard_normalize` (lowercase + space→dash
    + backslash→slash), so callers should provide already-normalised keys.
    """
    wildcard_dict.clear()
    wildcard_dict.update(mapping)
