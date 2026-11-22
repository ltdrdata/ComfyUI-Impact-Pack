from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable

from impact.wildcards import process as populate, wildcard_dict


def assert_determinism(text: str, seed: int) -> str:
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
    wildcard_dict.clear()
    wildcard_dict.update(mapping)
