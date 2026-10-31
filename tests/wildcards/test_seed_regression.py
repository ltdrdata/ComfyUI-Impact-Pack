"""Wildcard seed compatibility and isolation from Python's shared RNG."""
from concurrent.futures import ThreadPoolExecutor
import random
import threading

import pytest

from impact import wildcards as wc


@pytest.fixture(autouse=True)
def isolate_random_state(monkeypatch):
    state = random.getstate()
    monkeypatch.setattr(wc, "wildcard_dict", {"colors": ["red", "green", "blue"]})
    try:
        yield
    finally:
        random.setstate(state)


@pytest.mark.parametrize("seed, unsigned_seed", [
    (-1, 2**64 - 1),
    (-2, 2**64 - 2),
    (-(2**63), 2**63),
    (-(2**64), 0),
])
def test_negative_seed_matches_unsigned_seed(seed, unsigned_seed):
    text = "__colors__ {1-3$$, $$red|green|blue} {hot|cold}"
    expected = wc.process(text, unsigned_seed)
    assert wc.process(text, seed) == expected
    assert wc.process(text, seed) == expected


# Captured before #1224, at 5e7d1cd; preserve the existing sampling sequence.
@pytest.mark.parametrize("seed, expected", [
    (0, "red green, red hot"),
    (1, "blue green, blue hot"),
    (42, "blue blue, green cold"),
    (2**63, "green blue, green cold"),
    (2**64 - 1, "red blue, red cold"),
])
def test_existing_seed_outputs(seed, expected):
    text = "__colors__ {2$$, $$red|green|blue} {hot|cold}"
    assert wc.process(text, seed) == expected


@pytest.mark.parametrize("seed", [None, 0, 42, 2**64 - 1, -1])
def test_global_rng_unchanged(seed):
    random.seed(1224)
    state = random.getstate()
    assert wc.process("__colors__", seed) in {"red", "green", "blue"}
    assert random.getstate() == state


@pytest.mark.parametrize("seed", [None, 42, -1])
def test_global_rng_unchanged_after_error(seed):
    random.seed(1224)
    state = random.getstate()
    with pytest.raises(ZeroDivisionError):
        wc.process("{0::red|0::blue}", seed)
    assert random.getstate() == state


@pytest.mark.parametrize("seed", [None, 42])
def test_concurrent_random_draws_are_preserved(monkeypatch, seed):
    entered = threading.Event()
    resume = threading.Event()
    get_wildcard_dict = wc.get_wildcard_dict

    def paused_lookup():
        entered.set()
        assert resume.wait(5), "Concurrent consumer did not release wildcard lookup"
        return get_wildcard_dict()

    # Hold expansion at its lookup boundary while another thread consumes RNG.
    monkeypatch.setattr(wc, "get_wildcard_dict", paused_lookup)
    random.seed(1224)
    reference = random.Random(1224)
    expected = [reference.randint(0, 1125899906842624) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        expansion = executor.submit(wc.process, "__colors__", seed)
        try:
            assert entered.wait(5), "Wildcard expansion did not reach its lookup"
            during = random.randint(0, 1125899906842624)
        finally:
            resume.set()
        assert expansion.result(timeout=5) in {"red", "green", "blue"}
    after = random.randint(0, 1125899906842624)
    assert [during, after] == expected
