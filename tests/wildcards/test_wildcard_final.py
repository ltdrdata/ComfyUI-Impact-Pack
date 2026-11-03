#!/usr/bin/env python3
"""Compare wildcard API results on isolated full-cache and on-demand servers."""
import argparse

import requests


CASES = (
    ("Simple wildcard", "__samples/flower__", 42),
    ("Multiple wildcards", "a __samples/flower__ and a __samples/jewel__", 123),
    ("Dynamic prompt", "{red|blue|green} flower", 456),
    ("Combined", "{beautiful|elegant} __samples/flower__ with {gold|silver} __samples/jewel__", 789),
    ("Multi-select", "{2$$, $$__samples/flower__|rose|tulip|daisy}", 111),
    ("Transitive YAML", "__colors__", 222),
    ("Transitive with context", "a {beautiful|vibrant} __colors__ flower", 333),
)


def run_test(base_url, on_demand):
    base_url = base_url.rstrip("/")
    response = requests.get(f"{base_url}/impact/wildcards/list/loaded", timeout=10)
    response.raise_for_status()
    state = response.json()
    assert state["on_demand_mode"] is on_demand, state
    results = {}
    for description, text, seed in CASES:
        response = requests.post(f"{base_url}/impact/wildcards", json={"text": text, "seed": seed}, timeout=10)
        response.raise_for_status()
        output = response.json()["text"]
        assert isinstance(output, str) and output.strip() and "__" not in output and not output.startswith("ERROR:"), (description, output)
        results[description] = output
    assert set(results) == {description for description, _, _ in CASES}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-url", required=True)
    parser.add_argument("--on-demand-url", required=True)
    args = parser.parse_args()
    results_full = run_test(args.full_url, False)
    results_on_demand = run_test(args.on_demand_url, True)
    assert set(results_full) == set(results_on_demand) == {description for description, _, _ in CASES}
    for description, _, _ in CASES:
        assert results_full[description] == results_on_demand[description], (
            description, results_full[description], results_on_demand[description])
        print(f"PASS: {description}: {results_full[description]}")
    print(f"PASS: {len(CASES)} matching results; both server modes verified")


if __name__ == "__main__":
    main()
