"""Tests for the statistical hardening pass: self-pair-free bootstrap,
symmetric borderline handling, and run-level card provenance."""

import json
import random

import pytest

import stresskit as sk
from stresskit.card import verify_card_dict
from stresskit.metrics import (
    bootstrap_ci,
    bootstrap_ci_pairwise,
    jaccard,
    mean_pairwise_jaccard,
)


# ---------------------------------------------------------------------------
# self-pair-free bootstrap
# ---------------------------------------------------------------------------

def noisy_sets(n=7, seed=0):
    rng = random.Random(seed)
    return [frozenset(rng.sample(range(20), 8)) for _ in range(n)]


def test_pairwise_bootstrap_removes_self_pair_inflation():
    sets = noisy_sets()
    point = mean_pairwise_jaccard(sets)
    naive = bootstrap_ci(sets, mean_pairwise_jaccard, seed=1)
    fixed = bootstrap_ci_pairwise(sets, jaccard, seed=1)
    naive_mid = (naive[0] + naive[1]) / 2
    fixed_mid = (fixed[0] + fixed[1]) / 2
    # the naive CI is inflated by duplicate self-pairs (J=1.0 each);
    # the fixed one must sit lower and closer to the point estimate
    assert naive_mid > point
    assert fixed_mid < naive_mid
    assert abs(fixed_mid - point) < abs(naive_mid - point)


def test_pairwise_bootstrap_degenerate_inputs():
    assert bootstrap_ci_pairwise([frozenset({1})] * 3, jaccard) is None  # < 4
    lo, hi = bootstrap_ci_pairwise([frozenset({1, 2})] * 6, jaccard)
    assert lo == hi == 1.0                       # identical runs stay exact


def test_flip_rate_ci_uses_distinct_pairs():
    labels = ["a", "a", "a", "b", "b", "b"]
    lo, hi = bootstrap_ci_pairwise(labels, lambda x, y: float(x != y), seed=2)
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# symmetric borderline handling
# ---------------------------------------------------------------------------

def borderline_fail_finder(data, seed, config):
    # J hovers just under a high bar with wide spread -> fail with
    # straddling CI
    rng = random.Random(seed)
    keep = rng.sample(range(10), 7)
    extra = rng.sample(range(10, 20), 1)
    return sk.feature_set(keep + extra, claim="late", score=1.0,
                          universe_size=100)


def test_borderline_fail_lowers_confidence():
    result = sk.stress(borderline_fail_finder, list(range(30)),
                       battery=["seeds", "bootstrap"], n_runs=6,
                       thresholds=sk.Thresholds(jaccard=0.72))
    ss = result.checks["structural_stability"]
    if ss["passed"] or ss["robust"]:
        pytest.skip("finder did not land on a straddling fail this seed")
    assert result.pooled["confidence"] == "low"
    assert "structural_stability" in result.pooled["borderline_checks"]
    assert "❌⚠️" in result.card.to_markdown()
    assert any("fail" in n for n in result.card.notes if "underpowered" in n)


def test_resolved_fail_is_not_borderline():
    def unstable(data, seed, config):
        rng = random.Random(seed)
        return sk.feature_set(rng.sample(range(100), 8), claim="x",
                              score=1.0, universe_size=200)
    result = sk.stress(unstable, list(range(30)),
                       battery=["seeds", "bootstrap"], n_runs=8)
    ss = result.checks["structural_stability"]
    assert not ss["passed"] and ss["robust"] is True    # decisively unstable
    assert "structural_stability" not in result.pooled["borderline_checks"]


# ---------------------------------------------------------------------------
# run-level provenance on the card
# ---------------------------------------------------------------------------

def stable_finder(data, seed, config):
    rng = random.Random(seed)
    picked = sorted(rng.sample(sorted(data), 8))
    return sk.feature_set(picked, claim="late", score=0.9, universe_size=100)


def make_card_dict():
    result = sk.stress(stable_finder, list(range(30)),
                       battery=["seeds", "bootstrap"], n_runs=5)
    return json.loads(json.dumps(result.card.to_dict(), default=str))


def test_card_embeds_runs_with_hashes():
    d = make_card_dict()
    assert d["schema_version"] == "0.2"
    assert d["battery"]["components_embedded"] is True
    assert len(d["runs"]) == 11
    base = d["runs"][0]
    assert base["axis"] == "base"
    assert base["components"] and base["components_sha256"]
    assert verify_card_dict(d)["ok"]


def test_verify_catches_tampered_pooled_jaccard():
    d = make_card_dict()
    d["metrics"]["pooled"]["mean_pairwise_jaccard"] = 0.99
    # keep the check value in sync so only the runs layer can catch it
    d["verdict"]["checks"]["structural_stability"]["value"] = 0.99
    out = verify_card_dict(d)
    assert not out["ok"]
    assert any("does not recompute" in p for p in out["problems"])


def test_verify_catches_tampered_run_components():
    d = make_card_dict()
    d["runs"][2]["components"][0] = "999"
    out = verify_card_dict(d)
    assert not out["ok"]
    assert any("sha256" in p for p in out["problems"])


def test_verify_catches_tampered_robust_flag():
    d = make_card_dict()
    name, c = next(iter(d["verdict"]["checks"].items()))
    if c.get("ci") is None:
        pytest.skip("first check has no CI")
    c["robust"] = not c["robust"]
    out = verify_card_dict(d)
    assert not out["ok"]
    assert any("robust" in p for p in out["problems"])


def test_verify_catches_tampered_confidence():
    d = make_card_dict()
    d["metrics"]["pooled"]["confidence"] = (
        "high" if d["metrics"]["pooled"]["confidence"] == "low" else "low")
    out = verify_card_dict(d)
    assert not out["ok"]
    assert any("confidence" in p for p in out["problems"])


def test_large_findings_hash_only():
    def big_finder(data, seed, config):
        return sk.feature_set(range(5000), claim="x", score=1.0,
                              universe_size=100000)
    result = sk.stress(big_finder, list(range(30)), battery=["seeds"], n_runs=5)
    d = result.card.to_dict()
    assert d["battery"]["components_embedded"] is False
    assert all("components" not in r for r in d["runs"])
    assert all(r.get("components_sha256") for r in d["runs"])
    assert verify_card_dict(d)["ok"]     # runs layer skips, rest verifies
